"""SAFE-style fragmentation → CRS self-reaction string ("ECRS").

ECRS encodes a molecule as a single atom-mapped CRS string that captures
*which bonds were broken* to fragment the molecule into chemically meaningful
pieces. The CRS form makes the reconstruction exact (atom maps tell you which
atoms reform which bonds), unlike SAFE's dot-separated fragments where the
reconstruction is implicit.

## Pipeline

For a SMILES `M`:

    1. Choose a fragmentation method: `brics` (rdkit), `safe` (datamol's
       safe-mol library), or `lillymol` (external mol2SAFE C++ binary).
    2. The method produces a set of `(atom_i, atom_j)` bonds to break.
    3. Atom-map every heavy atom of `M` (1..N).
    4. Build an `RWMol` copy of `M`, remove those bonds → the mol now has
       multiple disconnected fragments but preserves atom maps + indices.
    5. Form the atom-mapped reaction `mapped_M >> mapped_fragments`.
    6. Pass that reaction through `CRSwriter` (from the rdkit-CRS fork) to get
       the single ECRS string with `{...}` bond-change annotations.

Round-trip via `CRSreader`, taking the reactant side, stripping atom maps.
Some molecules don't round-trip exactly (typically when sanitisation after
the bond removal changes aromaticity perception or stereo).

## Methods

- **`brics`** (default): rdkit's `Chem.BRICS.FindBRICSBonds`. Pure Python +
  rdkit, no extra install. Fastest path; what most generative-chemistry work
  uses.
- **`safe`**: datamol's safe-mol (`pip install safe-mol`). Same fragmentation
  rules as BRICS in practice but goes through their canonical encoder; we
  recover the bond cuts by diffing the SAFE string against the original mol.
- **`lillymol`**: Ian Watson's mol2SAFE C++ binary in LillyMol
  (https://github.com/IanAWatson/LillyMol). Uses dicer fragment-formation
  rules, processes ~19k SMILES/sec, no dummy atoms. Requires building LillyMol
  separately and pointing `LILLYMOL_MOL2SAFE_BIN` at the binary. Placeholder
  here; emits `NotImplementedError` until the binary path is wired.

## Requires the rdkit-CRS fork

    from rdkit.Chem.CondensedGraphRxn import rdCondensedGraphRxn

This module is in the `rdkit-crs` fork; the stock `pip install rdkit` does
NOT include it. Build the fork (see `rxnclean`'s
`cpp/compile_crsclean.sh`) and run inside an env that has the binding
available (e.g. the `rdkit_build_fb` conda env).
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import BRICS
    from rdkit.Chem.CondensedGraphRxn import rdCondensedGraphRxn as _CRS
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ECRS requires the rdkit-CRS fork "
        "(rdkit.Chem.CondensedGraphRxn). The stock `pip install rdkit` does "
        "NOT include it. See module docstring for the build path."
    ) from e


# ============================================================================
# Fragmentation methods — each returns a list of (atom_i, atom_j) bonds to cut
# ============================================================================

def _bonds_brics(mol: "Chem.Mol") -> List[Tuple[int, int]]:
    """rdkit BRICS: returns the bonds BRICS would cut, by atom-index pair."""
    return [(a, b) for (a, b), _rule in BRICS.FindBRICSBonds(mol)]


def _bonds_safe(mol: "Chem.Mol") -> List[Tuple[int, int]]:
    """datamol's safe-mol: encode + recover bond cuts by diffing.

    The SAFE string is itself valid SMILES (with ring-closure-style
    attachment markers); parsing it back gives a mol with the SAME atoms
    but with the cut bonds replaced by ring-closure-equivalent connectivity.
    We diff the bond set of the original vs the SAFE-parsed mol to recover
    the cut edges. Atom canonical-rank matching is required to align atoms
    between the two parses.
    """
    # Lazy-import; bypass the broken safe.__init__.py via direct submodule load.
    import importlib.util, sys, types
    from pathlib import Path
    try:
        import safe.converter  # might fail due to broken __init__
        encode = safe.converter.encode
    except Exception:
        # Try the direct-load workaround (works even with broken __init__).
        import safe as _safe_pkg  # whatever sys.modules has
        # Find the actual install dir
        for p in sys.path:
            cand = Path(p) / "safe" / "converter.py"
            if cand.is_file():
                spec = importlib.util.spec_from_file_location("safe.converter", cand)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["safe.converter"] = mod
                spec.loader.exec_module(mod)
                encode = mod.encode
                break
        else:
            raise ImportError(
                "safe-mol is needed for method='safe'. "
                "Install with `pip install safe-mol`."
            )

    safe_str = encode(Chem.MolToSmiles(mol))
    safe_mol = Chem.MolFromSmiles(safe_str)
    if safe_mol is None:
        return []
    # Build a (canonical_rank → atom_idx) map for both mols
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    Chem.AssignStereochemistry(safe_mol, cleanIt=True, force=True)
    rank_orig = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    rank_safe = list(Chem.CanonicalRankAtoms(safe_mol, breakTies=True))
    orig_by_rank = {r: i for i, r in enumerate(rank_orig)}
    safe_by_rank = {r: i for i, r in enumerate(rank_safe)}
    if set(orig_by_rank) != set(safe_by_rank):
        # Atoms don't correspond (shouldn't happen for valid SAFE) — give up.
        return []
    orig_bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
    safe_bonds_orig_idx = set()
    for b in safe_mol.GetBonds():
        i_safe, j_safe = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        i_orig = orig_by_rank[rank_safe[i_safe]]
        j_orig = orig_by_rank[rank_safe[j_safe]]
        safe_bonds_orig_idx.add(tuple(sorted((i_orig, j_orig))))
    return sorted(orig_bonds - safe_bonds_orig_idx)


def _bonds_lillymol(mol: "Chem.Mol") -> List[Tuple[int, int]]:
    """LillyMol mol2SAFE binary: requires LILLYMOL_MOL2SAFE_BIN env var pointing
    at the compiled `mol2SAFE` executable.

    Placeholder: writes the SMILES to a tempfile, invokes mol2SAFE, then
    uses the same SAFE-diff strategy as `_bonds_safe` to recover bond cuts.
    """
    binary = os.environ.get("LILLYMOL_MOL2SAFE_BIN")
    if not binary or not os.path.isfile(binary):
        raise NotImplementedError(
            "method='lillymol' requires the LillyMol mol2SAFE binary. "
            "Build LillyMol (https://github.com/IanAWatson/LillyMol) and set "
            "LILLYMOL_MOL2SAFE_BIN to the absolute path of the compiled "
            "`mol2SAFE` executable. See "
            "https://github.com/IanAWatson/LillyMol/blob/bazel_version_float/docs/Molecule_Tools/SAFE.md"
        )
    smi = Chem.MolToSmiles(mol)
    proc = subprocess.run(
        [binary, "-"],
        input=f"{smi}\n", text=True, capture_output=True, check=True,
    )
    safe_lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if not safe_lines:
        return []
    safe_str = safe_lines[-1].split()[0]
    safe_mol = Chem.MolFromSmiles(safe_str)
    if safe_mol is None:
        return []
    # Reuse the canonical-rank diff approach from _bonds_safe.
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    Chem.AssignStereochemistry(safe_mol, cleanIt=True, force=True)
    rank_orig = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    rank_safe = list(Chem.CanonicalRankAtoms(safe_mol, breakTies=True))
    orig_by_rank = {r: i for i, r in enumerate(rank_orig)}
    if set(orig_by_rank) != set(rank_safe):
        return []
    orig_bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
    safe_bonds_orig_idx = set()
    for b in safe_mol.GetBonds():
        i_safe, j_safe = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        i_orig = orig_by_rank[rank_safe[i_safe]]
        j_orig = orig_by_rank[rank_safe[j_safe]]
        safe_bonds_orig_idx.add(tuple(sorted((i_orig, j_orig))))
    return sorted(orig_bonds - safe_bonds_orig_idx)


METHODS = {
    "brics": _bonds_brics,
    "safe": _bonds_safe,
    "lillymol": _bonds_lillymol,
}


# ============================================================================
# Main encode/decode
# ============================================================================

@dataclass
class ECRSResult:
    ecrs: Optional[str]            # encoded string (None if encoding failed)
    status: str                    # 'ok' | 'no_bonds' | 'parse_err' | 'sanitize_err' | 'crs_err:<...>'
    method: str = ""
    mapped_reactant: Optional[str] = None
    mapped_product: Optional[str] = None


def mol_to_ecrs(smi: str, method: str = "brics", dearomatize: bool = True) -> ECRSResult:
    """SMILES → ECRS (single-string self-reaction).

    method: 'brics' | 'safe' | 'lillymol' — see module docstring.

    dearomatize: if True (default), Kekulize the molecule with
        `clearAromaticFlags=True` BEFORE removing bonds, and pass
        `aromatize=False` to CRSwriter. This prevents the "can't kekulize"
        sanitisation errors you get when an aromatic-ring bond is BRICS-
        cut (the partially-broken ring no longer satisfies aromaticity).
        The reverse perception is done in `ecrs_to_smi` via `MolFromSmiles`
        which re-perceives aromaticity by default.
        Set to False to use the aromatic form directly (faster but loses
        ~10% of mols to kekulization errors on chembl_9k).

    Returns an `ECRSResult` whose `ecrs` is `None` if the pipeline fails.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choices: {list(METHODS)}")

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ECRSResult(None, "parse_err", method)

    # 1. Find BRICS bonds FIRST on the aromatic form (the rules target
    # aryl-aryl / aryl-heteroatom etc. that vanish in Kekulé form).
    try:
        bonds = METHODS[method](mol)
    except NotImplementedError:
        raise
    except Exception as e:
        return ECRSResult(None, f"frag_err:{type(e).__name__}", method)
    if not bonds:
        return ECRSResult(None, "no_bonds", method)

    # 2. Now optionally Kekulize so the upcoming bond removal doesn't leave
    # a half-broken aromatic ring (the source of `sanitize_err` failures).
    if dearomatize:
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            return ECRSResult(None, "kekulize_err", method)

    # 3. Atom-map every heavy atom
    for i, atom in enumerate(mol.GetAtoms(), start=1):
        atom.SetAtomMapNum(i)
    # kekuleSmiles=True (when dearomatized) so the SMILES we feed to CRSwriter
    # matches the Kekulé form internally; otherwise rdkit re-emits aromatic
    # lowercase and CRSwriter sees a mismatch vs. the bond list we're about
    # to break.
    reactant_smi = Chem.MolToSmiles(mol, canonical=False, kekuleSmiles=dearomatize)

    # 3. Remove those bonds → disconnected fragments
    em = Chem.RWMol(mol)
    for a, b in bonds:
        if em.GetBondBetweenAtoms(a, b) is not None:
            em.RemoveBond(a, b)
    broken = em.GetMol()
    try:
        # In dearomatize mode the input was Kekulized, so the broken mol is
        # already in a valence-consistent state — partial sanitisation is
        # enough (don't re-perceive aromaticity here; CRSwriter handles it).
        if dearomatize:
            Chem.SanitizeMol(broken, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                                  Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                                                  Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                                  Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                                  Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        else:
            Chem.SanitizeMol(broken)
    except Exception:
        return ECRSResult(None, "sanitize_err", method, reactant_smi, None)
    product_smi = Chem.MolToSmiles(broken, canonical=False, kekuleSmiles=dearomatize)

    # 4. CRSwriter on the atom-mapped reaction. aromatize=False when we
    # dearomatized so it doesn't re-perceive (which can fail on broken rings).
    rxn = f"{reactant_smi}>>{product_smi}"
    try:
        ecrs = _CRS.CRSwriter(rxn, aromatize=not dearomatize)
    except Exception as e:
        return ECRSResult(None, f"crs_err:{type(e).__name__}", method, reactant_smi, product_smi)
    return ECRSResult(ecrs, "ok", method, reactant_smi, product_smi)


# CRS bond-change annotation format: `{XY}` where X, Y ∈ {'-', '=', '#', ':', '!'}
# encode the reactant and product bond orders (`!` = no bond). Optionally
# followed by zero or more `{int}` H-count delta annotations per atom.
# To reconstruct the REACTANT SMILES we keep the X char as the bond order
# (dropping `-` and `:` which are implicit in SMILES).
_BOND_CHARS = r"\-=#:!"
_ANNOT_RE = re.compile(rf"\{{([{_BOND_CHARS}])([{_BOND_CHARS}])\}}(?:\{{-?\d+\}})*")


def ecrs_to_smi(ecrs: str, method: str = "strip") -> Optional[str]:
    """Decode an ECRS string back to the canonical SMILES of the parent mol.

    Two reconstruction strategies (chosen via `method`):

    - `"strip"` (default, recommended): replace every `{XY}{Hr}{Hp}` bond-
      change annotation in the ECRS string with the REACTANT bond character
      `X` (or empty for `-` / `:` which are implicit in SMILES). What
      remains is the Kekulé SMILES of the reactant, parseable by RDKit
      which re-perceives aromaticity. Bypasses a known H-delta bug in the
      rdkit-CRS `CRSreader` (it can mis-apply H counts and produce
      hypervalent atoms in the reconstructed reactant).

      Crucially, we preserve `=` and `#` bond orders that the naive
      `re.sub(r'\\{[^}]*\\}', '', ecrs)` would drop (and thus turn a
      reactant double bond into a single bond, breaking branched / ring
      molecules — observed in chembl_9k).

    - `"crsreader"`: use the official `CRSreader` then take the reactant
      side. Kept for parity / debugging.

    Returns `None` if the SMILES is unparseable. Atom maps are stripped.

    Round-trip rate on the first 200 chembl_9k_organic molecules:
        - strip     : 184/189 = 97.4% (5 fails are CRSwriter-side mis-renders)
        - crsreader : 141/189 = 74.6% (H-delta bug in C++ reader)
    """
    if method == "strip":
        def repl(m: re.Match) -> str:
            x = m.group(1)
            if x in ("-", ":"):
                return ""  # implicit bond order in SMILES
            if x == "!":
                return ""  # no bond on reactant side shouldn't appear; defensive
            return x       # '=' or '#' kept explicit
        reactant = _ANNOT_RE.sub(repl, ecrs)
    elif method == "crsreader":
        try:
            rxn = _CRS.CRSreader(ecrs)
        except Exception:
            return None
        reactant = rxn.split(">>", 1)[0]
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'strip' or 'crsreader'.")

    m = Chem.MolFromSmiles(reactant)
    if m is None:
        return None
    for atom in m.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def is_roundtrip_equivalent(smi: str, method: str = "brics",
                            dearomatize: bool = True) -> bool:
    """Convenience: encode + decode + compare canonical forms."""
    out = mol_to_ecrs(smi, method=method, dearomatize=dearomatize)
    if out.ecrs is None:
        return False
    back = ecrs_to_smi(out.ecrs)
    if back is None:
        return False
    try:
        return Chem.CanonSmiles(smi) == Chem.CanonSmiles(back)
    except Exception:
        return False
