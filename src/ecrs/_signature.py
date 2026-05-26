"""Radial reaction signature extraction.

Lightweight (alpha) port of CRS ``getCRSsignature`` (Code/GraphMol/
CondensedGraphRxn/CondensedGraphRxn.cpp:988-1056). Two differences from
the C++ original:

1. We do *not* require the patched RDKit ``BondType::CRSXX`` enum values.
   Instead of encoding the bond change inside a single SMILES via the
   ``{r-p}`` curly-bond glyphs, we emit two canonical SMILES strings
   separated by ``>>`` (reactant_env >> product_env). The bond change is
   recoverable by diffing the two strings.

2. The "important" set is computed independently for the reactant and the
   product mols (since they're separate RDKit objects in stock RDKit),
   not on a single CRS-encoded combined mol.

The clustering power should match: two reactions of the same template
will produce the same (react_env, prod_env) pair regardless of the
peripheral structure beyond the chosen radius.
"""
import os
import sys
from typing import Iterable, Optional

from rdkit import Chem

from ._dative_neighbor import remove_disconnected_parts_using_matrix


# ---------------------------------------------------------------------------
# Optional CRS remapping bridge
# ---------------------------------------------------------------------------
# The CRS rdkit fork ships RXNCompleteMapping — a remapper that fills in
# missing atom-map numbers on partially-mapped reactions (e.g., where the
# Boc reagent atoms are unmapped on the reactant side but the Boc fragment
# appears mapped on the product). Without remapping, find_reaction_centers
# can't see those bond changes (mapped N to unmapped C(=O)) and returns no
# centers, producing an empty signature. With remapping, ~22% of N-Boc
# protections that previously returned empty get a real signature.
#
# This bridge is optional: if the rdkit-crs-backup install is on disk and
# loadable via PYTHONPATH/DYLD env vars, we use it; otherwise we fall back
# to the un-remapped path.

_CRS_PATH = "/Users/tgg/Github/rdkit-crs-backup"
_crs_module = None


def _load_crs_module():
    """Return the rdCondensedGraphRxn module if available, else None.

    Cached on first call. Side-effects sys.path so the CRS rdkit shadows
    the stock rdkit only on first import attempt; if anything fails, falls
    back to None silently and signature() works without CRS.
    """
    global _crs_module
    if _crs_module is not None:
        return _crs_module if _crs_module is not False else None
    if not os.path.isdir(_CRS_PATH):
        _crs_module = False
        return None
    try:
        # Append (not insert) so we don't override the user's primary rdkit
        # for the rest of the session — we only want the CRS submodule.
        if _CRS_PATH not in sys.path:
            sys.path.append(_CRS_PATH)
        # DYLD_FALLBACK_LIBRARY_PATH is read at process start; setting it
        # post-fork only helps subprocesses. For dlopen() inside Python the
        # rpath embedded in the .so usually works on macOS; if not, the
        # caller has to set DYLD_FALLBACK_LIBRARY_PATH before launching python.
        from rdkit.Chem.CondensedGraphRxn import rdCondensedGraphRxn  # type: ignore
        _crs_module = rdCondensedGraphRxn
        return rdCondensedGraphRxn
    except Exception:
        _crs_module = False
        return None


def _try_complete_mapping(rxn: str) -> str:
    """Run CRS RXNCompleteMapping if available; return the remapped reaction
    or the original on any error / when CRS is not installed.
    """
    crs = _load_crs_module()
    if crs is None:
        return rxn
    try:
        out = crs.RXNCompleteMapping(rxn, debug=False, addleavinggroups=False)
        if out and ">" in out and "DuplicateATNUM" not in out:
            return out
    except Exception:
        pass
    return rxn


def find_reaction_centers(rmol, pmol) -> set:
    """Return the set of atom map numbers whose local environment changed
    between reactant and product.

    Catches every kind of bond change at a mapped atom:

    1. Mapped-to-mapped bond formed or broken (the only case caught by
       :func:`ecrs.identify_changed_bonds`).
    2. Mapped-to-unmapped bond formed or broken — i.e., bond to a leaving
       or incoming fragment. This is the common case for protecting-group
       removal (Boc/Cbz/Ac deprotection), nucleophilic substitution with
       an unmapped halide, etc.
    3. Bond-order change between two mapped atoms (single → double, etc.).

    Returns map numbers (not atom indices) so callers can resolve them
    on either side without bookkeeping.
    """
    r_atoms = {a.GetAtomMapNum(): a for a in rmol.GetAtoms() if a.GetAtomMapNum() > 0}
    p_atoms = {a.GetAtomMapNum(): a for a in pmol.GetAtoms() if a.GetAtomMapNum() > 0}
    common = set(r_atoms) & set(p_atoms)

    def unmapped_neighbor_signature(atom):
        out = []
        own = atom.GetOwningMol()
        for n in atom.GetNeighbors():
            if n.GetAtomMapNum() == 0:
                bond = own.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx())
                out.append((n.GetAtomicNum(), bond.GetBondType()))
        return sorted(out)

    centers = set()
    for m in common:
        ra = r_atoms[m]
        pa = p_atoms[m]

        r_mapped_nbrs = {n.GetAtomMapNum() for n in ra.GetNeighbors() if n.GetAtomMapNum() > 0}
        p_mapped_nbrs = {n.GetAtomMapNum() for n in pa.GetNeighbors() if n.GetAtomMapNum() > 0}
        if r_mapped_nbrs != p_mapped_nbrs:
            centers.add(m)
            continue

        if unmapped_neighbor_signature(ra) != unmapped_neighbor_signature(pa):
            centers.add(m)
            continue

        # Same set of mapped neighbors on both sides; check for bond-order changes.
        for n in ra.GetNeighbors():
            mn = n.GetAtomMapNum()
            if mn <= 0 or mn not in p_atoms:
                continue
            rbond = rmol.GetBondBetweenAtoms(ra.GetIdx(), n.GetIdx())
            pbond = pmol.GetBondBetweenAtoms(pa.GetIdx(), p_atoms[mn].GetIdx())
            if rbond is not None and pbond is not None \
                    and rbond.GetBondType() != pbond.GetBondType():
                centers.add(m)
                centers.add(mn)
                break

    return centers


def _bfs_atoms(mol, seed_idxs, radius: int) -> set:
    """Return atom indices reachable from any seed within ``radius`` bonds.

    Mirrors CRS ``BFS`` (CondensedGraphRxn.cpp:924-958). Recursive in the
    original; we use an iterative shell expansion here.
    """
    if radius < 0:
        return set()
    visited = set(seed_idxs)
    if radius == 0:
        return visited
    frontier = set(seed_idxs)
    for _ in range(radius):
        next_frontier = set()
        for ai in frontier:
            for nb in mol.GetAtomWithIdx(ai).GetNeighbors():
                ni = nb.GetIdx()
                if ni not in visited:
                    next_frontier.add(ni)
        if not next_frontier:
            break
        visited |= next_frontier
        frontier = next_frontier
    return visited


def _ring_extension(mol, important: set, changed_atom_idxs: Iterable[int]) -> set:
    """Mirror CRS ``addAtomRingCRSIdx`` (CondensedGraphRxn.cpp:960-986).

    If any ring contains a changed-bond atom, include the entire ring's
    atom set in the important set. This catches ring-formation /
    ring-opening templates whose ring atoms would otherwise be at the
    radius boundary.
    """
    # Ring info on RWMols (post BatchEdit / mutation) is sometimes flagged
    # uninitialized; force perception unconditionally so GetRingInfo() can't
    # raise the "RingInfo not initialized" precondition violation.
    Chem.GetSSSR(mol)
    ri = mol.GetRingInfo()
    out = set(important)
    changed_set = set(changed_atom_idxs)
    for ring in ri.AtomRings():
        if any(ai in changed_set for ai in ring):
            out.update(ring)
    return out


def _extract_submol_smiles(mol, important_atoms: set, strip_atom_maps: bool = True) -> str:
    """Build a canonical SMILES from the submol containing only the
    atoms in ``important_atoms``. Atom map numbers are stripped by
    default so the signature is template-invariant.
    """
    if not important_atoms:
        return ""
    rw = Chem.RWMol(mol)
    rw.BeginBatchEdit()
    for ai in range(mol.GetNumAtoms()):
        if ai not in important_atoms:
            rw.RemoveAtom(ai)
    rw.CommitBatchEdit()
    if strip_atom_maps:
        for a in rw.GetAtoms():
            a.SetAtomMapNum(0)
    try:
        return Chem.MolToSmiles(rw.GetMol(), canonical=True)
    except Exception:
        return Chem.MolToSmiles(rw.GetMol(), canonical=False)


def reaction_signature(rxn: str,
                       radius: int = 1,
                       include_ring_change: bool = True,
                       keep_atom_maps_on_centers: bool = False,
                       complete_mapping: bool = True) -> str:
    """Return the radial reaction signature for clustering / templating.

    Args:
        rxn: atom-mapped reaction SMILES (``reactants[>reagents]>products``).
        radius: how many bond shells to include around each changed-bond
            endpoint. ``0`` = bare reaction centers; ``1`` = +immediate
            neighbors; ``2`` or ``3`` typical for NameRXN-level granularity.
        include_ring_change: if True (default, mirrors CRS), any ring
            touched by a changed bond contributes its full atom set.
        keep_atom_maps_on_centers: if True, atom map numbers on the
            reaction-center atoms are preserved in the output (useful
            for debugging / linking back). Default False = stripped.
        complete_mapping: if True (default), run CRS ``RXNCompleteMapping``
            to fill in missing atom-map numbers before signature
            extraction. This catches partially-mapped reactions like
            Boc-protection where the Boc reagent atoms are unmapped.
            Silently no-ops if rdkit-crs-backup is not installed.

    Returns:
        ``"{reactant_env_smiles}>>{product_env_smiles}"``, canonicalized.
        Empty string if the reaction has no changed bonds (identity or
        unmapped input).
    """
    if not rxn or ">" not in rxn:
        return ""
    if complete_mapping:
        rxn = _try_complete_mapping(rxn)
    parts = rxn.split(">")
    if len(parts) < 2:
        return ""
    r_smi = parts[0]
    p_smi = parts[-1]

    rm_raw = Chem.MolFromSmiles(r_smi)
    pm_raw = Chem.MolFromSmiles(p_smi)
    if rm_raw is None or pm_raw is None:
        return ""

    # Drop disconnected unmapped fragments (matches ecrs's whole-pipeline
    # convention) so the signature isn't polluted by spectators.
    rm_clean, _ = remove_disconnected_parts_using_matrix(rm_raw)
    pm_clean, _ = remove_disconnected_parts_using_matrix(pm_raw)

    center_mapnums = find_reaction_centers(rm_clean, pm_clean)
    if not center_mapnums:
        return ""

    def _side_signature(mol) -> str:
        seeds = [a.GetIdx() for a in mol.GetAtoms()
                 if a.GetAtomMapNum() in center_mapnums]
        if not seeds:
            return ""
        important = _bfs_atoms(mol, seeds, radius)
        if include_ring_change:
            important = _ring_extension(mol, important, seeds)
        return _extract_submol_smiles(
            mol, important,
            strip_atom_maps=not keep_atom_maps_on_centers,
        )

    return f"{_side_signature(rm_clean)}>>{_side_signature(pm_clean)}"


def reaction_signatures(rxn: str,
                        radii: Iterable[int] = (0, 1, 2, 3),
                        include_ring_change: bool = True) -> dict:
    """Multi-radius variant — returns ``{radius: signature}``.

    Useful for hierarchical clustering: r=0 is the most abstract (bond-
    change-only) key, r=3 captures the immediate chemical environment.
    """
    return {r: reaction_signature(rxn, radius=r, include_ring_change=include_ring_change)
            for r in radii}
