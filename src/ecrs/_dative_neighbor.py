"""Dative-bond detection and atom-mapping cleanup for reaction SMILES.

Internal module — most consumers should import the public names from
:mod:`ecrs`.
"""
import hashlib
import itertools
from typing import NamedTuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class CleanedReaction(NamedTuple):
    """Result of running ecrs on a reaction SMILES.

    Attributes:
        fictive: Dative-merged single-graph SMILES — fragments that were
            disconnected in the input are joined by virtual dative bonds
            (`<-`) wherever a chemistry-plausible donor-acceptor pair
            spans the would-be merge point. Intended for graph-neural-net
            consumers that need a single connected graph per reaction
            side. May not round-trip through stock RDKit's parser because
            datives count toward valence; myrdkit's patched Bond.cpp
            treats them as 0-contribution on both sides.
        true: Original-valence disconnected SMILES — same atom mapping
            as `fictive`, but without the ghost dative bonds. This is
            the form used to reconstitute the actual reactant and
            product molecules. Always parses through any RDKit.
        mode: ``"ORIGINAL"`` if the forward pass produced consistent
            mappings; ``"WORKAROUND"`` if reactant/product had to be
            swapped to recover consistency (output is then swapped back
            to the original direction).
        leaving_groups: SMILES of fragments in the reactant that have
            no analog in the product (atoms unmapped after global
            atom-property propagation). Empty when the reaction has
            full atom coverage.
        incoming_groups: Symmetric: SMILES of fragments in the product
            with no reactant analog (rare in well-curated data).
    """

    fictive: str
    true: str
    mode: str
    leaving_groups: list
    incoming_groups: list

# Atoms with lone pairs typically available for dative donation.
# Group 15 (N, P, As, Sb, Bi), Group 16 (O, S, Se, Te), Group 17 (halides).
LONE_PAIR_DONORS = frozenset({7, 8, 9, 15, 16, 17, 33, 34, 35, 51, 52, 53, 83})

# Atoms that are typically electrophilic / electron-deficient and can accept
# a lone pair: Group 13 (B, Al, Ga), Group 14 (C, Si, Ge), and transition
# metals (4th, 5th, 6th period d-block).
ELECTROPHILIC_ACCEPTORS = (
    frozenset({5, 6, 13, 14, 31, 32})
    | frozenset(range(21, 31))   # Sc..Zn
    | frozenset(range(39, 49))   # Y..Cd
    | frozenset(range(72, 81))   # Hf..Hg
)


def is_potential_donor(atom):
    """True if `atom` has a lone pair available to donate."""
    if atom.GetAtomicNum() not in LONE_PAIR_DONORS:
        return False
    # Positive formal charge depletes lone pairs (e.g., R4N+ has none).
    return atom.GetFormalCharge() <= 0


def is_potential_acceptor(atom):
    """True if `atom` is electrophilic / electron-deficient enough to accept a lone pair."""
    # Cations of any element accept.
    if atom.GetFormalCharge() > 0:
        return True
    return atom.GetAtomicNum() in ELECTROPHILIC_ACCEPTORS


def dative_donor_acceptor(a, b):
    """If atoms `a` and `b` form a chemically plausible dative pair, return
    ``(donor, acceptor)``; otherwise ``None``.

    Both orientations are tested. Ambiguous pairs (where both atoms could be
    donors *or* both could be acceptors) are rejected — the rule fires only
    when the chemistry has a clear direction (e.g., N→C, O→B).
    """
    a_donor = is_potential_donor(a)
    a_acc = is_potential_acceptor(a)
    b_donor = is_potential_donor(b)
    b_acc = is_potential_acceptor(b)

    a_only_donor = a_donor and not a_acc
    b_only_donor = b_donor and not b_acc
    a_only_acc = a_acc and not a_donor
    b_only_acc = b_acc and not b_donor

    if a_only_donor and b_only_acc:
        return (a, b)
    if b_only_donor and a_only_acc:
        return (b, a)
    return None


def get_atom_properties(atom):
    propd = {
        "stereo": atom.GetChiralTag(),
        "charge": atom.GetFormalCharge(),
        "numHs": atom.GetTotalNumHs(),
        "hybridization": atom.GetHybridization(),
        "isAromatic": atom.GetIsAromatic(),
        "atomicNum": atom.GetAtomicNum(),
        "isotope": atom.GetIsotope(),
        "degree": atom.GetDegree(),
        "implicitValence": atom.GetValence(Chem.ValenceType.IMPLICIT),
        "explicitValence": atom.GetValence(Chem.ValenceType.EXPLICIT),
        "numRadicalElectrons": atom.GetNumRadicalElectrons(),
        "totalDegree": atom.GetTotalDegree(),
        "totalValence": atom.GetTotalValence(),
    }
    return hashlib.md5(str(propd).encode(), usedforsecurity=False).hexdigest()


def identify_changed_bonds(reactant, product, atoms_to_keep):
    """Find bonds present on one side but absent on the other, between two
    atoms that are mapped on both sides.

    Only neighbors with non-zero map numbers are considered. Without that
    restriction, the next() lookup below would match arbitrary unmapped
    atoms (all share map num 0) and produce nonsensical correspondences.
    """
    changed_bonds = []
    for r_atom in reactant.GetAtoms():
        if r_atom.GetIdx() not in atoms_to_keep:
            continue
        map_num = r_atom.GetAtomMapNum()
        if map_num <= 0:
            continue
        p_atom = next((a for a in product.GetAtoms() if a.GetAtomMapNum() == map_num), None)
        if p_atom is None:
            continue

        r_neighbors = list(r_atom.GetNeighbors())
        p_neighbors = list(p_atom.GetNeighbors())

        for r_nbr in r_neighbors:
            if r_nbr.GetAtomMapNum() <= 0:
                continue
            r_bond = reactant.GetBondBetweenAtoms(r_atom.GetIdx(), r_nbr.GetIdx())
            p_nbr = next((a for a in p_neighbors if a.GetAtomMapNum() == r_nbr.GetAtomMapNum()), None)
            p_bond = product.GetBondBetweenAtoms(p_atom.GetIdx(), p_nbr.GetIdx()) if p_nbr else None
            if not p_bond and r_bond:
                changed_bonds.append((r_atom, r_nbr, 'product'))

        for p_nbr in p_neighbors:
            if p_nbr.GetAtomMapNum() <= 0:
                continue
            p_bond = product.GetBondBetweenAtoms(p_atom.GetIdx(), p_nbr.GetIdx())
            r_nbr = next((a for a in r_neighbors if a.GetAtomMapNum() == p_nbr.GetAtomMapNum()), None)
            r_bond = reactant.GetBondBetweenAtoms(r_atom.GetIdx(), r_nbr.GetIdx()) if r_nbr else None
            if not r_bond and p_bond:
                changed_bonds.append((p_atom, p_nbr, 'reactant'))
    return changed_bonds


def add_atom_mapping_to_neighbors(reactant, product, changed_bonds):
    max_atom_map_num = max(
        atom.GetAtomMapNum()
        for atom in itertools.chain(reactant.GetAtoms(), product.GetAtoms())
        if atom.GetAtomMapNum() > 0
    )
    new_atom_map_num = max_atom_map_num + 1

    # Work on copies so callers aren't surprised by mutation
    _reactant = Chem.RWMol(reactant)
    _product = Chem.RWMol(product)

    n_product_atoms = _product.GetNumAtoms()
    for atom1, _atom2, _context in changed_bonds:
        _atom1 = _reactant.GetAtomWithIdx(atom1.GetIdx())
        for neighbor in _atom1.GetNeighbors():
            if neighbor.GetAtomMapNum() != 0:
                continue
            # Reactant and product can differ in atom count after disconnected-
            # fragment cleanup. We can only correspond unmapped neighbors when
            # both sides share the index; otherwise skip rather than crash.
            if neighbor.GetIdx() >= n_product_atoms:
                continue
            p_neighbor = _product.GetAtomWithIdx(neighbor.GetIdx())
            if get_atom_properties(neighbor) == get_atom_properties(p_neighbor):
                neighbor.SetAtomMapNum(new_atom_map_num)
                p_neighbor.SetAtomMapNum(new_atom_map_num)
                new_atom_map_num += 1

    return _reactant, _product


def _atom_by_mapnum(mol, map_num):
    """Look up an atom in `mol` by its atom map number, or None if absent."""
    if map_num <= 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            return atom
    return None


def check_and_add_dative_bond(reactant, product, atoms_to_keep):
    """For each candidate changed bond between mapped atoms, add a dative
    bond on the side missing the bond — but only when the two atoms form a
    chemically plausible donor/acceptor pair.

    The dative bond here is a *graph manipulation*, not a real chemical
    object: it stitches what would otherwise be two disconnected fragments
    into a single graph so atom-mapping consumers don't have to special-case
    "the new C-N bond came from nowhere" reactions. We still gate on
    donor/acceptor chemistry so the ghost bond at least lands between atoms
    that *could* plausibly form one (N→C, O→B, halide→carbocation, etc.) —
    not arbitrarily between two carbons or two donors.

    The atoms in `changed_bonds` belong to the *opposite* mol from the one
    being edited (see identify_changed_bonds), so we re-resolve them in the
    target mol by atom map number, never by raw index — reactant and product
    can have different atom counts after disconnected-fragment cleanup.

    Direction: donor as begin, acceptor as end, matching RDKit's
    `BondType::DATIVE` convention.
    """
    changed_bonds = identify_changed_bonds(reactant, product, atoms_to_keep)
    reactant, product = add_atom_mapping_to_neighbors(reactant, product, changed_bonds)

    seen = set()  # avoid double-adding the same dative

    for atom1, atom2, context in changed_bonds:
        target = reactant if context == 'reactant' else product
        map1 = atom1.GetAtomMapNum()
        map2 = atom2.GetAtomMapNum()
        if map1 == 0 or map2 == 0:
            continue

        target_a = _atom_by_mapnum(target, map1)
        target_b = _atom_by_mapnum(target, map2)
        if target_a is None or target_b is None:
            continue

        pair = dative_donor_acceptor(target_a, target_b)
        if pair is None:
            continue
        donor, acceptor = pair

        # Dedup: same dative on the same side from both sweep directions.
        key = (context, min(map1, map2), max(map1, map2))
        if key in seen:
            continue
        seen.add(key)

        if target.GetBondBetweenAtoms(donor.GetIdx(), acceptor.GetIdx()) is not None:
            continue

        rw = Chem.RWMol(target)
        rw.AddBond(donor.GetIdx(), acceptor.GetIdx(), order=rdchem.BondType.DATIVE)
        if context == 'reactant':
            reactant = rw.GetMol()
        else:
            product = rw.GetMol()
    return reactant, product


def remove_disconnected_parts_using_matrix(mol):
    num_atoms = mol.GetNumAtoms()
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

    csr_mat = csr_matrix(adjacency_matrix)
    _n_components, labels = connected_components(csgraph=csr_mat, directed=False)

    component_to_keep = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() > 0:
            component_to_keep.add(labels[atom.GetIdx()])

    atoms_to_keep = {idx for idx, comp_idx in enumerate(labels) if comp_idx in component_to_keep}

    rw = Chem.RWMol(mol)
    rw.BeginBatchEdit()
    for idx in range(num_atoms):
        if idx not in atoms_to_keep:
            rw.RemoveAtom(idx)
    rw.CommitBatchEdit()

    return rw.GetMol(), atoms_to_keep


def is_mapping_consistent(reactant, product, **kwargs):
    reactant_mappings = {atom.GetAtomMapNum() for atom in reactant.GetAtoms() if atom.GetAtomMapNum() > 0}
    product_mappings = {atom.GetAtomMapNum() for atom in product.GetAtoms() if atom.GetAtomMapNum() > 0}
    if kwargs.get("verbose", False):
        print(reactant_mappings)
        print(product_mappings)
        print(reactant_mappings == product_mappings)
    return reactant_mappings == product_mappings


def clean_and_map_reaction(r_smiles, p_smiles):
    rm = Chem.MolFromSmiles(r_smiles)
    pm = Chem.MolFromSmiles(p_smiles)

    rm_clean, atoms_to_keep_rm = remove_disconnected_parts_using_matrix(rm)
    pm_clean, atoms_to_keep_pm = remove_disconnected_parts_using_matrix(pm)

    atoms_to_keep = atoms_to_keep_rm.union(atoms_to_keep_pm)

    rm_final, pm_final = check_and_add_dative_bond(rm_clean, pm_clean, atoms_to_keep)

    return rm_final, pm_final, Chem.MolToSmiles(rm_final) + ">>" + Chem.MolToSmiles(pm_final)


def _representation_invariant_fingerprint(atom):
    """Atom-property fingerprint that does NOT distinguish ``[CH3]`` from
    ``C`` (i.e., explicit vs implicit H representations of the same chemistry).

    The original :func:`get_atom_properties` includes
    ``implicitValence``/``explicitValence`` which split the H count between
    bracket and implicit forms; this drops them and uses the
    representation-invariant ``totalValence`` and ``totalDegree`` instead.
    Used by :func:`map_missing_atoms` so SMILES like ``CO`` and ``[CH3:1]O``
    fingerprint identically on the methyl carbon.
    """
    propd = {
        "stereo": atom.GetChiralTag(),
        "charge": atom.GetFormalCharge(),
        "numHs": atom.GetTotalNumHs(),
        "hybridization": atom.GetHybridization(),
        "isAromatic": atom.GetIsAromatic(),
        "atomicNum": atom.GetAtomicNum(),
        "isotope": atom.GetIsotope(),
        "totalDegree": atom.GetTotalDegree(),
        "totalValence": atom.GetTotalValence(),
        "numRadicalElectrons": atom.GetNumRadicalElectrons(),
    }
    return hashlib.md5(str(propd).encode(), usedforsecurity=False).hexdigest()


def map_missing_atoms(reactant, product):
    """Propagate atom-map numbers across reactant/product by atom-property
    fingerprint matching.

    For each unmapped atom on one side, look for a *mapped* atom on the
    other side whose atom-property fingerprint (atomic number, charge,
    Hs, hybridization, valence, isotope, ...) is identical. If found,
    copy that map number across. Mutates both molecules in place.

    Compared to the original ``rxnmapdiff.cpp::map_missing_atoms``, this
    version tracks which map numbers are already in use on each side
    and refuses to assign a duplicate. Without that guard, a molecule
    with multiple chemically-equivalent atoms (e.g., two CH3 groups)
    would have all of them collapse to the same map number — silently
    producing inconsistent reactions.

    Pairs first match wins (greedy). For production-grade curation use
    a richer fingerprint (Morgan radius-2) or bipartite matching.
    """
    used_in_reactant = {a.GetAtomMapNum() for a in reactant.GetAtoms()
                        if a.GetAtomMapNum() > 0}
    used_in_product = {a.GetAtomMapNum() for a in product.GetAtoms()
                       if a.GetAtomMapNum() > 0}

    # Pass 1: unmapped reactant atoms borrow from mapped product atoms.
    for r_atom in reactant.GetAtoms():
        if r_atom.GetAtomMapNum() != 0:
            continue
        r_props = _representation_invariant_fingerprint(r_atom)
        for p_atom in product.GetAtoms():
            mn = p_atom.GetAtomMapNum()
            if mn == 0 or mn in used_in_reactant:
                continue
            if _representation_invariant_fingerprint(p_atom) == r_props:
                r_atom.SetAtomMapNum(mn)
                used_in_reactant.add(mn)
                break

    # Pass 2: symmetric — unmapped product atoms borrow from mapped reactant atoms.
    for p_atom in product.GetAtoms():
        if p_atom.GetAtomMapNum() != 0:
            continue
        p_props = _representation_invariant_fingerprint(p_atom)
        for r_atom in reactant.GetAtoms():
            mn = r_atom.GetAtomMapNum()
            if mn == 0 or mn in used_in_product:
                continue
            if _representation_invariant_fingerprint(r_atom) == p_props:
                p_atom.SetAtomMapNum(mn)
                used_in_product.add(mn)
                break


def _unmapped_fragment_smiles(mol):
    """SMILES of each connected component whose atoms are *all* unmapped."""
    out = []
    for fm in Chem.GetMolFrags(mol, asMols=True):
        if all(a.GetAtomMapNum() == 0 for a in fm.GetAtoms()):
            out.append(Chem.MolToSmiles(fm))
    return out


def _find_and_cut_non_core_fragments(mol, core_mapnums):
    """Cut bonds bridging core to non-core atoms; return SMILES of the
    resulting non-core fragments.

    Ported from CRS ``find_and_cut_fragments`` (rdkit-crs-backup
    ``Code/GraphMol/CondensedGraphRxn/RxnCleaning.cpp``). For each atom
    whose map number is in ``core_mapnums``, examine its bonds. If a bond
    connects atoms with different map numbers AND at least one endpoint
    is *not* in core, mark for cutting. After all cuts, every fragment
    is either pure-core or pure-non-core; we return the non-core ones,
    which represent leaving / incoming groups (or spectators that the
    caller filters).

    This is more rigorous than the simple "mapped vs unmapped" cut: a
    leaving group whose atoms happen to bear map numbers (but those map
    numbers don't appear on the other side) is correctly classified as
    non-core, even though every atom is "mapped" in the lone-mol sense.
    """
    rw = Chem.RWMol(mol)
    rw.BeginBatchEdit()
    for bond in list(rw.GetBonds()):
        m1 = rw.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
        m2 = rw.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
        if m1 == m2:
            continue  # both core or both non-core (or both 0) — leave alone
        in_core_1 = m1 in core_mapnums
        in_core_2 = m2 in core_mapnums
        if in_core_1 != in_core_2:
            rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    rw.CommitBatchEdit()

    out = []
    for fm in Chem.GetMolFrags(rw, asMols=True, sanitizeFrags=False):
        frag_mapnums = {a.GetAtomMapNum() for a in fm.GetAtoms()}
        # Pure-core fragment: skip (it's the reaction center)
        if frag_mapnums and frag_mapnums.issubset(core_mapnums):
            continue
        # Otherwise this is a non-core fragment (leaving / incoming / spectator)
        try:
            out.append(Chem.MolToSmiles(fm))
        except Exception:
            out.append(Chem.MolToSmiles(fm, canonical=False))
    return out


def identify_leaving_groups(r_smiles, p_smiles):
    """Return ``(leaving_groups, incoming_groups)`` as two lists of SMILES.

    Algorithm (after CRS ``balance_reaction_using_atom_mapping``):

    1. Run :func:`map_missing_atoms` on fresh copies of reactant/product
       to propagate atom-property-matching map numbers — maximizes the
       set of atoms with usable mappings before set-theory analysis.
    2. ``core_mapnums = reactant_mapnums ∩ product_mapnums`` — atoms
       present (and mapped) on both sides; the reaction center.
    3. On each side, cut bonds bridging core to non-core atoms
       (:func:`_find_and_cut_non_core_fragments`).
    4. Non-core fragments on the reactant side are *leaving groups*;
       on the product side they are *incoming groups*.
    5. Spectators (fragments with identical SMILES on both sides)
       are filtered — they pass through the reaction unchanged.

    This catches three kinds of leaving groups:
    - Standalone unmapped byproducts (``CC`` in test_one)
    - Embedded unmapped substructures (acetate in ester hydrolysis)
    - "Labeled" leaving groups whose atoms bear map numbers that simply
      don't appear in the other side (rare but real in some datasets)
    """
    rm = Chem.RWMol(Chem.MolFromSmiles(r_smiles))
    pm = Chem.RWMol(Chem.MolFromSmiles(p_smiles))
    map_missing_atoms(rm, pm)

    r_mapnums = {a.GetAtomMapNum() for a in rm.GetAtoms() if a.GetAtomMapNum() > 0}
    p_mapnums = {a.GetAtomMapNum() for a in pm.GetAtoms() if a.GetAtomMapNum() > 0}
    core_mapnums = r_mapnums & p_mapnums

    leaving = sorted(set(_find_and_cut_non_core_fragments(rm, core_mapnums)))
    incoming = sorted(set(_find_and_cut_non_core_fragments(pm, core_mapnums)))

    # Spectator filter: identical SMILES on both sides → unchanged passthrough.
    spectators = set(leaving) & set(incoming)
    leaving = [s for s in leaving if s not in spectators]
    incoming = [s for s in incoming if s not in spectators]
    return leaving, incoming


def _true_disconnected_smiles(r_smiles, p_smiles):
    """Build the disconnected-form ("true") cleaned SMILES — fragments with
    no mapped atoms removed, but no dative bonds added. This preserves the
    original valences and is the canonical input for reconstituting the
    actual reactant/product molecules from the cleaner's output.
    """
    rm = Chem.MolFromSmiles(r_smiles)
    pm = Chem.MolFromSmiles(p_smiles)
    rm_clean, _ = remove_disconnected_parts_using_matrix(rm)
    pm_clean, _ = remove_disconnected_parts_using_matrix(pm)
    return Chem.MolToSmiles(rm_clean) + ">>" + Chem.MolToSmiles(pm_clean)


def ecrs(rxn):
    """Clean an atom-mapped reaction SMILES and return both fictive (dative-
    merged, single-graph) and true (disconnected, real-valence) forms.

    Returns a :class:`CleanedReaction` named tuple with fields
    ``fictive``, ``true``, ``mode``. See class docstring for semantics.
    """
    r_smiles = rxn.split('>')[0]
    p_smiles = rxn.split('>')[-1]
    true_smi = _true_disconnected_smiles(r_smiles, p_smiles)

    rm_final, pm_final, original_result = clean_and_map_reaction(r_smiles, p_smiles)

    # Workaround: when a dative bond is added on the reactant side, the original
    # forward run sometimes leaves mappings inconsistent. Re-running with reactant
    # and product swapped recovers a consistent mapping.
    leaving, incoming = identify_leaving_groups(r_smiles, p_smiles)

    if is_mapping_consistent(rm_final, pm_final):
        return CleanedReaction(fictive=original_result, true=true_smi, mode="ORIGINAL",
                               leaving_groups=leaving, incoming_groups=incoming)

    pm_final, rm_final, _inverted_result = clean_and_map_reaction(p_smiles, r_smiles)

    if is_mapping_consistent(rm_final, pm_final):
        workaround_result = Chem.MolToSmiles(rm_final) + ">>" + Chem.MolToSmiles(pm_final)
        return CleanedReaction(fictive=workaround_result, true=true_smi, mode="WORKAROUND",
                               leaving_groups=leaving, incoming_groups=incoming)

    return CleanedReaction(fictive=original_result, true=true_smi, mode="ORIGINAL",
                           leaving_groups=leaving, incoming_groups=incoming)
