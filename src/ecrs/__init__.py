"""ecrs — reaction SMILES cleanup with dative-bond detection.

Public API:
    ecrs(rxn) -> CleanedReaction(fictive, true, mode)
        - fictive: dative-merged single-graph SMILES for GNN input
        - true:    disconnected real-valence SMILES for product reconstruction
        - mode:    "ORIGINAL" | "WORKAROUND"
    clean_and_map_reaction(r_smiles, p_smiles) -> (rm, pm, smiles)

Lower-level helpers (useful for tests and bespoke pipelines):
    is_mapping_consistent, get_atom_properties, identify_changed_bonds,
    add_atom_mapping_to_neighbors, check_and_add_dative_bond,
    remove_disconnected_parts_using_matrix,
    is_potential_donor, is_potential_acceptor, dative_donor_acceptor
"""
from ecrs._dative_neighbor import (
    CleanedReaction,
    add_atom_mapping_to_neighbors,
    check_and_add_dative_bond,
    clean_and_map_reaction,
    dative_donor_acceptor,
    get_atom_properties,
    identify_changed_bonds,
    identify_leaving_groups,
    is_mapping_consistent,
    is_potential_acceptor,
    is_potential_donor,
    map_missing_atoms,
    remove_disconnected_parts_using_matrix,
    ecrs,
)
from ecrs._signature import (
    reaction_signature,
    reaction_signatures,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CleanedReaction",
    "ecrs",
    "clean_and_map_reaction",
    "is_mapping_consistent",
    "get_atom_properties",
    "identify_changed_bonds",
    "identify_leaving_groups",
    "map_missing_atoms",
    "add_atom_mapping_to_neighbors",
    "check_and_add_dative_bond",
    "remove_disconnected_parts_using_matrix",
    "is_potential_donor",
    "is_potential_acceptor",
    "dative_donor_acceptor",
    "reaction_signature",
    "reaction_signatures",
]
