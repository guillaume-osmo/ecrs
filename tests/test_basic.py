"""Hand-written unit tests covering the core ecrs behavior."""
import unittest

from rdkit import Chem

from ecrs import (
    CleanedReaction,
    dative_donor_acceptor,
    get_atom_properties,
    identify_leaving_groups,
    is_potential_acceptor,
    is_potential_donor,
    map_missing_atoms,
    remove_disconnected_parts_using_matrix,
    ecrs,
)


class TestRxnClean(unittest.TestCase):
    """Tests for ecrs."""

    def setUp(self):
        self.method = ecrs

    def test_returns_named_tuple_with_fictive_true_mode(self):
        rxn = "OCCCC[CH3:1].[NH2:2]C>>OCCCC[CH2:1][NH:2]C.CC"
        result = self.method(rxn)
        self.assertIsInstance(result, CleanedReaction)
        self.assertTrue(hasattr(result, "fictive"))
        self.assertTrue(hasattr(result, "true"))
        self.assertTrue(hasattr(result, "mode"))

    def test_one(self):
        # Dative direction is chemistry-correct: N (donor, lone pair) is the
        # begin atom, C (acceptor) is the end atom. RDKit's canonical writer
        # renders that as `[C]<-[N]`. The output retains original H counts —
        # the SMILES is intentionally fictive (graph trick) and over-valent
        # under stock RDKit's dative valence accounting; rdkit-fir's patched
        # Bond.cpp can be told (per-call) to treat datives as 0-contribution.
        rxn = "OCCCC[CH3:1].[NH2:2]C>>OCCCC[CH2:1][NH:2]C.CC"
        expected_fictive = "OCCC[CH2:3][CH3:1]<-[NH2:2][CH3:4]>>OCCC[CH2:3][CH2:1][NH:2][CH3:4]"
        result = self.method(rxn)
        self.assertEqual(result.fictive, expected_fictive)
        self.assertEqual(result.mode, "ORIGINAL")
        # The "true" form is real-valence (no datives) — should be parseable
        # by stock RDKit and reflect the original disconnected fragments.
        self.assertNotIn("<-", result.true)
        self.assertNotIn("->", result.true)
        # Mapped atoms preserved
        self.assertIn(":1", result.true)
        self.assertIn(":2", result.true)

    def test_two(self):
        rxn = "CC[CH2:1][NH:2]C.CC>>CC[CH3:1].[NH2:2]C"
        expected_fictive = "C[CH2:3][CH2:1][NH:2][CH3:4]>>C[CH2:3][CH3:1]<-[NH2:2][CH3:4]"
        result = self.method(rxn)
        self.assertEqual(result.fictive, expected_fictive)
        self.assertEqual(result.mode, "ORIGINAL")

    def test_true_smiles_is_parseable_by_stock_rdkit(self):
        # The "true" output is the disconnected real-valence form and must
        # round-trip cleanly even on stock RDKit (no patches required).
        rxn = "OCCCC[CH3:1].[NH2:2]C>>OCCCC[CH2:1][NH:2]C.CC"
        result = self.method(rxn)
        r_smi, p_smi = result.true.split(">>")
        self.assertIsNotNone(Chem.MolFromSmiles(r_smi))
        self.assertIsNotNone(Chem.MolFromSmiles(p_smi))

    def test_no_dative_needed(self):
        # Methanol-like reaction: every heavy atom mapped, no disconnect.
        # Cleaner should not add any dative.
        rxn = "[OH:1][CH3:2]>>[O:1]=[CH2:2]"
        result = self.method(rxn)
        self.assertIn(">>", result.fictive)
        self.assertNotIn("<-", result.fictive)
        self.assertNotIn("->", result.fictive)
        self.assertEqual(result.mode, "ORIGINAL")

    def test_unmapped_byproducts_are_dropped(self):
        rxn = "OCCCC[CH3:1].[NH2:2]C.CC>>OCCCC[CH2:1][NH:2]C.OCC"
        result = self.method(rxn)
        self.assertNotIn("CC.", result.fictive.split(">>")[0])
        self.assertNotIn(".OCC", result.fictive.split(">>")[1])

    def test_workaround_path(self):
        rxn = "OCCCC[CH2:1][NH:2]C.CC>>OCCCC[CH3:1].[NH2:2]C"
        result = self.method(rxn)
        self.assertIn(">>", result.fictive)
        self.assertIn(result.mode, {"ORIGINAL", "WORKAROUND"})

    def test_chemistry_rules_donor_acceptor(self):
        # N-C: classic donor-acceptor pair — N donor, C acceptor.
        m = Chem.MolFromSmiles("NC")
        n_atom = next(a for a in m.GetAtoms() if a.GetSymbol() == "N")
        c_atom = next(a for a in m.GetAtoms() if a.GetSymbol() == "C")
        self.assertTrue(is_potential_donor(n_atom))
        self.assertTrue(is_potential_acceptor(c_atom))
        self.assertEqual(dative_donor_acceptor(n_atom, c_atom), (n_atom, c_atom))
        self.assertEqual(dative_donor_acceptor(c_atom, n_atom), (n_atom, c_atom))

    def test_chemistry_rules_reject_two_donors(self):
        # N-N: both are donors, neither is electrophilic. No dative.
        m = Chem.MolFromSmiles("NN")
        a, b = m.GetAtomWithIdx(0), m.GetAtomWithIdx(1)
        self.assertIsNone(dative_donor_acceptor(a, b))

    def test_chemistry_rules_reject_two_carbons(self):
        # C-C: neither has a lone pair. Not a valid dative pair.
        m = Chem.MolFromSmiles("CC")
        a, b = m.GetAtomWithIdx(0), m.GetAtomWithIdx(1)
        self.assertIsNone(dative_donor_acceptor(a, b))

    def test_chemistry_rules_quaternary_amine_is_not_donor(self):
        # R4N+: positive formal charge, no lone pair available → not donor.
        m = Chem.MolFromSmiles("[N+](C)(C)(C)C")
        n_atom = next(a for a in m.GetAtoms() if a.GetSymbol() == "N")
        self.assertFalse(is_potential_donor(n_atom))


class TestLeavingGroups(unittest.TestCase):
    """Tests for map_missing_atoms + identify_leaving_groups (data curation)."""

    def test_map_missing_atoms_propagates_simple(self):
        # Reactant has unmapped C; product has same C with map :1.
        # map_missing_atoms should copy :1 onto the reactant C.
        rm = Chem.RWMol(Chem.MolFromSmiles("CO"))
        pm = Chem.RWMol(Chem.MolFromSmiles("[CH3:1]O"))
        map_missing_atoms(rm, pm)
        c_atoms = [a for a in rm.GetAtoms() if a.GetSymbol() == "C"]
        self.assertEqual(c_atoms[0].GetAtomMapNum(), 1)

    def test_map_missing_atoms_no_duplicate_assignment(self):
        # Reactant has TWO atoms with identical fingerprints; product has
        # ONE mapped atom with the same fingerprint. The crs original
        # would assign :1 to BOTH reactant atoms (silently producing
        # duplicate map numbers, which break atom-mapping invariants).
        # Our version's used-set guard prevents this.
        rm = Chem.RWMol(Chem.MolFromSmiles("CC"))         # two Cs each with 1 heavy nbr
        pm = Chem.RWMol(Chem.MolFromSmiles("[CH3:1]C"))   # mapped C also with 1 heavy nbr
        map_missing_atoms(rm, pm)
        nums = sorted(a.GetAtomMapNum() for a in rm.GetAtoms())
        self.assertEqual(nums, [0, 1])  # exactly one match, not two

    def test_identify_leaving_groups_standalone_byproduct(self):
        # Amine alkylation produces CC (ethane) as a standalone byproduct.
        leaving, incoming = identify_leaving_groups(
            "OCCCC[CH3:1].[NH2:2]C", "OCCCC[CH2:1][NH:2]C.CC"
        )
        self.assertIn("CC", incoming)
        # Reactant has no fully-disconnected unmapped fragment.
        self.assertEqual(leaving, [])

    def test_identify_leaving_groups_embedded(self):
        # Methyl acetate hydrolysis: acetate is embedded *within* the
        # mapped reactant fragment. Bond-cutting at the mapped/unmapped
        # boundary should expose it.
        leaving, incoming = identify_leaving_groups(
            "CC(=O)O[CH3:1]", "[CH3:1]O"
        )
        self.assertEqual(leaving, ["CC(=O)O"])
        self.assertEqual(incoming, ["O"])

    def test_identify_leaving_groups_spectators_filtered(self):
        # [Na+] and [Cl-] appear unmapped on both sides — they're
        # spectator counter-ions and must NOT be reported.
        leaving, incoming = identify_leaving_groups(
            "[CH3:1].[Na+].[Cl-]", "[CH3:1]O.[Na+].[Cl-]"
        )
        self.assertNotIn("[Na+]", leaving)
        self.assertNotIn("[Cl-]", leaving)
        self.assertNotIn("[Na+]", incoming)
        self.assertNotIn("[Cl-]", incoming)

    def test_identify_leaving_groups_full_coverage(self):
        # Fully-mapped reaction: no leaving / incoming groups.
        leaving, incoming = identify_leaving_groups("[CH3:1][OH:2]", "[O:1]=[CH2:2]")
        self.assertEqual(leaving, [])
        self.assertEqual(incoming, [])

    def test_ecrs_returns_leaving_groups_field(self):
        # The cleaner should populate the new fields on its return value.
        result = ecrs("OCCCC[CH3:1].[NH2:2]C.[Na+].[Cl-]>>OCCCC[CH2:1][NH:2]C.CC.[Na+].[Cl-]")
        self.assertIn("CC", result.incoming_groups)
        for spectator in ("[Na+]", "[Cl-]"):
            self.assertNotIn(spectator, result.leaving_groups)
            self.assertNotIn(spectator, result.incoming_groups)

    def test_atom_property_hash_is_stable(self):
        # Same atom in two equivalent molecules should hash identically.
        m1 = Chem.MolFromSmiles("CCO")
        m2 = Chem.MolFromSmiles("CCO")
        for i in range(m1.GetNumAtoms()):
            self.assertEqual(
                get_atom_properties(m1.GetAtomWithIdx(i)),
                get_atom_properties(m2.GetAtomWithIdx(i)),
            )

    def test_remove_disconnected_keeps_mapped_fragment(self):
        # Mapped fragment (CCO with map 1 on C) must survive; unmapped CCC must go.
        mol = Chem.MolFromSmiles("[CH3:1]CO.CCC")
        cleaned, kept = remove_disconnected_parts_using_matrix(mol)
        self.assertEqual(cleaned.GetNumAtoms(), 3)
        self.assertEqual(len(kept), 3)


if __name__ == "__main__":
    unittest.main()
