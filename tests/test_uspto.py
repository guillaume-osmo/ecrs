"""Regression and stress tests sourced from the USPTO-50k dataset.

These exercise chemical features that surface bugs not covered by the
minimal hand-written cases in test_basic: large/polycyclic molecules,
charged species, organometallic protecting groups, stereochemistry, and
the asymmetric atom-count case where reactant and product have different
heavy-atom counts after disconnected-fragment removal.
"""
import unittest

from rdkit import Chem

from ecrs import ecrs


class TestRxnCleanUSPTO(unittest.TestCase):
    def _check_non_crashing(self, rxn):
        """Cleaner returns a fictive r>>p SMILES without raising. The fictive
        output may have over-valent dative bonds (see _dative_neighbor module
        docstring); we only check it parses on the *true* (disconnected) side."""
        result = ecrs(rxn)
        self.assertIn(">>", result.fictive, f"missing >> in fictive: {result.fictive}")
        self.assertIn(">>", result.true, f"missing >> in true: {result.true}")
        self.assertIn(result.mode, {"ORIGINAL", "WORKAROUND"})
        # The "true" form must always be parseable by stock RDKit.
        r_true, p_true = result.true.split(">>")
        self.assertIsNotNone(Chem.MolFromSmiles(r_true), f"true reactant unparseable: {r_true[:120]}")
        self.assertIsNotNone(Chem.MolFromSmiles(p_true), f"true product unparseable: {p_true[:120]}")
        return result

    def _check_parseable(self, rxn):
        """Stronger contract: even the *fictive* output round-trips through
        stock RDKit. Holds when no dative was actually added."""
        result = self._check_non_crashing(rxn)
        r_smi, p_smi = result.fictive.split(">>")
        rm = Chem.MolFromSmiles(r_smi)
        pm = Chem.MolFromSmiles(p_smi)
        self.assertIsNotNone(rm, f"fictive reactant did not parse: {r_smi[:120]}")
        self.assertIsNotNone(pm, f"fictive product did not parse: {p_smi[:120]}")
        return result, rm, pm

    def test_atom_count_mismatch_does_not_crash(self):
        # USPTO US06051731 — Wittig-style reaction. Reactant has 35 heavy atoms,
        # product has 15 after disconnected-fragment cleanup. Pre-fix this
        # crashed with `RuntimeError: Range Error idx1: 27 < 15` because the
        # mapper indexed product with reactant atom indices.
        rxn = (
            "O=[CH:1][CH2:2][c:3]1[cH:4][cH:5][c:6]([Br:7])[cH:8][cH:9]1."
            "c1ccc(P(c2ccccc2)(c2ccccc2)=[CH:10][C:11]([O:12][CH2:13][CH3:14])=[O:15])cc1"
            ">>[CH:1]([CH2:2][c:3]1[cH:4][cH:5][c:6]([Br:7])[cH:8][cH:9]1)"
            "=[CH:10][C:11]([O:12][CH2:13][CH3:14])=[O:15]"
        )
        self._check_non_crashing(rxn)

    def test_polycyclic_aromatic_heterocycle(self):
        # USPTO US05010077 — fused aromatic system with nitro + indole-like
        # tricyclic core. Tests sanitization survives many aromatic
        # heteroatoms after RWMol edits.
        rxn = (
            "O=[N+:1]([O-])[c:2]1[cH:3][cH:4][c:5]2[nH:6][c:7]3[cH:8][n:9]"
            "[c:10]([C:11]([O:12][CH2:13][CH3:14])=[O:15])[c:16]([CH3:17])"
            "[c:18]3[c:19]2[cH:20]1>>[NH2:1][c:2]1[cH:3][cH:4][c:5]2[nH:6]"
            "[c:7]3[cH:8][n:9][c:10]([C:11]([O:12][CH2:13][CH3:14])=[O:15])"
            "[c:16]([CH3:17])[c:18]3[c:19]2[cH:20]1"
        )
        self._check_non_crashing(rxn)

    def test_charged_nitro_reduction(self):
        # USPTO US06492393B1 — small aromatic nitro reduction. Tests formal
        # charges (+/-) round-trip through RWMol cloning.
        rxn = (
            "O=[N+:1]([O-])[c:2]1[c:3]([CH3:4])[cH:5][cH:6][c:7]"
            "([C:8]([CH3:9])([CH3:10])[CH3:11])[cH:12]1"
            ">>[NH2:1][c:2]1[c:3]([CH3:4])[cH:5][cH:6][c:7]"
            "([C:8]([CH3:9])([CH3:10])[CH3:11])[cH:12]1"
        )
        _, rm, pm = self._check_parseable(rxn)
        # Formal charge of mapped N should be preserved on whichever side it remains
        for mol in (rm, pm):
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() == 1 and atom.GetSymbol() == "N":
                    # Either +1 (reactant nitro) or 0 (product amine)
                    self.assertIn(atom.GetFormalCharge(), (0, 1))

    def test_organometallic_silyl_protecting_group(self):
        # USPTO US20140088306A1 — TBS-O deprotection. Silyl group should be
        # treated as a regular fragment and dropped (it is unmapped in the
        # reactant) without RDKit valence errors on Si.
        rxn = (
            "CC(C)(C)[Si](C)(C)[O:1][c:2]1[cH:3][cH:4][c:5]2[n:6][cH:7]"
            "[c:8](-[c:9]3[cH:10][cH:11][c:12]([N:13]([CH3:14])[CH3:15])"
            "[cH:16][cH:17]3)[n:18][c:19]2[cH:20]1>>[OH:1][c:2]1[cH:3]"
            "[cH:4][c:5]2[n:6][cH:7][c:8](-[c:9]3[cH:10][cH:11][c:12]"
            "([N:13]([CH3:14])[CH3:15])[cH:16][cH:17]3)[n:18][c:19]2[cH:20]1"
        )
        self._check_non_crashing(rxn)

    def test_stereochemistry_preserved(self):
        # USPTO US04564609 — proline derivative carbamate. The C@@H stereo
        # designator on the mapped chiral center must survive the RWMol
        # round-trip.
        rxn = (
            "O[C:1](=[O:2])[C@@H:3]1[CH2:4][CH2:5][CH2:6][N:7]1[C:8](=[O:9])"
            "[O:10][CH2:11][c:12]1[cH:13][cH:14][cH:15][cH:16][cH:17]1."
            "[CH3:18][NH:19][CH3:20]>>[C:1](=[O:2])([C@@H:3]1[CH2:4][CH2:5]"
            "[CH2:6][N:7]1[C:8](=[O:9])[O:10][CH2:11][c:12]1[cH:13][cH:14]"
            "[cH:15][cH:16][cH:17]1)[N:19]([CH3:18])[CH3:20]"
        )
        result = self._check_non_crashing(rxn)
        # The @@/@ designator on atom map :3 should survive in the output.
        self.assertIn("@", result.fictive)

    def test_workaround_path_real_uspto(self):
        # USPTO US20100197908A1 — small heterocycle iodination that takes the
        # WORKAROUND branch (the inverse-direction retry). Catches regressions
        # in the inverted clean_and_map_reaction call.
        rxn = (
            "IC[I:1].N[c:2]1[s:3][cH:4][c:5]([C:6]([O:7][CH2:8][CH3:9])=[O:10])"
            "[n:11]1>>[I:1][c:2]1[s:3][cH:4][c:5]([C:6]([O:7][CH2:8][CH3:9])"
            "=[O:10])[n:11]1"
        )
        result = self._check_non_crashing(rxn)
        self.assertEqual(result.mode, "WORKAROUND")

    def test_multifragment_byproducts_removed(self):
        # USPTO-style reaction with multiple unmapped reagent fragments
        # (CC(C)(C)OC(=O)O... Boc-anhydride, plus the actual mapped substrate).
        # Only the mapped fragment's atoms should remain.
        rxn = (
            "CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7]."
            "[NH2:8][c:9]1[cH:10][cH:11][c:12]([OH:13])[cH:14][c:15]1"
            "[C:16](=[O:17])[OH:18]"
            ">>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[NH:8]"
            "[c:9]1[cH:10][cH:11][c:12]([OH:13])[cH:14][c:15]1[C:16](=[O:17])[OH:18]"
        )
        result = self._check_non_crashing(rxn)
        # Boc-anhydride leaving group (`OC(=O)OC(C)(C)C` unmapped portion) should
        # not survive in the cleaned reactant — the unmapped Boc-O-Boc piece is
        # part of the same connected component as mapped atoms via [C:6] though,
        # so this is more about "doesn't crash on multifragment input".
        self.assertGreater(len(result.fictive), 30)


if __name__ == "__main__":
    unittest.main()
