#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/ChemTransforms/MolFragmenter.h>
#include <iostream>
#include <set>
#include <vector>
#include <memory>

using namespace RDKit;

std::vector<std::unique_ptr<ROMol>> find_and_cut_fragments(const ROMol& mol, const std::set<int>& core_mapnums) {
    std::vector<unsigned int> bonds_to_break;

    for (const auto& atom : mol.atoms()) {
        if (core_mapnums.find(atom->getAtomMapNum()) != core_mapnums.end()) {
            for (const auto& bond : mol.atomBonds(atom)) {
                int begin_atom_mapnum = bond->getBeginAtom()->getAtomMapNum();
                int end_atom_mapnum = bond->getEndAtom()->getAtomMapNum();

                if ((begin_atom_mapnum != end_atom_mapnum) &&
                    (core_mapnums.find(begin_atom_mapnum) == core_mapnums.end() || core_mapnums.find(end_atom_mapnum) == core_mapnums.end())) {
                    bonds_to_break.push_back(bond->getIdx());
                }
            }
        }
    }

    // If there are no bonds to break, return the molecule as is
    if (bonds_to_break.empty()) {
        std::vector<std::unique_ptr<ROMol>> result;
        result.push_back(std::make_unique<ROMol>(mol));
        return result;
    }

    // Break bonds to generate fragments
    std::unique_ptr<ROMol> frag_mol(MolFragmenter::fragmentOnBonds(mol, bonds_to_break, false));
    std::vector<ROMOL_SPTR> frags = MolOps::getMolFrags(*frag_mol);

    std::vector<std::unique_ptr<ROMol>> result;
    for (const auto& frag : frags) {
        result.push_back(std::make_unique<ROMol>(*frag));
    }

    return result;
}

std::string balance_reaction_using_atom_mapping(const ROMol& reactant, const ROMol& product) {
    // Get atom map numbers from both reactant and product
    std::set<int> reactant_mapnums;
    std::set<int> product_mapnums;

    for (const auto& atom : reactant.atoms()) {
        if (atom->getAtomMapNum() > 0) {
            reactant_mapnums.insert(atom->getAtomMapNum());
        }
    }

    for (const auto& atom : product.atoms()) {
        if (atom->getAtomMapNum() > 0) {
            product_mapnums.insert(atom->getAtomMapNum());
        }
    }

    // Core mapnums are those common in both reactant and product
    std::set<int> core_mapnums;
    std::set_intersection(reactant_mapnums.begin(), reactant_mapnums.end(), product_mapnums.begin(), product_mapnums.end(),
                          std::inserter(core_mapnums, core_mapnums.begin()));

    // Find and cut fragments that are not part of the core
    std::vector<std::unique_ptr<ROMol>> reactant_frags = find_and_cut_fragments(reactant, core_mapnums);
    std::vector<std::unique_ptr<ROMol>> product_frags = find_and_cut_fragments(product, core_mapnums);

    // Identify the fragments to be exchanged
    std::vector<std::unique_ptr<ROMol>> reactant_only_fragments;
    std::vector<std::unique_ptr<ROMol>> product_only_fragments;

    for (auto& frag : reactant_frags) {
        std::set<int> frag_mapnums;
        for (const auto& atom : frag->atoms()) {
            if (atom->getAtomMapNum() > 0) {
                frag_mapnums.insert(atom->getAtomMapNum());
            }
        }
        if (!std::includes(core_mapnums.begin(), core_mapnums.end(), frag_mapnums.begin(), frag_mapnums.end()) &&
            std::none_of(frag_mapnums.begin(), frag_mapnums.end(), [&](int map_num) { return product_mapnums.count(map_num) > 0; })) {
            reactant_only_fragments.push_back(std::move(frag));
        }
    }

    for (auto& frag : product_frags) {
        std::set<int> frag_mapnums;
        for (const auto& atom : frag->atoms()) {
            if (atom->getAtomMapNum() > 0) {
                frag_mapnums.insert(atom->getAtomMapNum());
            }
        }
        if (!std::includes(core_mapnums.begin(), core_mapnums.end(), frag_mapnums.begin(), frag_mapnums.end()) &&
            std::none_of(frag_mapnums.begin(), frag_mapnums.end(), [&](int map_num) { return reactant_mapnums.count(map_num) > 0; })) {
            product_only_fragments.push_back(std::move(frag));
        }
    }

    // Create the final product by adding reactant-only fragments to the product
    RWMol final_product(product);
    for (auto& frag : reactant_only_fragments) {
        final_product.insertMol(*frag);
    }

    // Create the final reactant by adding product-only fragments to the reactant
    RWMol final_reactant(reactant);
    for (auto& frag : product_only_fragments) {
        final_reactant.insertMol(*frag);
    }

    // Convert to SMILES for the balanced reaction
    std::string balanced_reaction = MolToSmiles(final_reactant) + ">>" + MolToSmiles(final_product);
    return balanced_reaction;
}

int main() {
    // Example reaction for balancing with specific fragments
    std::string r_smiles = "[CH3:1][CH2:2][CH:3]([CH2:4][Cl:6])[Cl:5]";
    std::string p_smiles = "[CH3:1][CH2:2][CH:3]=[CH2:4]";

    std::unique_ptr<ROMol> reactant(SmilesToMol(r_smiles));
    std::unique_ptr<ROMol> product(SmilesToMol(p_smiles));

    std::string balanced_product = balance_reaction_using_atom_mapping(*reactant, *product);

    std::string expected = "[CH3:1][CH2:2][CH:3]([CH2:4][Cl:6])[Cl:5]>>[CH3:1][CH2:2][CH:3]=[CH2:4].[ClH:5].[ClH:6]";

    std::cout << "Balanced Reaction 1: " << balanced_product << std::endl;
    std::cout << "Expected 1: " << expected << std::endl;
    assert(balanced_product == expected);

    std::string r_smiles2 = "[CH3:1][CH2:2][CH:3]([CH2:4][OH:6])[OH:5]";
    std::string p_smiles2 = "[CH3:1][CH2:2][CH:3]=[CH2:4]";

    std::unique_ptr<ROMol> reactant2(SmilesToMol(r_smiles2));
    std::unique_ptr<ROMol> product2(SmilesToMol(p_smiles2));

    std::string balanced_product2 = balance_reaction_using_atom_mapping(*reactant2, *product2);

    std::string expected2 = "[CH3:1][CH2:2][CH:3]([CH2:4][OH:6])[OH:5]>>[CH3:1][CH2:2][CH:3]=[CH2:4].[OH2:5].[OH2:6]";

    std::cout << "Balanced Reaction 2: " << balanced_product2 << std::endl;
    std::cout << "Expected 2: " << expected2 << std::endl;
    assert(balanced_product2 == expected2);

    std::string r_smiles3 = "[CH3:1][CH2:2][CH:3]=[CH2:4]";
    std::string p_smiles3 = "[CH3:1][CH2:2][CH:3]([CH2:4][OH:6])[OH:5]";

    std::unique_ptr<ROMol> reactant3(SmilesToMol(r_smiles3));
    std::unique_ptr<ROMol> product3(SmilesToMol(p_smiles3));

    std::string balanced_product3 = balance_reaction_using_atom_mapping(*reactant3, *product3);

    std::string expected3 = "[CH3:1][CH2:2][CH:3]=[CH2:4].[OH2:5].[OH2:6]>>[CH3:1][CH2:2][CH:3]([CH2:4][OH:6])[OH:5]";

    std::cout << "Balanced Reaction 3: " << balanced_product3 << std::endl;
    std::cout << "Expected 3: " << expected3 << std::endl;
    assert(balanced_product3 == expected3);


    std::string r_smiles4 = "[CH3:1][CH2:2][CH:3]=[CH2:4]";
    std::string p_smiles4 = "[CH3:1][CH2:2][CH:3]([CH2:4][Cl:6])[Cl:5]";

    std::unique_ptr<ROMol> reactant4(SmilesToMol(r_smiles4));
    std::unique_ptr<ROMol> product4(SmilesToMol(p_smiles4));

    std::string balanced_product4 = balance_reaction_using_atom_mapping(*reactant4, *product4);

    std::string expected4 = "[CH3:1][CH2:2][CH:3]=[CH2:4].[ClH:5].[ClH:6]>>[CH3:1][CH2:2][CH:3]([CH2:4][Cl:6])[Cl:5]";

    std::cout << "Balanced Reaction 4: " << balanced_product4 << std::endl;
    std::cout << "Expected 4: " << expected4 << std::endl;
    assert(balanced_product4 == expected4);

    std::string r_smiles5 = "CC[CH2:1][CH2:2][CH:3]=[CH2:4]";
    std::string p_smiles5 = "CC[CH2:1][CH2:2][CH:3]([CH2:4][Cl:6])[Cl:5]";

    std::unique_ptr<ROMol> reactant5(SmilesToMol(r_smiles5));
    std::unique_ptr<ROMol> product5(SmilesToMol(p_smiles5));

    std::string balanced_product5 = balance_reaction_using_atom_mapping(*reactant5, *product5);

    std::string expected5 = "CC[CH2:1][CH2:2][CH:3]=[CH2:4].[ClH:5].[ClH:6]>>CC[CH2:1][CH2:2][CH:3]([CH2:4][Cl:6])[Cl:5]";

    std::cout << "Balanced Reaction 5: " << balanced_product5 << std::endl;
    std::cout << "Expected 5: " << expected5 << std::endl;
    assert(balanced_product5 == expected5);






    return 0;
}