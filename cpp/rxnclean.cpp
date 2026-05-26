#include <GraphMol/RDKitBase.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include "md5.h"

using namespace RDKit;

// myrdkit (formerly rdkit-fir) ships only the deprecated valence APIs; rdkit
// 2025.03 introduced Atom::ValenceType + a unified getValence(ValenceType)
// that we use in the Python sibling. Route through these inline helpers and
// flip the bodies once myrdkit has the new API merged in.
inline int implicitValence(const Atom* a) { return a->getImplicitValence(); }
inline int explicitValence(const Atom* a) { return a->getExplicitValence(); }

// ---------------------------------------------------------------------------
// Chemistry rules for dative gating
// ---------------------------------------------------------------------------

// Group 15/16/17 atoms typically carry an available lone pair.
// (N, O, F, P, S, Cl, As, Se, Br, Sb, Te, I, Bi)
static const std::unordered_set<int> LONE_PAIR_DONORS = {
    7, 8, 9, 15, 16, 17, 33, 34, 35, 51, 52, 53, 83};

// Group 13/14 + transition metals: typical electrophilic acceptors.
static bool isElectrophilicElement(int z) {
    if (z == 5 || z == 6 || z == 13 || z == 14 || z == 31 || z == 32) return true;
    if (z >= 21 && z <= 30) return true;   // Sc..Zn
    if (z >= 39 && z <= 48) return true;   // Y..Cd
    if (z >= 72 && z <= 80) return true;   // Hf..Hg
    return false;
}

static bool isPotentialDonor(const Atom* a) {
    if (!LONE_PAIR_DONORS.count(a->getAtomicNum())) return false;
    return a->getFormalCharge() <= 0;  // R4N+ etc. has no lone pair to donate
}

static bool isPotentialAcceptor(const Atom* a) {
    if (a->getFormalCharge() > 0) return true;  // any cation is an acceptor
    return isElectrophilicElement(a->getAtomicNum());
}

// If a, b form an unambiguous donor/acceptor pair, returns true and writes
// the pair into out_donor / out_acceptor. Otherwise returns false.
static bool dativeDonorAcceptor(Atom* a, Atom* b, Atom*& out_donor, Atom*& out_acceptor) {
    bool a_donor = isPotentialDonor(a);
    bool a_acc   = isPotentialAcceptor(a);
    bool b_donor = isPotentialDonor(b);
    bool b_acc   = isPotentialAcceptor(b);
    bool a_only_donor = a_donor && !a_acc;
    bool b_only_donor = b_donor && !b_acc;
    bool a_only_acc   = a_acc && !a_donor;
    bool b_only_acc   = b_acc && !b_donor;
    if (a_only_donor && b_only_acc) { out_donor = a; out_acceptor = b; return true; }
    if (b_only_donor && a_only_acc) { out_donor = b; out_acceptor = a; return true; }
    return false;
}

// Representation-invariant fingerprint — does not distinguish [CH3] from C.
// Drops implicit/explicit valence (which differ between bracketed and bare
// representations of the same chemistry) in favor of total* values that
// are invariant. Used by mapMissingAtoms below.
static std::string atomFingerprintInvariant(const Atom* atom) {
    std::ostringstream s;
    s << "z=" << atom->getAtomicNum()
      << ",chr=" << atom->getFormalCharge()
      << ",nH=" << atom->getTotalNumHs()
      << ",hyb=" << static_cast<int>(atom->getHybridization())
      << ",arom=" << atom->getIsAromatic()
      << ",iso=" << atom->getIsotope()
      << ",td=" << atom->getTotalDegree()
      << ",tv=" << atom->getTotalValence()
      << ",rad=" << atom->getNumRadicalElectrons()
      << ",chi=" << static_cast<int>(atom->getChiralTag());
    return md5(s.str());
}

// Propagate atom-map numbers across reactant <-> product by atom-property
// fingerprint matching. Tracks already-used map numbers on each side so a
// molecule with multiple chemically-equivalent atoms doesn't end up with
// duplicate map numbers (a bug in the original crs/rxnmapdiff.cpp).
//
// Mutates both molecules in place.
static void mapMissingAtoms(RWMol& reactant, RWMol& product) {
    std::set<int> usedInReactant, usedInProduct;
    for (auto a : reactant.atoms()) if (a->getAtomMapNum() > 0) usedInReactant.insert(a->getAtomMapNum());
    for (auto a : product.atoms())  if (a->getAtomMapNum() > 0) usedInProduct.insert(a->getAtomMapNum());

    for (auto r_atom : reactant.atoms()) {
        if (r_atom->getAtomMapNum() != 0) continue;
        const std::string r_fp = atomFingerprintInvariant(r_atom);
        for (auto p_atom : product.atoms()) {
            const int mn = p_atom->getAtomMapNum();
            if (mn == 0 || usedInReactant.count(mn)) continue;
            if (atomFingerprintInvariant(p_atom) == r_fp) {
                r_atom->setAtomMapNum(mn);
                usedInReactant.insert(mn);
                break;
            }
        }
    }
    for (auto p_atom : product.atoms()) {
        if (p_atom->getAtomMapNum() != 0) continue;
        const std::string p_fp = atomFingerprintInvariant(p_atom);
        for (auto r_atom : reactant.atoms()) {
            const int mn = r_atom->getAtomMapNum();
            if (mn == 0 || usedInProduct.count(mn)) continue;
            if (atomFingerprintInvariant(r_atom) == p_fp) {
                p_atom->setAtomMapNum(mn);
                usedInProduct.insert(mn);
                break;
            }
        }
    }
}

// Cut bonds bridging core to non-core atoms; return SMILES of non-core
// fragments. Ported from CRS find_and_cut_fragments in
// rdkit-crs-backup/Code/GraphMol/CondensedGraphRxn/RxnCleaning.cpp.
static std::vector<std::string> findAndCutNonCoreFragments(const ROMol& mol, const std::set<int>& coreMapnums) {
    RWMol rw(mol);
    rw.beginBatchEdit();
    for (auto bond : rw.bonds()) {
        const int m1 = bond->getBeginAtom()->getAtomMapNum();
        const int m2 = bond->getEndAtom()->getAtomMapNum();
        if (m1 == m2) continue;
        const bool inCore1 = coreMapnums.count(m1) > 0;
        const bool inCore2 = coreMapnums.count(m2) > 0;
        if (inCore1 != inCore2) {
            rw.removeBond(bond->getBeginAtomIdx(), bond->getEndAtomIdx());
        }
    }
    rw.commitBatchEdit();

    std::vector<ROMOL_SPTR> frags = MolOps::getMolFrags(rw, false);
    std::vector<std::string> out;
    for (const auto& fm : frags) {
        std::set<int> fragMapnums;
        for (auto a : fm->atoms()) fragMapnums.insert(a->getAtomMapNum());
        bool allCore = true;
        bool anyMapped = false;
        for (int mn : fragMapnums) {
            if (mn == 0) continue;
            anyMapped = true;
            if (coreMapnums.count(mn) == 0) { allCore = false; break; }
        }
        if (anyMapped && allCore) continue;
        out.push_back(MolToSmiles(*fm));
    }
    return out;
}

// (leavingGroups, incomingGroups) — same algorithm as Python's
// identify_leaving_groups: map_missing_atoms → core = intersection →
// cut bonds bridging core to non-core → spectator filter.
static std::pair<std::vector<std::string>, std::vector<std::string>>
identifyLeavingGroups(const std::string& r_smiles, const std::string& p_smiles) {
    auto rm = std::unique_ptr<RWMol>(SmilesToMol(r_smiles));
    auto pm = std::unique_ptr<RWMol>(SmilesToMol(p_smiles));
    if (!rm || !pm) return {{}, {}};
    mapMissingAtoms(*rm, *pm);

    std::set<int> rMap, pMap;
    for (auto a : rm->atoms()) if (a->getAtomMapNum() > 0) rMap.insert(a->getAtomMapNum());
    for (auto a : pm->atoms()) if (a->getAtomMapNum() > 0) pMap.insert(a->getAtomMapNum());
    std::set<int> core;
    std::set_intersection(rMap.begin(), rMap.end(), pMap.begin(), pMap.end(),
                          std::inserter(core, core.begin()));

    auto leaving = findAndCutNonCoreFragments(*rm, core);
    auto incoming = findAndCutNonCoreFragments(*pm, core);
    std::sort(leaving.begin(), leaving.end()); leaving.erase(std::unique(leaving.begin(), leaving.end()), leaving.end());
    std::sort(incoming.begin(), incoming.end()); incoming.erase(std::unique(incoming.begin(), incoming.end()), incoming.end());

    // Spectator filter: identical SMILES on both sides → unchanged.
    std::set<std::string> incomingSet(incoming.begin(), incoming.end());
    std::vector<std::string> leavingFiltered;
    for (const auto& s : leaving) if (!incomingSet.count(s)) leavingFiltered.push_back(s);
    std::set<std::string> leavingSet(leaving.begin(), leaving.end());
    std::vector<std::string> incomingFiltered;
    for (const auto& s : incoming) if (!leavingSet.count(s)) incomingFiltered.push_back(s);
    return {std::move(leavingFiltered), std::move(incomingFiltered)};
}

static Atom* atomByMapNum(ROMol& mol, int mapnum) {
    if (mapnum <= 0) return nullptr;
    for (auto a : mol.atoms()) {
        if (a->getAtomMapNum() == mapnum) return a;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------

std::string get_atom_properties(const Atom* atom) {
    std::ostringstream propd;
    propd << "stereo=" << static_cast<int>(atom->getChiralTag()) << ",";
    propd << "charge=" << atom->getFormalCharge() << ",";
    propd << "numHs=" << atom->getTotalNumHs() << ",";
    propd << "hybridization=" << static_cast<int>(atom->getHybridization()) << ",";
    propd << "isAromatic=" << atom->getIsAromatic() << ",";
    propd << "atomicNum=" << atom->getAtomicNum() << ",";
    propd << "isotope=" << atom->getIsotope() << ",";
    propd << "degree=" << atom->getDegree() << ",";
    propd << "implicitValence=" << implicitValence(atom) << ",";
    propd << "explicitValence=" << explicitValence(atom) << ",";
    propd << "numRadicalElectrons=" << atom->getNumRadicalElectrons() << ",";
    propd << "totalDegree=" << atom->getTotalDegree() << ",";
    propd << "totalValence=" << atom->getTotalValence();
    return md5(propd.str());
}

// Bonds present on one side but absent on the other, between two MAPPED atoms.
// Skipping unmapped neighbors avoids spurious correspondences (any two
// unmapped neighbors would otherwise match each other on map-num 0).
std::vector<std::tuple<Atom*, Atom*, std::string>>
identify_changed_bonds(const ROMol& reactant, const ROMol& product,
                       const std::set<int>& atoms_to_keep) {
    std::vector<std::tuple<Atom*, Atom*, std::string>> changed_bonds;
    std::map<int, Atom*> p_by_mapnum;
    for (auto a : product.atoms()) {
        if (a->getAtomMapNum() > 0) p_by_mapnum[a->getAtomMapNum()] = a;
    }
    for (auto r_atom : reactant.atoms()) {
        if (atoms_to_keep.find(r_atom->getIdx()) == atoms_to_keep.end()) continue;
        const int map_num = r_atom->getAtomMapNum();
        if (map_num <= 0) continue;
        auto it = p_by_mapnum.find(map_num);
        if (it == p_by_mapnum.end()) continue;
        Atom* p_atom = it->second;

        std::map<int, Atom*> r_nbr_by_mn, p_nbr_by_mn;
        for (auto a : reactant.atomNeighbors(r_atom)) r_nbr_by_mn[a->getAtomMapNum()] = a;
        for (auto a : product.atomNeighbors(p_atom)) p_nbr_by_mn[a->getAtomMapNum()] = a;

        for (auto r_nbr : reactant.atomNeighbors(r_atom)) {
            if (r_nbr->getAtomMapNum() <= 0) continue;
            const Bond* r_bond = reactant.getBondBetweenAtoms(r_atom->getIdx(), r_nbr->getIdx());
            auto pit = p_nbr_by_mn.find(r_nbr->getAtomMapNum());
            Atom* p_nbr = pit == p_nbr_by_mn.end() ? nullptr : pit->second;
            const Bond* p_bond = p_nbr ? product.getBondBetweenAtoms(p_atom->getIdx(), p_nbr->getIdx()) : nullptr;
            if (!p_bond && r_bond) changed_bonds.emplace_back(r_atom, r_nbr, "product");
        }
        for (auto p_nbr : product.atomNeighbors(p_atom)) {
            if (p_nbr->getAtomMapNum() <= 0) continue;
            const Bond* p_bond = product.getBondBetweenAtoms(p_atom->getIdx(), p_nbr->getIdx());
            auto rit = r_nbr_by_mn.find(p_nbr->getAtomMapNum());
            Atom* r_nbr = rit == r_nbr_by_mn.end() ? nullptr : rit->second;
            const Bond* r_bond = r_nbr ? reactant.getBondBetweenAtoms(r_atom->getIdx(), r_nbr->getIdx()) : nullptr;
            if (!r_bond && p_bond) changed_bonds.emplace_back(p_atom, p_nbr, "reactant");
        }
    }
    return changed_bonds;
}

void add_atom_mapping_to_neighbors(RWMol& reactant, RWMol& product,
                                   const std::vector<std::tuple<Atom*, Atom*, std::string>>& changed_bonds) {
    int max_atom_map_num = 0;
    for (const auto& atom : reactant.atoms()) {
        if (atom->getAtomMapNum() > max_atom_map_num) max_atom_map_num = atom->getAtomMapNum();
    }
    for (const auto& atom : product.atoms()) {
        if (atom->getAtomMapNum() > max_atom_map_num) max_atom_map_num = atom->getAtomMapNum();
    }
    int new_atom_map_num = max_atom_map_num + 1;
    const unsigned int n_product_atoms = product.getNumAtoms();
    for (const auto& [atom1, atom2, context] : changed_bonds) {
        Atom* _atom1 = reactant.getAtomWithIdx(atom1->getIdx());
        for (const auto& neighbor : reactant.atomNeighbors(_atom1)) {
            if (neighbor->getAtomMapNum() != 0) continue;
            if (neighbor->getIdx() >= n_product_atoms) continue;
            Atom* p_neighbor = product.getAtomWithIdx(neighbor->getIdx());
            if (get_atom_properties(neighbor) == get_atom_properties(p_neighbor)) {
                neighbor->setAtomMapNum(new_atom_map_num);
                p_neighbor->setAtomMapNum(new_atom_map_num);
                new_atom_map_num++;
            }
        }
    }
}

// Adds ghost dative bonds where chemistry permits (donor with lone pair +
// electrophilic acceptor). The dative is a graph-merge trick, not a real
// chemical bond; we set "_IgnoreDativeValence" on the resulting mol so that
// rdkit's sanitizer (when patched) treats those bonds as 0-contribution and
// the over-valent SMILES round-trips.
std::pair<std::unique_ptr<RWMol>, std::unique_ptr<RWMol>>
check_and_add_dative_bond(const ROMol& reactant, const ROMol& product,
                          const std::set<int>& atoms_to_keep) {
    auto _reactant = std::make_unique<RWMol>(reactant);
    auto _product = std::make_unique<RWMol>(product);

    auto changed_bonds = identify_changed_bonds(*_reactant, *_product, atoms_to_keep);
    add_atom_mapping_to_neighbors(*_reactant, *_product, changed_bonds);

    std::set<std::tuple<std::string, int, int>> seen;  // (context, lo_map, hi_map)

    for (const auto& [atom1, atom2, context] : changed_bonds) {
        RWMol* target = (context == "reactant") ? _reactant.get() : _product.get();
        const int map1 = atom1->getAtomMapNum();
        const int map2 = atom2->getAtomMapNum();
        if (map1 == 0 || map2 == 0) continue;

        const int lo = std::min(map1, map2);
        const int hi = std::max(map1, map2);
        auto key = std::make_tuple(context, lo, hi);
        if (seen.count(key)) continue;
        seen.insert(key);

        Atom* tA = atomByMapNum(*target, map1);
        Atom* tB = atomByMapNum(*target, map2);
        if (!tA || !tB) continue;

        Atom* donor = nullptr;
        Atom* acceptor = nullptr;
        if (!dativeDonorAcceptor(tA, tB, donor, acceptor)) continue;

        if (target->getBondBetweenAtoms(donor->getIdx(), acceptor->getIdx()) != nullptr) continue;

        target->addBond(donor->getIdx(), acceptor->getIdx(), Bond::BondType::DATIVE);
        // Mark the mol so myrdkit's patched sanitizer (Atom.cpp) treats dative
        // bonds as 0-contribution to valence — see calculateExplicitValence.
        target->setProp("_IgnoreDativeValence", true);
    }
    return std::make_pair(std::move(_reactant), std::move(_product));
}

std::pair<std::unique_ptr<RWMol>, std::set<int>>
remove_disconnected_parts_using_matrix(const ROMol& mol) {
    std::vector<int> labels;
    const int n_components = MolOps::getMolFrags(mol, labels);
    (void)n_components;

    std::set<int> component_to_keep;
    for (const auto& atom : mol.atoms()) {
        if (atom->getAtomMapNum() > 0) {
            component_to_keep.insert(labels[atom->getIdx()]);
        }
    }
    std::set<int> atoms_to_keep;
    const int num_atoms = mol.getNumAtoms();
    for (int idx = 0; idx < num_atoms; ++idx) {
        if (component_to_keep.find(labels[idx]) != component_to_keep.end()) {
            atoms_to_keep.insert(idx);
        }
    }
    auto editable_mol = std::make_unique<RWMol>(mol);
    editable_mol->beginBatchEdit();
    for (int idx = 0; idx < num_atoms; ++idx) {
        if (atoms_to_keep.find(idx) == atoms_to_keep.end()) {
            editable_mol->removeAtom(idx);
        }
    }
    editable_mol->commitBatchEdit();
    return std::make_pair(std::move(editable_mol), atoms_to_keep);
}

bool is_mapping_consistent(const ROMol& reactant, const ROMol& product, bool verbose = false) {
    std::set<int> reactant_mappings, product_mappings;
    for (const auto& atom : reactant.atoms()) {
        if (atom->getAtomMapNum() > 0) reactant_mappings.insert(atom->getAtomMapNum());
    }
    for (const auto& atom : product.atoms()) {
        if (atom->getAtomMapNum() > 0) product_mappings.insert(atom->getAtomMapNum());
    }
    if (verbose) {
        std::cout << "Reactant mappings: ";
        for (auto m : reactant_mappings) std::cout << m << " ";
        std::cout << "\nProduct mappings: ";
        for (auto m : product_mappings) std::cout << m << " ";
        std::cout << "\nMapping consistency: " << (reactant_mappings == product_mappings) << std::endl;
    }
    return reactant_mappings == product_mappings;
}

// myrdkit was built with RDK_BUILD_THREADSAFE_SSS=ON, so SmilesToMol and
// MolToSmiles are safe to call concurrently from different threads (each
// thread parses/writes its own Mol; no shared mutable state). We verified
// this empirically by diffing single-threaded output against 8-threaded
// output on USPTO-50k — they match. Plain calls below.
static std::unique_ptr<RWMol> safeSmilesToMol(const std::string& s) {
    return std::unique_ptr<RWMol>(SmilesToMol(s));
}

static std::string safeMolToSmiles(const ROMol& m) {
    return MolToSmiles(m);
}

std::pair<std::unique_ptr<RWMol>, std::unique_ptr<RWMol>>
clean_and_map_reaction(const std::string& r_smiles, const std::string& p_smiles) {
    auto reactant = safeSmilesToMol(r_smiles);
    auto product = safeSmilesToMol(p_smiles);
    if (!reactant) throw std::invalid_argument("failed to parse reactant SMILES: " + r_smiles);
    if (!product) throw std::invalid_argument("failed to parse product SMILES: " + p_smiles);

    auto [rm_clean, atoms_to_keep_rm] = remove_disconnected_parts_using_matrix(*reactant);
    auto [pm_clean, atoms_to_keep_pm] = remove_disconnected_parts_using_matrix(*product);

    std::set<int> atoms_to_keep;
    atoms_to_keep.insert(atoms_to_keep_rm.begin(), atoms_to_keep_rm.end());
    atoms_to_keep.insert(atoms_to_keep_pm.begin(), atoms_to_keep_pm.end());

    auto [rm_final, pm_final] = check_and_add_dative_bond(*rm_clean, *pm_clean, atoms_to_keep);
    return {std::move(rm_final), std::move(pm_final)};
}

// Mirrors Python's CleanedReaction(fictive, true, mode, leaving_groups, incoming_groups):
//   fictive:         dative-merged single-graph form (for GNN consumers)
//   true_smiles:     disconnected real-valence form (for product reconstruction)
//   mode:            "ORIGINAL" or "WORKAROUND"
//   leaving_groups:  reactant-only fragments after CRS-style core/non-core analysis
//   incoming_groups: symmetric on the product side (rare; data-quality flag)
struct RxnCleanResult {
    std::string fictive;
    std::string true_smiles;
    std::string mode;
    std::vector<std::string> leaving_groups;
    std::vector<std::string> incoming_groups;
};

static std::string trueDisconnectedSmiles(const std::string& r_smiles, const std::string& p_smiles) {
    auto rm = safeSmilesToMol(r_smiles);
    auto pm = safeSmilesToMol(p_smiles);
    if (!rm || !pm) throw std::invalid_argument("failed to parse for true SMILES");
    auto [rm_clean, _r] = remove_disconnected_parts_using_matrix(*rm);
    auto [pm_clean, _p] = remove_disconnected_parts_using_matrix(*pm);
    return safeMolToSmiles(*rm_clean) + ">>" + safeMolToSmiles(*pm_clean);
}

RxnCleanResult rxnclean(const std::string& rxn, bool verbose = false) {
    const auto first_gt = rxn.find('>');
    const auto last_gt = rxn.rfind('>');
    if (first_gt == std::string::npos || last_gt == std::string::npos || first_gt == last_gt) {
        throw std::invalid_argument("expected reaction SMILES of the form 'reactants>>products'");
    }
    const std::string r_smiles = rxn.substr(0, first_gt);
    const std::string p_smiles = rxn.substr(last_gt + 1);

    const std::string true_smi = trueDisconnectedSmiles(r_smiles, p_smiles);
    auto [leaving, incoming] = identifyLeavingGroups(r_smiles, p_smiles);

    auto [rm_final, pm_final] = clean_and_map_reaction(r_smiles, p_smiles);
    if (is_mapping_consistent(*rm_final, *pm_final, verbose)) {
        return {safeMolToSmiles(*rm_final) + ">>" + safeMolToSmiles(*pm_final),
                true_smi, "ORIGINAL", leaving, incoming};
    }

    auto [pm_final_inv, rm_final_inv] = clean_and_map_reaction(p_smiles, r_smiles);
    if (is_mapping_consistent(*rm_final_inv, *pm_final_inv, verbose)) {
        return {safeMolToSmiles(*rm_final_inv) + ">>" + safeMolToSmiles(*pm_final_inv),
                true_smi, "WORKAROUND", leaving, incoming};
    }
    return {safeMolToSmiles(*rm_final) + ">>" + safeMolToSmiles(*pm_final),
            true_smi, "ORIGINAL", leaving, incoming};
}

// ---------------------------------------------------------------------------
// Batch CSV mode — multi-threaded
// ---------------------------------------------------------------------------
//
// RDKit's per-molecule operations (parsing, RWMol mutation, MolToSmiles) hold
// no shared mutable state across distinct Mol objects, so the cleaner is
// embarassingly parallel at the reaction level: each thread owns its own
// RWMol instances and writes only to its assigned output slot. We use a
// std::atomic<size_t> as a work counter (cheaper than a mutex-guarded queue
// for fixed-size workloads) and a single std::mutex only for stderr writes
// from the error path.
//
// USPTO-50k columns: reactants,products,id,split
// We emit:           id,fictive,true,mode

namespace batch {

struct Row { std::string id; std::string rxn; };

static std::vector<Row> readUsptoCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input: " + path);
    std::vector<Row> rows;
    std::string line;
    if (!std::getline(in, line)) return rows;  // header
    while (std::getline(in, line)) {
        // Find the 4 commas; SMILES never contain commas in USPTO-50k.
        size_t c1 = line.find(',');
        if (c1 == std::string::npos) continue;
        size_t c2 = line.find(',', c1 + 1);
        if (c2 == std::string::npos) continue;
        size_t c3 = line.find(',', c2 + 1);
        if (c3 == std::string::npos) continue;
        Row r;
        const std::string r_smi = line.substr(0, c1);
        const std::string p_smi = line.substr(c1 + 1, c2 - c1 - 1);
        r.id = line.substr(c2 + 1, c3 - c2 - 1);
        r.rxn = r_smi + ">>" + p_smi;
        rows.push_back(std::move(r));
    }
    return rows;
}

struct Out {
    std::string id;
    std::string fictive;
    std::string true_smi;
    std::string mode;
    std::string leaving;    // semicolon-separated SMILES
    std::string incoming;   // semicolon-separated SMILES
    std::string err;
};

static std::string joinSemicolon(const std::vector<std::string>& v) {
    std::ostringstream s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s << ';';
        s << v[i];
    }
    return s.str();
}

static void worker(const std::vector<Row>& rows, std::vector<Out>& out,
                   std::atomic<size_t>& next, std::atomic<size_t>& n_done,
                   std::atomic<size_t>& n_err) {
    while (true) {
        const size_t idx = next.fetch_add(1, std::memory_order_relaxed);
        if (idx >= rows.size()) return;
        const Row& r = rows[idx];
        try {
            auto res = rxnclean(r.rxn);
            out[idx] = {r.id, res.fictive, res.true_smiles, res.mode,
                        joinSemicolon(res.leaving_groups),
                        joinSemicolon(res.incoming_groups), ""};
        } catch (const std::exception& e) {
            out[idx] = {r.id, "", "", "ERROR", "", "", e.what()};
            n_err.fetch_add(1, std::memory_order_relaxed);
        }
        n_done.fetch_add(1, std::memory_order_relaxed);
    }
}

static int processCsv(const std::string& input_path, const std::string& output_path, int n_threads) {
    auto rows = readUsptoCsv(input_path);
    std::cout << "Loaded " << rows.size() << " reactions from " << input_path << "\n";
    std::vector<Out> out(rows.size());
    std::atomic<size_t> next{0}, done{0}, err{0};

    if (n_threads <= 0) n_threads = std::max(1u, std::thread::hardware_concurrency());

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back(worker, std::cref(rows), std::ref(out),
                             std::ref(next), std::ref(done), std::ref(err));
    }
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;

    std::ofstream w(output_path);
    if (!w) throw std::runtime_error("cannot open output: " + output_path);
    w << "id,fictive,true,mode,leaving_groups,incoming_groups\n";
    for (const auto& o : out) {
        w << o.id << "," << o.fictive << "," << o.true_smi << "," << o.mode
          << "," << o.leaving << "," << o.incoming << "\n";
    }

    std::cout << "Processed " << rows.size() << " reactions in " << elapsed.count()
              << "s using " << n_threads << " threads ("
              << static_cast<int>(rows.size() / elapsed.count()) << " rxn/s, "
              << err.load() << " errors)\n";
    return 0;
}

}  // namespace batch

static void runSelfTests() {
    auto start = std::chrono::high_resolution_clock::now();
    const int nstep = 100000;
    for (int i = 0; i < nstep; ++i) {
        std::string rxn = "OCCCC[CH3:1].[NH2:2]C>>OCCCC[CH2:1][NH:2]C.CC";
        auto result = rxnclean(rxn);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Single-thread benchmark: " << elapsed.count() / nstep << " s/rxn\n";

    struct Case { std::string rxn; std::string expected_fictive; std::string mode; };
    const Case cases[] = {
        {"OCCCC[CH3:1].[NH2:2]C>>OCCCC[CH2:1][NH:2]C.CC",
         "OCCC[CH2:3][CH3:1]<-[NH2:2][CH3:4]>>OCCC[CH2:3][CH2:1][NH:2][CH3:4]",
         "ORIGINAL"},
        {"CC[CH2:1][NH:2]C.CC>>CC[CH3:1].[NH2:2]C",
         "C[CH2:3][CH2:1][NH:2][CH3:4]>>C[CH2:3][CH3:1]<-[NH2:2][CH3:4]",
         "ORIGINAL"},
    };
    for (const auto& tc : cases) {
        auto result = rxnclean(tc.rxn);
        assert(result.fictive == tc.expected_fictive);
        assert(result.mode == tc.mode);
    }
    std::cout << "Self-tests OK.\n";
}

static void usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << "                                run self-tests + benchmark\n"
              << "  " << prog << " --input INPUT.csv --output OUT.csv [--threads N]\n"
              << "                                                process USPTO-format CSV\n"
              << "                                                (cols: reactants,products,id,split)\n"
              << "                                                threads default = std::thread::hardware_concurrency\n";
}

int main(int argc, char** argv) {
    std::string input_path, output_path;
    int n_threads = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) { input_path = argv[++i]; }
        else if (a == "--output" && i + 1 < argc) { output_path = argv[++i]; }
        else if (a == "--threads" && i + 1 < argc) { n_threads = std::stoi(argv[++i]); }
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::cerr << "unknown arg: " << a << "\n"; usage(argv[0]); return 2; }
    }

    if (input_path.empty()) {
        runSelfTests();
        return 0;
    }
    if (output_path.empty()) {
        std::cerr << "--output is required when --input is given\n";
        usage(argv[0]);
        return 2;
    }
    return batch::processCsv(input_path, output_path, n_threads);
}
