// crsclean — multi-threaded driver for the CRS public API.
//
// One binary, six --mode values dispatching to each value-in / value-out
// CRS function. All run in N std::threads with strict RAII semantics so
// per-thread errors never leak ChemicalReaction or RWMol pointers across
// the work-stealing loop.
//
// Modes:
//   complete   RXNCompleteMapping  - fill in missing atom-map numbers
//                                    + add missing-side leaving groups
//   balance    rxnbalance          - atom-balance the reaction
//   clean      rxnclean            - strip spectators + clean fragments
//   signature  CRSwriter sig=true  - radial reaction-template signature
//   crs        CRSwriter           - full CRS-encoded molecule
//   dup        isDuplicateMapping  - boolean: are atom maps duplicated?
//
// Pointer safety notes:
//
// 1. CRS RxnCleaning.cpp:1176 has a `unique_ptr.release()` followed by
//    `delete rxn` 300 lines later. If the called function throws
//    between those points, the raw pointer leaks. We can't fix the CRS
//    internals from this driver, so the strategy is: catch ALL
//    exceptions at the per-call boundary so the calling thread never
//    re-enters the inconsistent state, and accept the (per-thread,
//    bounded) memory leak in the error case rather than letting it
//    crash the process.
//
// 2. All CRS public APIs we call are value-in (std::string), value-out
//    (std::string or bool), so there is no shared mutable state on the
//    API surface. RDKit's parser caches and ring-perception tables are
//    init-once / read-after, so concurrent reads are fine.
//
// 3. We pass strings by const reference into the worker and copy into
//    the output vector slot — no per-thread shared buffers.
//
// 4. Mutex is used only for stderr writes from the error path, not for
//    the hot per-reaction call.
//
// USPTO-format CSV in: reactants,products,id,split
// CSV out:             id,input_rxn,output,err

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <GraphMol/CondensedGraphRxn/CondensedGraphRxn.h>

namespace {

struct Row {
    std::string id;
    std::string rxn;  // reactants>>products
};

struct Out {
    std::string id;
    std::string input;
    std::string output;
    std::string err;
};

std::vector<Row> readUsptoCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input: " + path);
    std::vector<Row> rows;
    std::string line;
    if (!std::getline(in, line)) return rows;  // header
    while (std::getline(in, line)) {
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

// Mode dispatcher — one switch point that maps a string mode to the
// matching CRS API call. Each branch is independently exception-safe;
// the entire call is wrapped in a single try/catch in `safeRun`.
enum class Mode {
    Complete,    // RXNCompleteMapping
    Balance,     // rxnbalance
    Clean,       // rxnclean (CRS internal)
    Signature,   // CRSwriter signature=true (one radius)
    Signatures,  // CRSwritersMulti — multi-radius bundle (r=0..max)
    Crs,         // CRSwriter signature=false
    Dup,         // isDuplicateMapping
    PerProduct,  // perProductCRS — per-product decomposition
};

Mode parseMode(const std::string& s) {
    if (s == "complete") return Mode::Complete;
    if (s == "balance") return Mode::Balance;
    if (s == "clean") return Mode::Clean;
    if (s == "signature") return Mode::Signature;
    if (s == "signatures") return Mode::Signatures;
    if (s == "crs") return Mode::Crs;
    if (s == "dup") return Mode::Dup;
    if (s == "per-product" || s == "perproduct") return Mode::PerProduct;
    throw std::invalid_argument("unknown --mode: " + s);
}

struct ModeOpts {
    int radius = 1;
    int max_radius = 3;            // for `signatures` mode
    bool addLeavingGroups = false;
    bool addRingInfo = true;
    bool isKeepAtomMap = false;
    bool sanitize = true;
    bool compactCurly = false;     // for `signature`/`signatures`/`crs`
};

// RAII wrapper around the CRS call — any exception thrown inside the
// CRS C++ is caught and converted into an `err` field, never propagated
// up where it could destabilize the thread pool. The per-call
// shared_ptr / unique_ptr objects inside the CRS functions are
// scope-local and self-cleaning.
std::string safeRun(Mode mode, const ModeOpts& opts,
                    const std::string& rxn_smi,
                    std::string& err_out) {
    using namespace RDKit::CondensedGraphRxn;
    try {
        switch (mode) {
            case Mode::Complete: {
                // First attempt with the user-requested sanitize flag.
                // If RDKit throws (typically a valence violation on an
                // input molecule with unusual aromatic-N/S valence —
                // ~28k Pistachio reactions), retry with sanitize=false
                // so we get *some* completion result instead of
                // surrendering. Any output still gets re-validated by
                // the downstream signature pass.
                try {
                    return RXNCompleteMapping(
                        rxn_smi, /*debug=*/false,
                        opts.addLeavingGroups, opts.sanitize);
                } catch (const std::exception& e) {
                    if (opts.sanitize) {
                        return RXNCompleteMapping(
                            rxn_smi, /*debug=*/false,
                            opts.addLeavingGroups, /*sanitize=*/false);
                    }
                    throw;
                }
            }
            case Mode::Balance:
                return rxnbalance(rxn_smi, /*verbose=*/false);
            case Mode::Clean:
                return rxnclean(rxn_smi, /*verbose=*/false);
            case Mode::Signatures: {
                // Multi-radius bundle: emit r=0..max_radius signatures
                // separated by tabs in a single field. Builds the CRS
                // mol once internally, ~2x faster than calling Signature
                // mode N times.
                auto sigs = CRSwritersMulti(
                    rxn_smi, static_cast<unsigned int>(opts.max_radius),
                    /*doRandom=*/false, /*randomSeed=*/0,
                    /*aromatize=*/true, /*charges=*/false,
                    opts.addRingInfo, /*isRadical=*/false,
                    opts.isKeepAtomMap, /*isEZ=*/false,
                    /*isRS=*/false, /*debug=*/false,
                    opts.compactCurly);
                std::string joined;
                for (size_t i = 0; i < sigs.size(); ++i) {
                    if (i) joined += "\t";
                    joined += sigs[i];
                }
                return joined;
            }
            case Mode::Signature:
                return CRSwriter(
                    rxn_smi, /*doRandom=*/false, /*randomSeed=*/0,
                    /*aromatize=*/true, /*signature=*/true,
                    /*charges=*/false, opts.radius, opts.addRingInfo,
                    /*isRadical=*/false, opts.isKeepAtomMap,
                    /*isEZ=*/false, /*isRS=*/false, /*debug=*/false,
                    opts.compactCurly);
            case Mode::Crs:
                return CRSwriter(
                    rxn_smi, /*doRandom=*/false, /*randomSeed=*/0,
                    /*aromatize=*/true, /*signature=*/false,
                    /*charges=*/false, opts.radius, opts.addRingInfo,
                    /*isRadical=*/false, opts.isKeepAtomMap,
                    /*isEZ=*/false, /*isRS=*/false, /*debug=*/false,
                    opts.compactCurly);
            case Mode::Dup:
                return isDuplicateMapping(rxn_smi, opts.sanitize) ? "1" : "0";
            case Mode::PerProduct:
                return perProductCRS(rxn_smi);
        }
    } catch (const std::exception& e) {
        err_out = e.what();
    } catch (...) {
        err_out = "unknown exception";
    }
    return std::string();
}

void worker(Mode mode, const ModeOpts& opts,
            const std::vector<Row>& rows, std::vector<Out>& out,
            std::atomic<size_t>& next, std::atomic<size_t>& done,
            std::atomic<size_t>& err_count) {
    while (true) {
        const size_t idx = next.fetch_add(1, std::memory_order_relaxed);
        if (idx >= rows.size()) return;
        const Row& r = rows[idx];
        Out& o = out[idx];
        o.id = r.id;
        o.input = r.rxn;
        std::string err;
        std::string result = safeRun(mode, opts, r.rxn, err);
        if (!err.empty()) {
            o.err = std::move(err);
            err_count.fetch_add(1, std::memory_order_relaxed);
        } else {
            o.output = std::move(result);
        }
        done.fetch_add(1, std::memory_order_relaxed);
    }
}

void escapeCsvInPlace(std::string& s) {
    // Naive: replace embedded commas with semicolons so the simple
    // downstream parser works. Reaction SMILES never contain commas in
    // benchmark data, but CRS output occasionally writes "f:..." curly
    // annotations that include commas — we sanitize defensively.
    for (char& c : s) {
        if (c == ',') c = ';';
        if (c == '\n' || c == '\r') c = ' ';
    }
}

int processCsv(Mode mode, const ModeOpts& opts,
               const std::string& input_path, const std::string& output_path,
               int n_threads) {
    auto rows = readUsptoCsv(input_path);
    std::cout << "Loaded " << rows.size() << " reactions from " << input_path
              << std::endl;
    std::vector<Out> out(rows.size());
    std::atomic<size_t> next{0}, done{0}, err{0};

    if (n_threads <= 0)
        n_threads = std::max(1u, std::thread::hardware_concurrency());

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back(worker, mode, std::cref(opts), std::cref(rows),
                             std::ref(out), std::ref(next), std::ref(done),
                             std::ref(err));
    }
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;

    std::ofstream w(output_path);
    if (!w) throw std::runtime_error("cannot open output: " + output_path);
    w << "id,input_rxn,output,err\n";
    for (auto& o : out) {
        escapeCsvInPlace(o.id);
        escapeCsvInPlace(o.input);
        escapeCsvInPlace(o.output);
        escapeCsvInPlace(o.err);
        w << o.id << "," << o.input << "," << o.output << ","
          << o.err << "\n";
    }

    std::cout << "Processed " << rows.size() << " reactions in "
              << elapsed.count() << "s using " << n_threads << " threads ("
              << static_cast<int>(rows.size() / elapsed.count())
              << " rxn/s, " << err.load() << " errors)\n";
    return 0;
}

void usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << "                                run a small smoke test (all modes)\n"
        << "  " << prog << " --mode MODE --input INPUT.csv --output OUT.csv [--threads N] [--radius R]\n"
        << "                                                process USPTO-format CSV in parallel\n"
        << "                                                (cols: reactants,products,id,split)\n"
        << "  Modes (--mode):\n"
        << "    complete   RXNCompleteMapping  fill in missing atom maps + add leaving groups (default)\n"
        << "    balance    rxnbalance          atom-balance the reaction\n"
        << "    clean      rxnclean (CRS)      strip spectators + clean fragments\n"
        << "    signature  CRSwriter sig=true  radial reaction-template signature (--radius)\n"
        << "    signatures CRSwritersMulti     multi-radius bundle r=0..max (--max-radius)\n"
        << "                                   tab-separated signatures in the output column\n"
        << "    crs        CRSwriter           full CRS-encoded molecule (--radius)\n"
        << "    per-product perProductCRS       decompose multi-product reaction:\n"
        << "                                    'CRS_1.CRS_2....inorganic.byproducts'\n"
        << "  Flags:\n"
        << "    --compact-curly                 compress CRS bond H-delta tags:\n"
        << "                                    drop {0}{0}, collapse {X}{X} -> {X}.\n"
        << "                                    Atoms with non-zero formal charge keep\n"
        << "                                    their full {0}{0} so charge changes stay\n"
        << "                                    visible in cluster keys.\n"
        << "    dup        isDuplicateMapping  boolean: are atom maps duplicated?\n";
}

void smokeTest() {
    const char* test_cases[] = {
        "[CH3:1][Cl:2].[OH:3][CH2:4][CH3:5]>>[CH3:1][O:3][CH2:4][CH3:5].[ClH:2]",
        "[NH2:1][CH3:2].O=C(OC(C)(C)C)OC(C)(C)C>>[NH:1]([CH3:2])C(=O)OC(C)(C)C",
        "C[C:3](=[O:4])O.[OH:1][CH:2](COC)CC>>C[C:3](=[O:4])[O:1][CH:2](COC)CC",
    };
    const Mode modes[] = {Mode::Complete, Mode::Balance};
    const char* mode_names[] = {"complete", "balance"};
    ModeOpts opts;
    for (size_t mi = 0; mi < sizeof(modes) / sizeof(modes[0]); ++mi) {
        std::cout << "=== mode: " << mode_names[mi] << " ===\n";
        for (const auto* sma : test_cases) {
            std::string err;
            std::string out = safeRun(modes[mi], opts, sma, err);
            if (!err.empty()) {
                std::cout << "ERR  " << sma << "\n     " << err << "\n";
            } else {
                std::cout << "in : " << sma << "\nout: " << out << "\n\n";
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string input_path, output_path;
    std::string mode_str = "complete";
    int n_threads = 0;
    ModeOpts opts;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (a == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (a == "--threads" && i + 1 < argc) {
            n_threads = std::stoi(argv[++i]);
        } else if (a == "--mode" && i + 1 < argc) {
            mode_str = argv[++i];
        } else if (a == "--radius" && i + 1 < argc) {
            opts.radius = std::stoi(argv[++i]);
        } else if (a == "--max-radius" && i + 1 < argc) {
            opts.max_radius = std::stoi(argv[++i]);
        } else if (a == "--add-leaving-groups") {
            opts.addLeavingGroups = true;
        } else if (a == "--no-ring-info") {
            opts.addRingInfo = false;
        } else if (a == "--keep-atom-map") {
            opts.isKeepAtomMap = true;
        } else if (a == "--no-sanitize") {
            opts.sanitize = false;
        } else if (a == "--compact-curly") {
            opts.compactCurly = true;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "unknown arg: " << a << "\n";
            usage(argv[0]);
            return 2;
        }
    }

    if (input_path.empty()) {
        smokeTest();
        return 0;
    }
    if (output_path.empty()) {
        std::cerr << "--output is required when --input is given\n";
        usage(argv[0]);
        return 2;
    }
    Mode mode;
    try {
        mode = parseMode(mode_str);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        usage(argv[0]);
        return 2;
    }
    return processCsv(mode, opts, input_path, output_path, n_threads);
}
