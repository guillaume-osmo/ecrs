#!/usr/bin/env bash
# Compile the parallel C++ implementation. Run from anywhere — this script
# changes into its own directory so all paths below are relative to cpp/.
set -euo pipefail
cd "$(dirname "$0")"

# DYLD_FALLBACK_LIBRARY_PATH (not DYLD_LIBRARY_PATH) is required on macOS
# Sonoma+ because conda's libopenblas ships duplicate @loader_path rpaths,
# which the hardened dyld rejects when the path is forced. Fallback is only
# consulted after the rpath search, so duplicate-rpath libs still load via
# their (broken) rpath while libgfortran etc. resolve here.
export DYLD_FALLBACK_LIBRARY_PATH=/Users/tgg/Github/myrdkit/lib:/Users/tgg/miniforge3/envs/rdkit_build_fb/lib:/Users/tgg/miniforge3/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}
g++ -std=c++17 -O2 -o rxnclean rxnclean.cpp md5.cpp -I/Users/tgg/Github/myrdkit/Code  -I/Users/tgg/miniforge3/envs/rdkit_build_fb/include/eigen3 -I/Users/tgg/miniforge3/envs/rdkit_build_fb/include -L/Users/tgg/Github/myrdkit/lib/ -L/Users/tgg/miniforge3/envs/rdkit_build_fb/lib -Wl,-rpath,/Users/tgg/Github/myrdkit/lib -Wl,-rpath,/Users/tgg/miniforge3/envs/rdkit_build_fb/lib -Wl,-rpath,/Users/tgg/miniforge3/lib -lboost_system -lboost_filesystem -lRDKitChemReactions -lRDKitChemTransforms -lRDKitGraphMol -lRDKitSmilesParse -lRDKitSubstructMatch -lRDKitFileParsers -lRDKitRDGeneral -lRDKitForceFieldHelpers -lRDKitMolAlign -lRDKitPartialCharges -lRDKitDescriptors -lRDKitMolTransforms
g++ -std=c++17 -O2 -o bal balance.cpp -I/Users/tgg/Github/myrdkit/Code  -I/Users/tgg/miniforge3/envs/rdkit_build_fb/include/eigen3 -I/Users/tgg/miniforge3/envs/rdkit_build_fb/include -L/Users/tgg/Github/myrdkit/lib/ -L/Users/tgg/miniforge3/envs/rdkit_build_fb/lib -Wl,-rpath,/Users/tgg/Github/myrdkit/lib -Wl,-rpath,/Users/tgg/miniforge3/envs/rdkit_build_fb/lib -Wl,-rpath,/Users/tgg/miniforge3/lib -lboost_system -lboost_filesystem -lRDKitChemReactions -lRDKitChemTransforms -lRDKitGraphMol -lRDKitSmilesParse -lRDKitSubstructMatch -lRDKitFileParsers -lRDKitRDGeneral -lRDKitForceFieldHelpers -lRDKitMolAlign -lRDKitPartialCharges -lRDKitDescriptors -lRDKitMolTransforms

#g++ -std=c++17 -O2 -o rxnmapdiff rxnmapdiff.cpp -I/Users/tgg/Github/myrdkit/Code -I/Users/tgg/miniforge3/envs/rdkit_build_fb/include -L/Users/tgg/Github/myrdkit/lib/ -L/Users/tgg/miniforge3/envs/rdkit_build_fb/lib -Wl,-rpath,/Users/tgg/Github/myrdkit/lib -Wl,-rpath,/Users/tgg/miniforge3/envs/rdkit_build_fb/lib -Wl,-rpath,/Users/tgg/miniforge3/lib -lboost_system -lboost_filesystem -lRDKitChemReactions -lRDKitChemTransforms -lRDKitGraphMol -lRDKitSmilesParse -lRDKitSubstructMatch -lRDKitFileParsers -lRDKitRDGeneral -lRDKitForceFieldHelpers -lRDKitMolAlign -lRDKitPartialCharges -lRDKitDescriptors -lRDKitMolTransforms