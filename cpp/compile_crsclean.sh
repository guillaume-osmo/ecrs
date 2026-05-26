#!/usr/bin/env bash
# Build crsclean — multi-threaded CRS RXNCompleteMapping driver.
# Links against rdkit-crs-backup (which has the CondensedGraphRxn module
# and the CRSXX bond-type extensions); NOT against myrdkit, since they
# have incompatible BondType enum extensions and the CRS lib was built
# against rdkit-crs.
set -euo pipefail
cd "$(dirname "$0")"

CRS=/Users/tgg/Github/rdkit-crs-backup
LIB=$CRS/build/lib
INC=$CRS/Code

# DYLD_FALLBACK_LIBRARY_PATH is required at runtime for any binary linked
# against the rdkit-crs build (its libraries reference @rpath that isn't
# always resolvable on macOS). We set the rpath at link time below so
# users typically don't need DYLD_* env vars at all.
export DYLD_FALLBACK_LIBRARY_PATH=$LIB:/Users/tgg/miniforge3/envs/rdkit_build_fb/lib:/Users/tgg/miniforge3/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}

g++ -std=c++17 -O2 -pthread -o crsclean crsclean.cpp \
    -I$INC \
    -I/Users/tgg/miniforge3/envs/rdkit_build_fb/include/eigen3 \
    -I/Users/tgg/miniforge3/envs/rdkit_build_fb/include \
    -L$LIB \
    -L/Users/tgg/miniforge3/envs/rdkit_build_fb/lib \
    -Wl,-rpath,$LIB \
    -Wl,-rpath,/Users/tgg/miniforge3/envs/rdkit_build_fb/lib \
    -Wl,-rpath,/Users/tgg/miniforge3/lib \
    -lRDKitCondensedGraphRxn \
    -lRDKitChemReactions -lRDKitChemTransforms -lRDKitGraphMol \
    -lRDKitSmilesParse -lRDKitSubstructMatch -lRDKitFileParsers \
    -lRDKitRDGeneral -lRDKitDescriptors \
    -lboost_system -lboost_filesystem
