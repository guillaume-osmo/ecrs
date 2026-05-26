# ECRS

**Extended Condensed Reaction String** — a single-string molecular representation
where each molecule is encoded as its own *self-reaction* (the bond breaks that
fragment it into chemically meaningful units, with atom-mapped reconstruction).

Seeded from [rxnclean](https://github.com/thegodone/rxnclean) (atom-mapped reaction
SMILES cleanup + dative-bond detection + a multi-threaded C++ CRS engine), refocused
on the *molecule → ECRS* direction.

## Why ECRS

[SAFE](https://github.com/datamol-io/safe) (Datamol, 2023) represents a molecule
as a concatenation of fragments separated by `.`, then leans on an autoregressive
language model to learn the joining order. The fragmentation is decoupled from
the representation, so the reconstruction is heuristic — the model must learn it.

ECRS takes the opposite view: **the fragmentation IS the representation**. A
molecule's ECRS encodes:

- the constituent fragments,
- the bonds that were broken to obtain them,
- the atom mapping that lets you reform the original molecule exactly.

This is a familiar object — atom-mapped reaction SMILES with the original molecule
as the "reactant" and its fragments as the "products". CRS (Condensed Reaction
String) is the single-string canonical form for that object. So ECRS = CRS where
the input "reaction" happens to be a molecule's own decomposition.

Practical consequences for generative modelling:

- **Reconstruction is exact, not heuristic.** Atom maps tell you which atoms
  reform which bonds.
- **Same downstream stories as SAFE** — scaffold decoration, motif extension,
  linker generation, scaffold morphing — all reduce to ECRS infilling.
- **Reuses the rxnclean C++ engine.** No Python-perf concerns at corpus scale.
- **Fragmentation can be chemically grounded** (retrosynthetic cuts) rather than
  rule-based (BRICS), since the same machinery handles both.

## Status

🚧 Early — repo just scaffolded from rxnclean. The reaction-cleanup path
(`from ecrs import rxnclean`) still works; the molecule → ECRS path is the
work in progress.

## Install

```bash
git clone https://github.com/guillaume-osmo/ECRS
cd ECRS
pip install -e ".[dev]"
```

Requires Python ≥ 3.9 and RDKit ≥ 2025.3 (uses
`Atom.GetValence(ValenceType.IMPLICIT|EXPLICIT)`, which replaced the
deprecated `GetImplicitValence` / `GetExplicitValence` in 2025.03).

## Inherited reaction-cleanup API

```python
from ecrs import rxnclean

rxn = "OCCCC[CH3:1].[NH2:2]C>>OCCCC[CH2:1][NH:2]C.CC"
cleaned, mode = rxnclean(rxn)
# cleaned == "OCCC[CH2:3][CH3:1]->[NH2:2][CH3:4]>>OCCC[CH2:3][CH2:1][NH:2][CH3:4]"
# mode    == "ORIGINAL"
```

## C++ engine

The `cpp/` tree carries the multi-threaded CRS driver (`crsclean`) inherited
from rxnclean. Six modes today (`complete`, `balance`, `clean`, `signature`,
`crs`, `dup`); the molecule → ECRS direction will be added as a new mode that
takes a single molecule + a fragmentation policy and emits the self-reaction
ECRS string.

Build:

```bash
cd cpp && bash compile.sh    # macOS / Linux
```

Not built or installed by `pip`.

## Roadmap

1. **`mol_to_ecrs(smi) -> str`** — Python entry point that fragments via BRICS
   (default, matches SAFE) and emits a CRS-encoded self-reaction.
2. **`ecrs_to_mol(ecrs) -> Mol`** — exact reconstruction from the atom-mapped
   string (atom maps make this deterministic, no model needed).
3. **C++ `CRSwriter` extension** — generalise the existing `crsclean --mode crs`
   driver to accept a single molecule and a fragmentation policy.
4. **Tokenisation for generative models** — char-level baseline + BPE alternative
   trained on ECRS corpora.
5. **Fragment-conditional sampling** — drop-in for the
   [geneva2s](https://github.com/guillaume-osmo/geneva2s) adaptive loop: seed
   = ECRS scaffold + mask, model completes; reconstruct via `ecrs_to_mol`.

## Development

```bash
pip install -e ".[dev]"
python -m pytest                       # all tests
python -m pytest tests/test_basic.py   # hand-written cases only
python -m pytest tests/test_uspto.py   # USPTO-50k regression cases
```

## Project layout

```
.
├── src/ecrs/             # importable package (renamed from rxnclean)
│   ├── __init__.py       # public API re-exports
│   ├── _dative_neighbor.py
│   └── _signature.py
├── tests/                # pytest test suite (renamed imports)
├── cpp/                  # parallel C++ implementation (not installed by pip)
├── pyproject.toml        # build config (hatchling) + pytest + ruff
└── README.md
```

## Credits

Seeded from [rxnclean](https://github.com/thegodone/rxnclean) by Guillaume Godin
and Ruud van Deursen. Same authorship + MIT license.
