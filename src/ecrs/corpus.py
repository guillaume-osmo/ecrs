"""Convert a SMILES corpus to ECRS strings (one per line).

CLI:
    python -m ecrs.corpus convert \\
        --in  data/chembl_9k_organic.smi \\
        --out data/chembl_9k_organic.ecrs.txt

Skips molecules that can't be encoded (logged with counts at the end).
Per-line format: `<ecrs>\\t<original_smiles>` so downstream consumers can
correlate without a round-trip.

Use `--stats-only` for a dry-run that just prints the success-rate table.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ecrs._safe_ecrs import mol_to_ecrs


def _iter_smi(path: Path):
    with open(path) as f:
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            if s:
                yield s


def convert(in_path: Path, out_path: Path | None, limit: int | None = None,
            verbose: bool = True) -> dict:
    counts: Counter = Counter()
    out_fh = open(out_path, "w") if out_path is not None else None
    try:
        for i, smi in enumerate(_iter_smi(in_path)):
            if limit and i >= limit:
                break
            result = mol_to_ecrs(smi)
            status = result.status if not result.status.startswith("crs_err") else "crs_err"
            counts[status] += 1
            if result.ecrs is not None and out_fh is not None:
                out_fh.write(f"{result.ecrs}\t{smi}\n")
    finally:
        if out_fh is not None:
            out_fh.close()
    n = sum(counts.values())
    if verbose:
        print(f"\nProcessed {n:,} SMILES from {in_path}")
        for k in ("ok", "no_brics", "parse_err", "sanitize_err", "crs_err"):
            if counts.get(k):
                print(f"  {k:13s} {counts[k]:>7,d}  ({100*counts[k]/max(n,1):5.1f}%)")
        if out_path is not None:
            print(f"\nWrote ECRS-encoded entries to {out_path}")
    return dict(counts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    pc = sub.add_parser("convert", help="SMILES file -> ECRS file")
    pc.add_argument("--in", dest="in_path", required=True, help="Input .smi (one SMILES per line)")
    pc.add_argument("--out", dest="out_path", default=None, help="Output .ecrs.txt (omit for --stats-only)")
    pc.add_argument("--limit", type=int, default=None, help="Process only first N SMILES (debugging)")
    pc.add_argument("--stats-only", action="store_true", help="Don't write output, just report status counts")
    args = p.parse_args(argv)
    if args.stats_only:
        args.out_path = None
    elif args.out_path is None:
        print("error: --out is required (or pass --stats-only for a dry-run)", file=sys.stderr)
        return 2
    convert(Path(args.in_path), Path(args.out_path) if args.out_path else None,
            limit=args.limit, verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
