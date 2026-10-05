"""Compute the claim-only bound on a file of your own, or on a fetched corpus.

    doubt my_claims.csv                 # columns: claim, evidence, label
    doubt my_claims.jsonl --json
    doubt --corpus vitaminc --split test
    python -m doubt ...                 # same thing

Accepted files: .csv, .tsv, .jsonl, .json (a list of objects), .parquet. Column
names can be changed with --claim-col / --evidence-col / --label-col. Labels
can be any strings; the bound does not care what they mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from . import ceilings, corpus
from .corpus import Claim


class InputError(ValueError):
    """The file cannot be turned into claim/evidence/label rows."""


def _records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ","))
    if suffix == ".jsonl":
        out = []
        with path.open(encoding="utf-8-sig") as handle:
            for n, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise InputError(f"{path}:{n}: not valid JSON ({exc.msg})") from None
        return out
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise InputError(f"{path}: expected a JSON list of objects")
        return data
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    raise InputError(
        f"{path}: unsupported file type {suffix!r}; use csv, tsv, jsonl, json or parquet"
    )


def read_rows(
    path: str | Path,
    claim_col: str = "claim",
    evidence_col: str = "evidence",
    label_col: str = "label",
) -> list[Claim]:
    """Load a file into `Claim` rows, refusing missing or blank claims and labels."""
    path = Path(path)
    if not path.is_file():
        raise InputError(f"{path}: no such file")
    rows = []
    for n, rec in enumerate(_records(path), 1):
        if not isinstance(rec, dict):
            raise InputError(f"{path}: record {n} is not an object")
        values = {}
        for col, required in ((claim_col, True), (label_col, True), (evidence_col, False)):
            value = rec.get(col)
            if value is None or (isinstance(value, float) and value != value):
                value = ""
            value = str(value).strip()
            if required and not value:
                raise InputError(f"{path}: record {n} has no {col!r} (columns: {sorted(rec)})")
            values[col] = value
        rows.append(
            Claim(
                claim=values[claim_col],
                evidence=values[evidence_col],
                label=values[label_col],
                group=values[claim_col],
                source=path.name,
            )
        )
    if not rows:
        raise InputError(f"{path}: no rows")
    return rows


def report(rows: list[Claim]) -> dict:
    """Every number the bound needs, as plain data."""
    claim = ceilings.ceiling(rows, ceilings.by_claim)
    evidence = ceilings.ceiling(rows, ceilings.by_evidence)
    pair = ceilings.ceiling(rows, ceilings.by_pair)
    floor = ceilings.majority(rows)
    labels = Counter(r.label for r in rows)
    return {
        "rows": len(rows),
        "labels": dict(labels.most_common()),
        "distinct_claims": claim.groups,
        "majority": floor,
        "claim_only_ceiling": claim.score,
        "leak": claim.score - floor,
        "evidence_only_ceiling": evidence.score,
        "claim_evidence_ceiling": pair.score,
        "contradictory_rows": pair.ambiguous_rows,
    }


def _print(name: str, r: dict) -> None:
    print(name)
    print(f"  rows                    {r['rows']:>10,}")
    print(f"  distinct claims         {r['distinct_claims']:>10,}")
    for label, count in r["labels"].items():
        print(f"  {label[:22]:<22}  {count:>10,}  {count / r['rows']:.1%}")
    print(f"  majority (floor)        {r['majority']:>10.3f}")
    print(f"  claim-only ceiling      {r['claim_only_ceiling']:>10.3f}")
    print(f"  leak above floor        {r['leak']:>+10.3f}")
    print(f"  evidence-only ceiling   {r['evidence_only_ceiling']:>10.3f}")
    print(
        f"  claim+evidence ceiling  {r['claim_evidence_ceiling']:>10.3f}"
        f"   ({r['contradictory_rows']:,} rows contradict an identical pair)"
    )
    print("\n  No predictor that never reads the evidence can score above the")
    print("  claim-only ceiling on these rows.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="doubt",
        description="Exact upper bound on claim-only (evidence-blind) accuracy.",
    )
    parser.add_argument("file", nargs="?", help="csv / tsv / jsonl / json / parquet")
    parser.add_argument("--corpus", choices=(corpus.VITAMINC, corpus.FEVER))
    parser.add_argument("--split", default="test", choices=corpus.SPLITS)
    parser.add_argument("--claim-col", default="claim")
    parser.add_argument("--evidence-col", default="evidence")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)

    if bool(args.file) == bool(args.corpus):
        parser.error("give either a FILE or --corpus, not both or neither")
    try:
        if args.corpus:
            rows = list(corpus.load(args.corpus, args.split))
            name = f"{args.corpus} / {args.split}"
        else:
            rows = read_rows(args.file, args.claim_col, args.evidence_col, args.label_col)
            name = args.file
    except (InputError, corpus.CorpusMissingError) as exc:
        print(f"doubt: {exc}", file=sys.stderr)
        return 2
    result = report(rows)
    if args.json:
        print(json.dumps({"source": name, **result}, indent=2))
    else:
        _print(name, result)
    return 0
