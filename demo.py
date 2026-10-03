"""The bound in thirty seconds, with or without the corpora on disk.

    python demo.py

First on eight hand-built rows (no download needed), then on the VitaminC and
FEVER test splits if scripts/fetch_data.py has been run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from doubt import corpus  # noqa: E402
from doubt.cli import report  # noqa: E402
from doubt.corpus import NOT_ENOUGH_INFO as NEI  # noqa: E402
from doubt.corpus import REFUTES as R  # noqa: E402
from doubt.corpus import SUPPORTS as S  # noqa: E402
from doubt.corpus import Claim  # noqa: E402


def row(claim: str, evidence: str, label: str) -> Claim:
    return Claim(claim, evidence, label, claim, "demo")


# Contrastive: each claim seen against evidence that supports and refutes it.
CONTRASTIVE = [
    row("Paris has 2.1M people", "2018: 2.1M", S), row("Paris has 2.1M people", "2023: 2.0M", R),
    row("Oslo is the capital", "Oslo is the capital", S), row("Oslo is the capital", "-", NEI),
    row("X won in 2020", "X won 2020", S), row("X won in 2020", "Y won 2020", R),
    row("Rome is in Spain", "Rome is in Italy", R), row("Rome is in Spain", "Rome, Spain", S),
]
# Written once per claim: the claim alone determines the label.
ONE_SHOT = [
    row(f"claim {i}", f"evidence {i}", label)
    for i, label in enumerate([S, S, R, NEI, S, R, NEI, S])
]


def show(name: str, rows) -> None:
    r = report(list(rows))
    print(f"  {name:<34}{r['rows']:>8,} rows   majority {r['majority']:.3f}"
          f"   claim-only ceiling {r['claim_only_ceiling']:.3f}   leak {r['leak']:+.3f}")


def main() -> None:
    print("hand-built rows")
    show("contrastive (VitaminC-like)", CONTRASTIVE)
    show("one claim, one label (FEVER-like)", ONE_SHOT)

    print("\nreal corpora (test split)")
    have = corpus.available()
    for source in (corpus.VITAMINC, corpus.FEVER):
        if "test" in have[source]:
            show(source, corpus.load(source, "test"))
        else:
            print(f"  {source:<34}not on disk; run `python scripts/fetch_data.py`"
                  " (or set DOUBT_DATA)")


if __name__ == "__main__":
    main()
