"""The best score obtainable while ignoring part of the input.

A model that reads only the claim and never the evidence is not doing fact
verification. On most such benchmarks it nevertheless scores far above chance,
because of how the claims were written.

The usual way to show this is to train a claim-only classifier and report what
it gets. That measures the classifier as much as the benchmark, and it leaves
open whether a better one would do better.

This computes the **exact ceiling** instead. Group the rows by claim; a
claim-only predictor must give every row in a group the same answer, so the
most it can get right in that group is the size of its largest label. Sum over
groups and divide. No model can beat that number, and no model needs to be
trained to find it.

    ceiling(rows, by_claim)   # nothing that ignores evidence can exceed this
    ceiling(rows, by_evidence)
    majority(rows)            # the floor: always predict the commonest label

The gap between `majority` and the claim ceiling is exactly how much a claim
leaks. If they are equal, the claim leaks nothing at all.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from .corpus import LABELS


def by_claim(row) -> str:
    return row.claim


def by_evidence(row) -> str:
    return row.evidence


def by_pair(row) -> tuple[str, str]:
    """Both. The ceiling here is 1.0 unless the corpus is self-contradictory,
    so it doubles as a check that the same (claim, evidence) never carries two
    different labels."""
    return row.claim, row.evidence


@dataclass(frozen=True)
class Ceiling:
    """What a predictor blind to everything outside `key` can reach."""

    name: str
    groups: int
    rows: int
    correct: int

    @property
    def score(self) -> float:
        return self.correct / self.rows if self.rows else 0.0

    @property
    def ambiguous_rows(self) -> int:
        """Rows in a group whose label is not unanimous."""
        return self.rows - self.correct


def _require_rows(rows) -> list:
    rows = list(rows)
    if not rows:
        raise ValueError("no rows: a ceiling over an empty set is undefined")
    return rows


def ceiling(rows, key: Callable, name: str = "") -> Ceiling:
    rows = _require_rows(rows)
    grouped: dict[object, Counter] = defaultdict(Counter)
    for row in rows:
        grouped[key(row)][row.label] += 1
    correct = sum(counts.most_common(1)[0][1] for counts in grouped.values())
    return Ceiling(
        name=name or getattr(key, "__name__", "key"),
        groups=len(grouped),
        rows=len(rows),
        correct=correct,
    )


def majority(rows) -> float:
    """Always predict the commonest label. The floor everything is measured from."""
    rows = _require_rows(rows)
    counts = Counter(row.label for row in rows)
    return counts.most_common(1)[0][1] / len(rows)


def label_mix(rows) -> dict[str, float]:
    rows = _require_rows(rows)
    counts = Counter(row.label for row in rows)
    return {label: counts[label] / len(rows) for label in LABELS}


def group_shapes(rows) -> Counter:
    """How many rows share a group, and how many labels each group spans.

    VitaminC's design shows up here: most groups hold four rows spanning two
    labels, which is the contrastive pair structure.
    """
    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[row.group].append(row)
    return Counter(
        (len(members), len({m.label for m in members})) for members in grouped.values()
    )


def leak(rows) -> float:
    """How much the claim gives away, in points above the floor.

    Zero means a claim-only model can do no better than guessing the commonest
    label — which is the strongest thing a verification benchmark can say about
    itself.
    """
    return ceiling(rows, by_claim).score - majority(rows)
