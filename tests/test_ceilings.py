"""The ceiling arithmetic, on hand-built cases where the answer is obvious.

The whole repository rests on one claim: grouping rows by a field and keeping
the largest label in each group gives the exact best score of any predictor
that sees only that field. These tests check that on examples small enough to
count by hand, because on 370,653 rows the number is unfalsifiable by eye.
"""

from __future__ import annotations

import pytest

from doubt import ceilings
from doubt.corpus import NOT_ENOUGH_INFO, REFUTES, SUPPORTS, Claim


def row(claim: str, evidence: str, label: str, group: str = "g") -> Claim:
    return Claim(claim=claim, evidence=evidence, label=label, group=group, source="test")


def test_ceiling_is_one_when_the_field_determines_the_label():
    rows = [
        row("a", "e1", SUPPORTS),
        row("a", "e2", SUPPORTS),
        row("b", "e3", REFUTES),
    ]
    assert ceilings.ceiling(rows, ceilings.by_claim).score == 1.0


def test_ceiling_is_the_majority_within_each_group():
    """Claim 'a' is 2 SUPPORTS to 1 REFUTES, so one row is unreachable."""
    rows = [
        row("a", "e1", SUPPORTS),
        row("a", "e2", SUPPORTS),
        row("a", "e3", REFUTES),
    ]
    got = ceilings.ceiling(rows, ceilings.by_claim)
    assert got.correct == 2
    assert got.score == pytest.approx(2 / 3)
    assert got.ambiguous_rows == 1


def test_a_perfectly_balanced_pair_caps_at_one_half():
    """The contrastive design, in miniature."""
    rows = [row("a", "e1", SUPPORTS), row("a", "e2", REFUTES)]
    assert ceilings.ceiling(rows, ceilings.by_claim).score == 0.5


def test_evidence_ceiling_uses_the_evidence():
    rows = [
        row("a", "shared", SUPPORTS),
        row("b", "shared", SUPPORTS),
        row("c", "other", REFUTES),
    ]
    assert ceilings.ceiling(rows, ceilings.by_evidence).score == 1.0
    assert ceilings.ceiling(rows, ceilings.by_claim).score == 1.0


def test_pair_ceiling_is_one_unless_the_corpus_contradicts_itself():
    rows = [row("a", "e", SUPPORTS), row("a", "e", REFUTES)]
    assert ceilings.ceiling(rows, ceilings.by_pair).score == 0.5


def test_majority_is_the_floor():
    rows = [row("a", "e1", SUPPORTS), row("b", "e2", SUPPORTS), row("c", "e3", REFUTES)]
    assert ceilings.majority(rows) == pytest.approx(2 / 3)


def test_ceiling_is_never_below_the_floor():
    """Grouping can only help: the all-rows group is itself a valid grouping."""
    rows = [
        row("a", "e1", SUPPORTS),
        row("a", "e2", REFUTES),
        row("b", "e3", SUPPORTS),
        row("c", "e4", NOT_ENOUGH_INFO),
    ]
    assert ceilings.ceiling(rows, ceilings.by_claim).score >= ceilings.majority(rows)


def test_leak_is_zero_when_claims_are_balanced():
    rows = [
        row("a", "e1", SUPPORTS),
        row("a", "e2", REFUTES),
        row("b", "e3", SUPPORTS),
        row("b", "e4", REFUTES),
    ]
    assert ceilings.leak(rows) == pytest.approx(0.0)


def test_leak_is_positive_when_claims_give_it_away():
    rows = [
        row("a", "e1", SUPPORTS),
        row("a", "e2", SUPPORTS),
        row("b", "e3", REFUTES),
        row("c", "e4", SUPPORTS),
    ]
    assert ceilings.leak(rows) > 0


def test_group_shapes_reports_size_and_label_span():
    rows = [
        row("a", "e1", SUPPORTS, group="g1"),
        row("a", "e2", REFUTES, group="g1"),
        row("b", "e3", SUPPORTS, group="g2"),
    ]
    shapes = ceilings.group_shapes(rows)
    assert shapes[(2, 2)] == 1
    assert shapes[(1, 1)] == 1


def test_label_mix_sums_to_one():
    rows = [row("a", "e1", SUPPORTS), row("b", "e2", NOT_ENOUGH_INFO)]
    assert sum(ceilings.label_mix(rows).values()) == pytest.approx(1.0)
