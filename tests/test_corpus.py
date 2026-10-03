"""Both corpora, loaded from the real parquet.

The point of the repository is a comparison, so the tests that matter are the
ones asserting the two corpora are loaded the same way and are what they claim
to be.
"""

from __future__ import annotations

import pytest

from doubt import ceilings, corpus

_missing = [
    f"{source}/{split}"
    for source, splits in ((corpus.VITAMINC, corpus.SPLITS), (corpus.FEVER, corpus.SPLITS))
    for split in splits
    if split not in corpus.available()[source]
]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=f"corpus not on disk ({', '.join(_missing)}); run `python scripts/fetch_data.py` "
    "or point DOUBT_DATA at a folder that has it",
)

SIZES = {
    (corpus.VITAMINC, "train"): 370_653,
    (corpus.VITAMINC, "validation"): 63_054,
    (corpus.VITAMINC, "test"): 55_197,
    (corpus.FEVER, "train"): 228_277,
    (corpus.FEVER, "validation"): 15_935,
    (corpus.FEVER, "test"): 16_039,
}


@pytest.mark.parametrize("key,expected", SIZES.items(), ids=[f"{s}-{p}" for s, p in SIZES])
def test_split_sizes(key, expected):
    source, split = key
    assert len(corpus.load(source, split)) == expected


@pytest.mark.parametrize("source", [corpus.VITAMINC, corpus.FEVER])
def test_labels_are_the_three_known_ones(source):
    """FEVER spells the third label with an underscore in some releases."""
    labels = {row.label for row in corpus.load(source, "test")}
    assert labels <= set(corpus.LABELS), labels


@pytest.mark.parametrize("source", [corpus.VITAMINC, corpus.FEVER])
def test_every_row_has_a_claim_and_evidence(source):
    rows = corpus.load(source, "test")
    assert all(r.claim.strip() for r in rows)
    # FEVER's NEI rows carry retrieved rather than gold evidence, but never none.
    assert sum(1 for r in rows if not r.evidence.strip()) / len(rows) < 0.01


def test_fever_evidence_is_flattened_to_text():
    """It ships as [page, sentence_id, text] triples; a model reads the text."""
    row = corpus.load(corpus.FEVER, "test")[0]
    assert isinstance(row.evidence, str)
    assert "[" not in row.evidence[:1]


def test_vitaminc_claims_are_almost_all_contrastive():
    """The design property the whole finding rests on."""
    rows = corpus.load(corpus.VITAMINC, "test")
    shapes = ceilings.group_shapes(rows)
    multi = sum(c for (_, labels), c in shapes.items() if labels > 1)
    assert multi / sum(shapes.values()) > 0.95


def test_vitaminc_claim_ceiling_is_at_the_floor():
    """No claim-only function can beat guessing the commonest label.

    This is the headline, and it is an exact bound rather than a measurement:
    grouping by claim and keeping each group's largest label is the best any
    predictor blind to the evidence could do.
    """
    rows = corpus.load(corpus.VITAMINC, "test")
    assert ceilings.leak(rows) < 0.01


def test_fever_claim_ceiling_is_near_perfect():
    """The contrast. FEVER permits a claim-only function to be nearly right.

    It does not show that FEVER models use shortcuts — only that FEVER cannot
    rule it out, where VitaminC can.
    """
    rows = corpus.load(corpus.FEVER, "test")
    assert ceilings.ceiling(rows, ceilings.by_claim).score > 0.99


@pytest.mark.parametrize("source", [corpus.VITAMINC, corpus.FEVER])
def test_train_and_test_claims_barely_overlap(source):
    """Neither corpus is memorisable across splits, so the ceilings above are
    about claim-only *functions*, not about having seen the claim before."""
    train = {r.claim for r in corpus.load(source, "train")}
    test = corpus.load(source, "test")
    overlap = sum(1 for r in test if r.claim in train) / len(test)
    assert overlap < 0.02
