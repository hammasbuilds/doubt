"""VitaminC and FEVER, loaded the same way so they can be compared.

Both are three-way fact verification: a claim, a piece of evidence, and a label
in {SUPPORTS, REFUTES, NOT ENOUGH INFO}. They differ in how they were built,
and that difference is the whole subject of this repository.

**FEVER** claims were written by annotators looking at a Wikipedia sentence and
then labelled. A claim appears once, with one label.

**VitaminC** pairs each claim with *revisions* of the same Wikipedia page, so
the same claim appears against evidence that supports it and evidence that does
not. 73,401 of its 112,426 training case groups contain four rows.

That construction is meant to stop a model scoring well by reading the claim and
ignoring the evidence. `ceilings.py` measures whether it worked, exactly rather
than by training something.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data"


def data_dir() -> Path:
    """Where the corpora live: ``$DOUBT_DATA`` if set, else ``<repo>/data``.

    Read on every call, so setting the variable after import still takes effect.
    VitaminC sits at the top level, FEVER in a ``fever/`` subfolder.
    """
    override = os.environ.get("DOUBT_DATA")
    return Path(override).expanduser() if override else DEFAULT_DATA

SUPPORTS = "SUPPORTS"
REFUTES = "REFUTES"
NOT_ENOUGH_INFO = "NOT ENOUGH INFO"
LABELS = (SUPPORTS, REFUTES, NOT_ENOUGH_INFO)

VITAMINC, FEVER = "vitaminc", "fever"
SPLITS = ("train", "validation", "test")

# FEVER's own spelling of the third label.
_FEVER_LABELS = {
    "SUPPORTS": SUPPORTS,
    "REFUTES": REFUTES,
    "NOT ENOUGH INFO": NOT_ENOUGH_INFO,
    "NOT_ENOUGH_INFO": NOT_ENOUGH_INFO,
}


class CorpusMissingError(FileNotFoundError):
    """A corpus is not on disk."""


@dataclass(frozen=True)
class Claim:
    """One claim, one piece of evidence, one label."""

    claim: str
    evidence: str
    label: str
    group: str
    source: str

    @property
    def uncertain(self) -> bool:
        return self.label == NOT_ENOUGH_INFO


def _path(source: str, split: str) -> Path:
    root = data_dir()
    return root / f"{split}.parquet" if source == VITAMINC else root / "fever" / f"{split}.parquet"


def load(source: str = VITAMINC, split: str = "train") -> tuple[Claim, ...]:
    """One split of one corpus, as `Claim` rows. Cached per file path."""
    if source not in (VITAMINC, FEVER):
        raise ValueError(f"unknown source {source!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; have {SPLITS}")

    path = _path(source, split)
    if not path.exists():
        raise CorpusMissingError(
            f"{path} is missing. Run scripts/fetch_data.py, which pulls both "
            "corpora from their HuggingFace parquet conversions (set DOUBT_DATA "
            "to keep them elsewhere)."
        )
    return _load(str(path), source)


@lru_cache(maxsize=8)
def _load(path: str, source: str) -> tuple[Claim, ...]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if source == VITAMINC:
        return tuple(
            Claim(
                claim=r["claim"],
                evidence=r["evidence"],
                label=r["label"],
                # Rows sharing a case_id are revisions of one another. The
                # grouping is what makes the contrastive structure visible.
                group=r["case_id"],
                source=VITAMINC,
            )
            for r in rows
        )

    # FEVER's gold-evidence release carries evidence as a list of
    # [page, sentence_id, text] triples; the text is what a model would read.
    out = []
    for r in rows:
        evidence = r.get("evidence")
        if isinstance(evidence, list):
            evidence = " ".join(
                part[-1] if isinstance(part, list) else str(part) for part in evidence
            )
        out.append(
            Claim(
                claim=r["claim"],
                evidence=evidence or "",
                label=_FEVER_LABELS.get(r["label"], r["label"]),
                group=str(r.get("id", "")),
                source=FEVER,
            )
        )
    return tuple(out)


def available() -> dict[str, list[str]]:
    """Which corpora are actually on disk, so a partial download is visible."""
    return {
        source: [s for s in SPLITS if _path(source, s).exists()]
        for source in (VITAMINC, FEVER)
    }
