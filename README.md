# doubt

> No model that ignores the evidence can score above **0.502** on VitaminC — where always guessing the commonest label gets **0.501**. On FEVER the same bound is **0.999**.

**Status:** complete as a measurement. Nothing is trained. Every figure is an exact property
of the corpora, obtained by grouping rows and counting.

## The idea

A fact-verification model that reads only the claim, never the evidence, is not verifying
anything. The usual way to show a benchmark permits that is to train a claim-only
classifier and report its score — which measures the classifier as much as the benchmark,
and leaves open whether a better one would do better.

There is an exact answer instead. Group the rows by claim. A claim-only predictor must give
every row in a group the same label, so the most it can get right in that group is the size
of the group's largest label. Sum over groups, divide by rows:

```
ceiling = sum(max(label_counts) for each claim) / total_rows
```

**No predictor that ignores the evidence can beat that number**, however it is built, and
nothing has to be trained to find it.

## The corpora

| | Rows (test) | SUPPORTS | REFUTES | NOT ENOUGH INFO |
|---|---:|---:|---:|---:|
| [VitaminC](https://huggingface.co/datasets/tals/vitaminc) | 55,197 | 47.3% | 39.5% | 13.2% |
| [FEVER](https://huggingface.co/datasets/copenlu/fever_gold_evidence) (gold evidence) | 16,039 | 40.3% | 19.4% | 40.3% |

VitaminC pairs each claim with *successive revisions* of the same Wikipedia page, so the
same claim appears against evidence that supports it and evidence that does not. FEVER's
claims were written once against one sentence.

```
pip install -e ".[dev]"
python demo.py                 # the bound on hand-built rows; real corpora if fetched
python scripts/fetch_data.py   # ~95 MB across both corpora, not in git
python scripts/measure.py      # every table below
python -m pytest               # 44 tests; 16 need the corpora and skip without them
```

`fetch_data.py` downloads in byte ranges to `*.parquet.part` and renames only when the
size matches, so an interrupted run resumes where it stopped. Set `DOUBT_DATA` to keep the
corpora somewhere other than `./data` (VitaminC at the top level, FEVER in `fever/`).

### Your own data

```
doubt my_claims.csv                      # columns: claim, evidence, label
doubt my_claims.jsonl --claim-col q --label-col y --json
doubt --corpus vitaminc --split test     # the fetched corpora
```

Reads csv, tsv, jsonl, json (a list of objects) and parquet; labels can be any strings.
It prints the majority floor, the claim-only ceiling, the leak between them, the
evidence-only ceiling, and how many rows contradict an identical claim/evidence pair.
`--json` gives the same numbers for scripts. `python -m doubt` works too.

## The bound

| Corpus | Split | Rows | Majority | **Claim-only ceiling** | Leak | Evidence-only ceiling |
|---|---|---:|---:|---:|---:|---:|
| vitaminc | train | 370,653 | 0.501 | **0.502** | +0.001 | 0.600 |
| vitaminc | validation | 63,054 | 0.499 | **0.502** | +0.003 | 0.590 |
| vitaminc | test | 55,197 | 0.501 | **0.502** | +0.001 | 0.593 |
| fever | train | 228,277 | 0.503 | **0.997** | +0.494 | 0.728 |
| fever | validation | 15,935 | 0.402 | **0.998** | +0.596 | 0.718 |
| fever | test | 16,039 | 0.403 | **0.999** | +0.596 | 0.722 |

On VitaminC the ceiling sits **one thousandth** above the floor. Reading the claim tells you
nothing you did not already know from the label prior.

## Why VitaminC has no room

Each group is one claim against revisions of one page (train split):

| Rows in group | Labels spanned | Groups |
|---:|---:|---:|
| 4 | 2 | 73,328 |
| 2 | 2 | 37,955 |
| 1 | 1 | 1,015 |
| 4 | 3 | 73 |
| 2 | 1 | 44 |
| 3 | 2 | 8 |

**99.1% of groups span more than one label.** The same claim is both true and false
depending on which revision you read, and a claim-only predictor has to give them all one
answer.

## About the FEVER number

**It is not evidence that FEVER models take shortcuts.** It is an upper bound on a
hypothesis class, not a measurement of any model.

It is also not an artefact of unique claims. FEVER's test claims repeat — 9,854 distinct
claims over 16,039 rows, 66% of rows sharing a claim with another — and the repeats
essentially always carry the same label. So grouping by claim still reaches 0.999.

Nor is it memorisation across splits. **0.1% of test rows have a claim that appears in
train** in both corpora (23 claims in VitaminC, 2 in FEVER), and in VitaminC not one of
those 23 carries a single consistent training label.

What it means is narrower and, I think, more interesting: **a claim-only function could in
principle be nearly perfect on FEVER, and provably cannot be on VitaminC.** FEVER cannot
rule the shortcut out. VitaminC can. That asymmetry is the argument for building benchmarks
contrastively — it is the only construction that turns "we looked and didn't find a
shortcut" into "there isn't one".

## The label everybody drops

| | Rows | NOT ENOUGH INFO | Two-way majority | Three-way majority |
|---|---:|---:|---:|---:|
| vitaminc / test | 55,197 | 7,268 (13.2%) | 0.577 | 0.501 |
| fever / test | 16,039 | 6,456 (40.3%) | 0.510 | 0.403 |

Reporting two-way accuracy raises the floor and deletes the only class where the right
answer is *the evidence does not say*. On FEVER that is 40% of the test set. It is the class
a deployed system needs most and the one benchmarks shed first — which is what this
repository is named after.

## Layout

```
demo.py                     the bound on hand-built rows, then the real test splits
scripts/fetch_data.py       both corpora, byte-ranged, resumable, length-checked
src/doubt/corpus.py         VitaminC and FEVER loaded into one shape
src/doubt/ceilings.py       the exact bound, and the group-shape analysis
src/doubt/cli.py            `doubt` on your own csv/jsonl/parquet, text or JSON
scripts/measure.py          every table above
tests/                      44 tests; the bound is checked on hand-built cases
```

## Licence

Code: MIT, see [LICENSE](LICENSE). No dataset is committed; the fetch script downloads
each source under its own terms.
