"""Every number in the README.

    python scripts/measure.py

Nothing is trained. Every figure is an exact property of the corpora, computed
by grouping rows and counting, so none of it depends on how good a particular
classifier happened to be.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doubt import ceilings, corpus  # noqa: E402


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def the_corpora() -> None:
    rule("the corpora")
    have = corpus.available()
    print(f"{'corpus':<12}{'split':<12}{'rows':>9}{'SUPPORTS':>11}{'REFUTES':>10}{'NEI':>9}")
    for source in (corpus.VITAMINC, corpus.FEVER):
        for split in have[source]:
            rows = corpus.load(source, split)
            mix = ceilings.label_mix(rows)
            print(f"{source:<12}{split:<12}{len(rows):>9,}"
                  f"{mix[corpus.SUPPORTS]:>11.1%}{mix[corpus.REFUTES]:>10.1%}"
                  f"{mix[corpus.NOT_ENOUGH_INFO]:>9.1%}")
        if not have[source]:
            print(f"{source:<12}not downloaded — run scripts/fetch_data.py")


def the_ceilings() -> None:
    rule("what a model that ignores the evidence can reach")
    have = corpus.available()
    print(f"{'corpus':<12}{'split':<12}{'rows':>9}{'majority':>10}"
          f"{'claim-only':>12}{'leak':>8}{'evidence-only':>15}")
    for source in (corpus.VITAMINC, corpus.FEVER):
        for split in have[source]:
            rows = corpus.load(source, split)
            claim = ceilings.ceiling(rows, ceilings.by_claim)
            evidence = ceilings.ceiling(rows, ceilings.by_evidence)
            floor = ceilings.majority(rows)
            print(f"{source:<12}{split:<12}{len(rows):>9,}{floor:>10.3f}"
                  f"{claim.score:>12.3f}{claim.score - floor:>+8.3f}{evidence.score:>15.3f}")

    print("\n  ^ 'claim-only' is not a model's score. It is the highest score any")
    print("    predictor could get while never looking at the evidence, computed")
    print("    by grouping rows on the claim and keeping the commonest label in")
    print("    each group. 'leak' is how far that is above always guessing the")
    print("    commonest label overall.")


def the_structure() -> None:
    rule("why VitaminC has no room to leak")
    rows = corpus.load(corpus.VITAMINC, "train")
    shapes = ceilings.group_shapes(rows)
    print(f"{'rows in group':>14}{'labels spanned':>16}{'groups':>10}")
    for (size, labels), count in shapes.most_common(6):
        print(f"{size:>14}{labels:>16}{count:>10,}")

    multi = sum(c for (_, labels), c in shapes.items() if labels > 1)
    print(f"\n  groups spanning more than one label: {multi:,} of "
          f"{sum(shapes.values()):,}  ({multi / sum(shapes.values()):.1%})")
    print("\n  ^ each group is one claim against successive revisions of the same")
    print("    Wikipedia page. Most hold four rows across two labels, so the same")
    print("    claim is both true and false depending on which revision you read.")
    print("    A claim-only predictor has to give them all one answer.")


def the_third_label() -> None:
    rule("the label everybody drops")
    for source in (corpus.VITAMINC, corpus.FEVER):
        splits = corpus.available()[source]
        if not splits:
            continue
        split = "test" if "test" in splits else splits[0]
        rows = corpus.load(source, split)
        nei = [r for r in rows if r.uncertain]
        two_way = [r for r in rows if not r.uncertain]
        print(f"\n{source} / {split}")
        print(f"  rows                       {len(rows):>8,}")
        print(f"  NOT ENOUGH INFO            {len(nei):>8,}   {len(nei) / len(rows):.1%}")
        print(f"  dropping it leaves         {len(two_way):>8,}")
        if two_way:
            floor2 = ceilings.majority(two_way)
            print(f"  two-way majority baseline  {floor2:>8.3f}"
                  f"   (vs {ceilings.majority(rows):.3f} three-way)")
    print("\n  ^ reporting two-way accuracy raises the floor and removes the only")
    print("    class where the right answer is 'the evidence does not say'. That")
    print("    is the class a system needs most and the one benchmarks shed first.")


def main() -> None:
    the_corpora()
    the_ceilings()
    the_structure()
    the_third_label()
    print()


if __name__ == "__main__":
    main()
