"""Pull VitaminC and FEVER from their HuggingFace parquet conversions.

    python scripts/fetch_data.py

Both are needed: the point of the repository is the comparison between a
benchmark built to forbid claim-only shortcuts and one that was not.

Fetched in byte ranges and checked against Content-Length, because a single GET
of a 40 MB file over this link truncates often enough to matter, and a short
parquet fails later and less clearly than it should.

Each file is written to ``<name>.parquet.part`` and renamed only once its size
matches Content-Length, so an interrupted run never leaves a short parquet
behind; re-running resumes the ``.part`` from where it stopped.

Set ``DOUBT_DATA`` to download somewhere other than ``<repo>/data``.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(os.environ.get("DOUBT_DATA") or Path(__file__).resolve().parents[1] / "data")
SPLITS = ("train", "validation", "test")
CHUNK = 4_000_000

SOURCES = {
    "vitaminc": (
        "https://huggingface.co/datasets/tals/vitaminc/"
        "resolve/refs%2Fconvert%2Fparquet/default",
        DATA,
    ),
    "fever": (
        "https://huggingface.co/datasets/copenlu/fever_gold_evidence/"
        "resolve/refs%2Fconvert%2Fparquet/default",
        DATA / "fever",
    ),
}

# Published split sizes. A different count is a different benchmark.
EXPECT = {
    "vitaminc": {"train": 370_653, "validation": 63_054, "test": 55_197},
    "fever": {"train": 228_277, "validation": 15_935, "test": 16_039},
}


def expected_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "doubt"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def fetch(url: str, out: Path) -> None:
    total = expected_size(url)
    if out.exists() and out.stat().st_size == total:
        print(f"  {out.name} already complete ({total / 1e6:.0f} MB)")
        return

    part = out.with_name(out.name + ".part")
    written = part.stat().st_size if part.exists() else 0
    if written > total:
        part.unlink()
        written = 0
    resumed = f", resuming at {written / 1e6:.0f} MB" if written else ""
    print(f"  {out.name}  {total / 1e6:.0f} MB{resumed} ", end="", flush=True)
    with part.open("ab") as handle:
        while written < total:
            end = min(written + CHUNK, total) - 1
            request = urllib.request.Request(
                url, headers={"User-Agent": "doubt", "Range": f"bytes={written}-{end}"}
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    if response.status != 206 and written:
                        raise OSError(f"{out.name}: server ignored the byte range")
                    block = response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"\n    failed at byte {written:,}: {exc}")
                print("    re-run scripts/fetch_data.py to resume from here")
                raise
            if not block:
                raise OSError(f"{out.name}: empty response at byte {written:,}")
            handle.write(block)
            written += len(block)
            print(".", end="", flush=True)

    got = part.stat().st_size
    if got != total:
        part.unlink()
        raise OSError(f"{out.name}: got {got:,} bytes, expected {total:,}. Removed.")
    os.replace(part, out)
    print(" ok")


def main() -> None:
    for name, (base, folder) in SOURCES.items():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"{name}")
        for split in SPLITS:
            fetch(f"{base}/{split}/0000.parquet", folder / f"{split}.parquet")

    import pyarrow.parquet as pq

    print("\nchecking")
    ok = True
    for name, (_, folder) in SOURCES.items():
        for split in SPLITS:
            rows = pq.read_table(folder / f"{split}.parquet").num_rows
            want = EXPECT[name][split]
            good = rows == want
            ok &= good
            print(f"  {name:<10}{split:<11}{rows:>8,}  {'ok' if good else f'EXPECTED {want:,}'}")

    if not ok:
        print("\nThe splits are not the published ones; stop rather than measure "
              "something else.", file=sys.stderr)
        raise SystemExit(1)
    print("\nboth corpora ready")


if __name__ == "__main__":
    main()
