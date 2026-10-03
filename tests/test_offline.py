"""Everything that must work on a fresh clone with no corpora on disk."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from doubt import ceilings, cli, corpus

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def empty_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DOUBT_DATA", str(tmp_path))
    return tmp_path


def test_unknown_source_and_split_are_rejected():
    with pytest.raises(ValueError):
        corpus.load("nope", "test")
    with pytest.raises(ValueError):
        corpus.load(corpus.VITAMINC, "nope")


def test_data_dir_follows_env_var(empty_data):
    assert corpus.data_dir() == empty_data
    assert corpus.available() == {corpus.VITAMINC: [], corpus.FEVER: []}
    with pytest.raises(corpus.CorpusMissingError, match="fetch_data.py"):
        corpus.load(corpus.VITAMINC, "test")


def test_a_part_file_is_not_reported_as_available(empty_data):
    (empty_data / "test.parquet.part").write_bytes(b"short")
    assert corpus.available()[corpus.VITAMINC] == []


@pytest.mark.parametrize("fn", [ceilings.majority, ceilings.label_mix, ceilings.leak,
                                lambda rows: ceilings.ceiling(rows, ceilings.by_claim)])
def test_empty_rows_raise_a_clear_error(fn):
    with pytest.raises(ValueError, match="no rows"):
        fn([])


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_cli_on_a_csv(tmp_path, capsys):
    f = _write(tmp_path / "c.csv", "claim,evidence,label\n"
               "a,e1,SUPPORTS\na,e2,REFUTES\nb,e3,REFUTES\nb,e4,REFUTES\n")
    assert cli.main([str(f), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rows"] == 4
    assert out["claim_only_ceiling"] == 0.75
    assert out["majority"] == 0.75
    assert out["leak"] == 0.0


def test_cli_on_jsonl_with_custom_columns_and_unicode(tmp_path, capsys):
    f = tmp_path / "c.jsonl"
    f.write_text('{"q": "Zürich is in Switzerland", "y": "yes"}\n\n'
                 '{"q": "Zürich is in Switzerland", "y": "yes"}\n'
                 '{"q": "東京 is in Japan", "y": "no"}\n', encoding="utf-8")
    assert cli.main([str(f), "--claim-col", "q", "--label-col", "y"]) == 0
    assert "claim-only ceiling           1.000" in capsys.readouterr().out


def test_cli_rejects_missing_labels_and_bad_files(tmp_path, capsys):
    f = _write(tmp_path / "c.csv", "claim,evidence,label\na,e,\n")
    assert cli.main([str(f)]) == 2
    assert "record 1 has no 'label'" in capsys.readouterr().err
    assert cli.main([str(tmp_path / "missing.csv")]) == 2
    assert cli.main([str(_write(tmp_path / "c.txt", "x"))]) == 2
    assert cli.main([str(_write(tmp_path / "e.csv", "claim,label\n"))]) == 2
    assert cli.main([str(_write(tmp_path / "b.jsonl", "{oops\n"))]) == 2
    assert "b.jsonl:1" in capsys.readouterr().err


def test_cli_corpus_without_data_exits_cleanly(empty_data, capsys):
    assert cli.main(["--corpus", "vitaminc"]) == 2
    assert "fetch_data.py" in capsys.readouterr().err


def test_cli_needs_exactly_one_source(tmp_path):
    with pytest.raises(SystemExit):
        cli.main([])


def _run(script: str, env_dir: Path) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, "DOUBT_DATA": str(env_dir), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(ROOT / script)], capture_output=True,
                          text=True, encoding="utf-8", env=env, cwd=env_dir, timeout=120)


@pytest.mark.parametrize("script", ["demo.py", "scripts/measure.py"])
def test_scripts_run_without_data(script, tmp_path):
    done = _run(script, tmp_path)
    assert done.returncode == 0, done.stderr
    assert "fetch_data.py" in done.stdout
    assert list(tmp_path.iterdir()) == []


def test_demo_shows_the_contrast_on_hand_built_rows(tmp_path):
    out = _run("demo.py", tmp_path).stdout
    assert "claim-only ceiling 0.500   leak +0.000" in out
    assert "claim-only ceiling 1.000   leak +0.500" in out


# --- fetch_data: atomic and resumable ------------------------------------------------


def _load_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("DOUBT_DATA", str(tmp_path))
    spec = importlib.util.spec_from_file_location("fetch_data", ROOT / "scripts" / "fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Response(io.BytesIO):
    status = 206

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_resumes_a_part_file_and_renames_on_success(tmp_path, monkeypatch):
    fetch = _load_fetch(tmp_path, monkeypatch)
    assert fetch.DATA == tmp_path
    payload = bytes(range(256)) * 40
    monkeypatch.setattr(fetch, "CHUNK", 1000)
    monkeypatch.setattr(fetch, "expected_size", lambda url: len(payload))
    starts = []

    def urlopen(request, timeout):
        start, end = map(int, request.get_header("Range").split("=")[1].split("-"))
        starts.append(start)
        return _Response(payload[start:end + 1])

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    out = tmp_path / "test.parquet"
    (tmp_path / "test.parquet.part").write_bytes(payload[:2500])
    fetch.fetch("https://example.invalid/f", out)
    assert starts[0] == 2500
    assert out.read_bytes() == payload
    assert not (tmp_path / "test.parquet.part").exists()


def test_fetch_failure_leaves_only_a_part_file(tmp_path, monkeypatch):
    fetch = _load_fetch(tmp_path, monkeypatch)
    monkeypatch.setattr(fetch, "CHUNK", 10)
    monkeypatch.setattr(fetch, "expected_size", lambda url: 100)
    calls = []

    def urlopen(request, timeout):
        calls.append(1)
        if len(calls) > 3:
            raise TimeoutError("stalled")
        return _Response(b"x" * 10)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    out = tmp_path / "test.parquet"
    with pytest.raises(TimeoutError):
        fetch.fetch("https://example.invalid/f", out)
    assert not out.exists()
    assert (tmp_path / "test.parquet.part").stat().st_size == 30
