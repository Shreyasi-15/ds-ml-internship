import csv
import json

import pytest

from app.dataset import DatasetProtocolError, build_manifest


def test_build_manifest_and_summary(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "real_001.wav").write_bytes(b"fixture")
    (audio_dir / "fake_001.wav").write_bytes(b"fixture")
    protocol = tmp_path / "protocol.txt"
    protocol.write_text(
        "SPK_001 real_001 - - bonafide\n"
        "SPK_002 fake_001 - A01 spoof\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    summary = tmp_path / "summary.json"

    build_manifest(
        str(protocol),
        str(audio_dir),
        str(manifest),
        audio_ext=".wav",
        split="demo",
        summary_json=str(summary),
        source_name="test-fixture",
    )

    with manifest.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    report = json.loads(summary.read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert {row["label_name"] for row in rows} == {"bonafide", "spoof"}
    assert report["total"] == 2
    assert report["bonafide"] == 1
    assert report["spoof"] == 1
    assert report["missing_files"] == 0


def test_unknown_label_is_rejected(tmp_path):
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("SPK_001 clip_001 - - unknown\n", encoding="utf-8")

    with pytest.raises(DatasetProtocolError, match="unsupported label"):
        build_manifest(str(protocol), str(tmp_path), str(tmp_path / "manifest.csv"))
