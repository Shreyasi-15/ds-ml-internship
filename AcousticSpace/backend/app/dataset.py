"""
Dataset Curation (Week 1 deliverable)
Loader/index builder for ASVspoof-style datasets (bonafide vs spoof).

ASVspoof protocol files are typically plain-text with rows like:
    <speaker_id> <filename> - <system_id> <label: bonafide|spoof>

This module builds a simple manifest (CSV) that downstream training code
and the audio_pipeline can consume: filepath, label.
"""

import csv
import os


def build_manifest(protocol_path: str, audio_dir: str, output_csv: str, audio_ext=".flac"):
    """
    Parses an ASVspoof protocol file and writes a manifest CSV of
    (filepath, label) rows, label in {0: bonafide, 1: spoof}.
    """
    rows = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            filename = parts[1]
            label_str = parts[-1]
            label = 0 if label_str == "bonafide" else 1
            filepath = os.path.join(audio_dir, filename + audio_ext)
            rows.append((filepath, label))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(rows)

    n_bonafide = sum(1 for _, l in rows if l == 0)
    n_spoof = sum(1 for _, l in rows if l == 1)
    print(f"Manifest written to {output_csv}: {n_bonafide} bonafide, {n_spoof} spoof")
    return output_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ASVspoof manifest CSV")
    parser.add_argument("--protocol", required=True, help="Path to ASVspoof protocol .txt file")
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files")
    parser.add_argument("--output", default="manifest.csv", help="Output CSV path")
    args = parser.parse_args()

    build_manifest(args.protocol, args.audio_dir, args.output)
