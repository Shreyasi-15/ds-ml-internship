"""Generate a tiny redistributable audio set for pipeline smoke tests only."""

import argparse
import math
import struct
import wave
from pathlib import Path


SAMPLE_RATE = 16_000


def _samples(frequency: float, reverb: bool, duration: float = 1.5) -> list[float]:
    count = int(SAMPLE_RATE * duration)
    dry = []
    for index in range(count):
        time = index / SAMPLE_RATE
        gate = 1.0 if (time % 0.5) < 0.18 else 0.0
        dry.append(0.28 * math.sin(2 * math.pi * frequency * time) * gate)

    if not reverb:
        return dry
    wet = dry.copy()
    for delay_seconds, gain in ((0.05, 0.45), (0.10, 0.25)):
        delay = int(delay_seconds * SAMPLE_RATE)
        for index in range(delay, count):
            wet[index] += gain * dry[index - delay]
    return wet


def _write_wav(path: Path, samples: list[float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"".join(
        struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767))
        for sample in samples
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset/demo/audio")
    args = parser.parse_args()
    output = Path(args.output)

    examples = (
        ("demo_bonafide_001.wav", 180.0, False),
        ("demo_bonafide_002.wav", 240.0, False),
        ("demo_spoof_001.wav", 180.0, True),
        ("demo_spoof_002.wav", 240.0, True),
    )
    for filename, frequency, reverb in examples:
        _write_wav(output / filename, _samples(frequency, reverb))
    print(f"Generated {len(examples)} smoke-test clips in {output}")


if __name__ == "__main__":
    main()
