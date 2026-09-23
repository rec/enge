import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_synth import check_audio
from ufor.events import ControlChange, Release, Trigger

from enge.midi import MidiPerformance
from enge.presets import Patch
from enge.render import render_midi, write_flac


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        [
            "flac",
            "--silent",
            "--decode",
            "--stdout",
            "--force-raw-format",
            "--endian=little",
            "--sign=signed",
            str(path),
        ],
        check=True,
        capture_output=True,
    ).stdout
    octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    samples = octets[:, 0] | octets[:, 1] << 8 | octets[:, 2] << 16
    samples = (samples ^ (1 << 23)) - (1 << 23)
    return samples.reshape(-1, 2).astype(np.float64) / (2**23 - 1)


def test_mixed_engine_render_preserves_notes_tails_and_block_boundaries(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
) -> None:
    performance = MidiPerformance(
        sample_rate=48000,
        frames=48000,
        parts={
            c: [
                Trigger(
                    tick=c * 1200,
                    ordinal=0,
                    part=f"channel-{c}",
                    trigger_id="note",
                    key=60 + c * 4,
                    pitch_hz=220 * 2 ** (c / 3),
                    controls={"velocity": 0.8},
                ),
                ControlChange(
                    tick=18000,
                    ordinal=1,
                    control="expression",
                    value=0.8,
                    scope="part",
                    part=f"channel-{c}",
                ),
                Release(tick=40000, ordinal=2, part=f"channel-{c}", trigger_id="note"),
            ]
            for c in range(4)
        },
    )
    patches = {
        c: Patch(engine=e, pan=(c - 1.5) / 3)
        for c, e in enumerate(("fm", "synth", "sample", "noise"))
    }
    full = tmp_path / "full.flac"
    partitioned = tmp_path / "partitioned.flac"
    report = render_midi(performance, patches, full, block_size=60000, backend=backend)
    other = render_midi(
        performance, patches, partitioned, block_size=997, backend=backend
    )
    actual, expected = decode(partitioned), decode(full)
    check_audio(tmp_path / "mixed-engines.wav", actual, expected)
    assert report.frames == other.frames == 60000
    assert report.notes == 4
    assert report.peak == pytest.approx(other.peak)
    assert 0 < report.peak < 1
    assert np.all(actual[44000:] == 0)


def test_flac_round_trips_pcm_and_preserves_destination_on_overload(
    tmp_path: Path,
) -> None:
    t = np.arange(48000) / 48000
    samples = np.column_stack(
        [0.2 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 660 * t)]
    )
    path = tmp_path / "audio.flac"
    assert write_flac([samples], path) == pytest.approx(0.3)
    expected = np.rint(samples * (2**23 - 1)) / (2**23 - 1)
    check_audio(tmp_path / "pcm.wav", decode(path), expected)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="headroom"):
        write_flac([samples, np.full((48000, 2), 1.1)], path)
    assert path.read_bytes() == original
