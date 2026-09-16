import shutil
import subprocess
import wave
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from reccy.runtime.files import atomic_output
from test_dynamic_synth import change, dynamic_score
from test_synth import check_audio
from ufor import synth_trace
from ufor.events import ControlChange, Release, Trigger
from ufor.synth import SynthInstrumentScore

from enge.synth import OfflineSynth, prepare


def test_synth_demo_preserves_envelopes_and_live_controls(
    tmp_path: Path, pytestconfig: pytest.Config, backend: Literal["numpy", "native"]
) -> None:
    """Render an arpeggio and a bent final note, retaining a listenable FLAC."""
    rate = 48000
    frames = 204000
    raw = dynamic_score(smoothing="1/20").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["channels"] = [
        {"input": "mono", "output": n, "gain": 1} for n in ("left", "right")
    ]
    voice["bindings"][1]["smoothing"] = "1/8"
    voice["envelope"] = {
        "initial": 0,
        "segments": [
            {"duration": "1/100", "target": 1},
            {"duration": "9/100", "target": 0.7},
        ],
        "release": [{"duration": "1/4", "target": 0}],
    }
    document = SynthInstrumentScore.model_validate(raw)
    events: list[Trigger | Release | ControlChange] = []
    expected = np.zeros((frames, 2))
    for i, key in enumerate((60, 64, 67, 72, 67, 64, 60)):
        start = i * 24000
        held = 36000 if i == 6 else 18000
        pitch = 440 * 2 ** ((key - 69) / 12)
        trigger_id = f"note-{i}"
        events.extend(
            [
                Trigger(
                    tick=start,
                    ordinal=0,
                    part="main",
                    trigger_id=trigger_id,
                    key=key,
                    pitch_hz=pitch,
                    controls={"gain": 0.22},
                ),
                Release(
                    tick=start + held,
                    ordinal=0,
                    part="main",
                    trigger_id=trigger_id,
                ),
            ]
        )
        age = np.arange(held + 12000)
        gain = np.full(len(age), 0.22)
        phase = age * pitch / rate
        if i == 6:
            events.extend(
                [
                    change(start + 6000, 0.35, trigger_id=trigger_id),
                    change(start + 12000, 1 / 6, "bend", trigger_id=trigger_id),
                    change(start + 24000, 0, "bend", trigger_id=trigger_id),
                ]
            )
            gain += 0.13 * np.clip((age - 6000) / 2400, 0, 1)
            # A linear cents ramp is a geometric frequency sequence. Sum it
            # analytically so the oracle does not accumulate rounding drift.
            step = np.log(2) / (6 * 6000)
            phase = (
                pitch
                / rate
                * (
                    np.minimum(age, 12000)
                    + np.expm1(step * np.clip(age - 12000, 0, 6000)) / np.expm1(step)
                    + 2 ** (1 / 6) * np.clip(age - 18000, 0, 6000)
                    + 2 ** (1 / 6)
                    * np.expm1(-step * np.clip(age - 24000, 0, 6000))
                    / np.expm1(-step)
                    + np.maximum(age - 30000, 0)
                )
            )
        envelope = np.where(
            age < 480,
            age / 480,
            1 - 0.3 * np.clip((age - 480) / 4320, 0, 1),
        )
        envelope *= 1 - np.clip((age - held) / 12000, 0, 1)
        note = np.sin(2 * np.pi * phase) * envelope * gain
        expected[start : start + len(age)] += note[:, None]

    events.sort(key=lambda e: (e.tick, e.ordinal))
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    renderer = OfflineSynth(prepare(document), backend)
    actual = renderer.advance(actions, 0, frames)
    assert renderer.snapshot().voices == []
    assert np.max(np.abs(actual)) < 1
    assert np.all(actual[192000:] == 0)
    check_audio(tmp_path / "synth-demo.wav", actual, expected)

    # Verify the lossless file contains exactly the PCM we compared above.
    source = tmp_path / "synth-demo-actual.wav"
    encoded = tmp_path / "synth-demo.flac"
    subprocess.run(
        ["flac", "--silent", "--output-name", str(encoded), str(source)],
        check=True,
    )
    decoded = subprocess.run(
        [
            "flac",
            "--silent",
            "--decode",
            "--stdout",
            "--force-raw-format",
            "--endian=little",
            "--sign=signed",
            str(encoded),
        ],
        check=True,
        capture_output=True,
    )
    with wave.open(str(source)) as audio:
        assert decoded.stdout == audio.readframes(frames)
    destination = pytestconfig.cache.mkdir("audio") / f"synth-demo-{backend}.flac"
    with atomic_output(destination) as temporary:
        shutil.copyfile(encoded, temporary)
    assert destination.read_bytes() == encoded.read_bytes()
