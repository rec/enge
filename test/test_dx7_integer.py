from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from ufor.dx7 import DX7Voice

from enge.yamaha import DX7Renderer, DX7Snapshot


@pytest.mark.parametrize("case", range(176))
def test_dx7_integer_core_matches_pinned_mki_and_restores(
    tmp_path: Path, case: int
) -> None:
    with np.load(Path(__file__).parent / "data/dx7-mki-core.npz") as reference:
        algorithm, feedback, level = map(int, reference["parameters"][case])
        expected = reference[f"case{case}"].astype(float)[:, None] / 2**24
    raw = bytearray(155)
    for i in range(0, 126, 21):
        raw[i : i + 8] = bytes([99] * 8)
        raw[i + 16], raw[i + 18], raw[i + 20] = 99, 1, 7
    raw[126:134] = bytes([99] * 4 + [50] * 4)
    raw[134], raw[135], raw[136], raw[144] = algorithm - 1, feedback, 1, 24
    note = DX7Renderer(
        DX7Voice(data=bytes(raw)), 69, 100, pitch_hz=153791 * 48000 / 2**24
    )
    snapshot = note.snapshot()
    for envelope in snapshot.envelopes:
        envelope.level, envelope.stage, envelope.held = level * 2**24, 3, True
    note.restore(snapshot)
    first = note.render(17003)
    saved = DX7Snapshot.model_validate_json(note.snapshot().model_dump_json())
    tail = note.render(30997)
    actual = np.concatenate([first, tail])
    check_audio(tmp_path / "mki-core.wav", actual, expected)
    np.testing.assert_array_equal(actual, expected)
    note.restore(saved)
    np.testing.assert_array_equal(note.render(30997), tail)


@pytest.mark.parametrize("case", range(9))
def test_dynamic_core_matches_reference_with_captured_envelope_trajectories(
    tmp_path: Path, case: int
) -> None:
    # Upstream integer kernels receive enge's captured attenuation rows and
    # per-frame silence policy. This does not qualify envelope timing itself.
    with np.load(Path(__file__).parent / "data/dx7-mki-dynamic.npz") as reference:
        patch = DX7Voice(data=reference[f"patch{case}"].tobytes())
        expected = reference[f"case{case}"].astype(float)[:, None] / 2**24
    note = DX7Renderer(patch, 69, 100, pitch_hz=153791 * 48000 / 2**24)
    head = note.render(17003)
    saved = DX7Snapshot.model_validate_json(note.snapshot().model_dump_json())
    middle = note.render(6997)
    note.release()
    tail = note.render(24000)
    actual = np.concatenate([head, middle, tail])
    check_audio(tmp_path / "mki-dynamic.wav", actual, expected)
    np.testing.assert_array_equal(actual, expected)
    note.restore(saved)
    np.testing.assert_array_equal(note.render(6997), middle)
    note.release()
    np.testing.assert_array_equal(note.render(24000), tail)
