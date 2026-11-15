from pathlib import Path
from typing import Literal

import numpy as np
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects
from ufor.samples.processing import FilterResponse, ResonantFilter

from enge import effects


def test_cached_coefficients_follow_controls_and_restore_across_parameter_changes(
    tmp_path: Path,
) -> None:
    graph = audio_effects.EffectGraph(
        scope=audio_effects.AttachmentScope.master,
        inputs=[graph_input()],
        processors=[
            audio_effects.Filter(
                name="band",
                filters=[
                    ResonantFilter(
                        name="tone",
                        response=FilterResponse.bandpass,
                        cutoff_hz=6000,
                        q=0.7,
                    )
                ],
            ),
            audio_effects.Compressor(name="compressor"),
        ],
        connections=[
            audio_effects.Connection(
                processor="band",
                port="input",
                source=audio_effects.InputSource(input="main"),
            ),
            *[
                audio_effects.Connection(
                    processor="compressor",
                    port=p,
                    source=audio_effects.ProcessorSource(processor="band"),
                )
                for p in ("input", "detector")
            ],
        ],
        output=audio_effects.ProcessorSource(processor="compressor"),
        maximum_block_frames=48000,
    )
    prepared = effects.prepare(graph, 48000)
    time = np.arange(48000) / 48000
    source = np.column_stack(
        [
            0.2 * np.sin(2 * np.pi * 997 * time)
            + 0.3 * np.sin(2 * np.pi * 6000 * time),
            0.1 * np.cos(2 * np.pi * 330 * time)
            + 0.4 * np.cos(2 * np.pi * 7001 * time),
        ]
    )
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=t, ordinal=o, processor=p, parameter=n, value=v, duration_frames=d
        )
        for t, o, p, n, v, d in [
            (4001, 0, "band", "tone-cutoff_hz", 1000, 0),
            (8003, 0, "band", "tone-q", 0.1, 0),
            (12007, 0, "compressor", "attack_seconds", 0, 0),
            (16011, 0, "compressor", "release_seconds", 0.02, 0),
            (20003, 0, "band", "tone-cutoff_hz", 6500, 2000),
            (24011, 0, "band", "tone-q", 1.4, 1000),
            (28013, 0, "compressor", "attack_seconds", 0.005, 500),
            (32017, 0, "compressor", "release_seconds", 0, 0),
            (36019, 0, "band", "tone-cutoff_hz", 6000, 0),
            (36019, 1, "band", "tone-q", 0.7, 0),
            (40021, 0, "compressor", "release_seconds", 0.1, 0),
            (40021, 1, "compressor", "attack_seconds", 0.01, 0),
        ]
    ]
    actions.append(audio_effects.StopAction(tick=45007, ordinal=0))
    expected = effects.process_audio(prepared, {"main": source}, actions, "numpy", 997)
    pairs: list[tuple[Literal["numpy", "native"], Literal["numpy", "native"]]] = [
        ("numpy", "native"),
        ("native", "numpy"),
    ]
    for origin, target in pairs:
        first = effects.OfflineEffects(prepared, origin)
        head = first.advance(
            {"main": source[:23928]}, [a for a in actions if a.tick < 23928], 0, 23928
        )
        saved = effects.EffectSnapshot.model_validate_json(
            first.snapshot().model_dump_json()
        )
        second = effects.OfflineEffects(prepared, target)
        second.restore(saved)
        tail = second.advance(
            {"main": source[23928:]},
            [a for a in actions if a.tick >= 23928],
            23928,
            48000,
        )
        check_audio(
            tmp_path / f"{origin}-{target}.wav", np.vstack([head, tail]), expected
        )
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)
