"""Measure the callback-shaped heterogeneous engine path without device I/O."""

from statistics import median
from time import perf_counter_ns

import numpy as np
import tyro
from pydantic import BaseModel, ConfigDict, Field
from ufor import audio_effects
from ufor.events import Trigger
from ufor.synth import SynthInstrumentScore
from ufor.synth_trace import prepare as prepare_trace

from enge import effects, live, synth


class Settings(BaseModel):
    block_frames: int = Field(default=64, gt=0)
    seconds: int = Field(default=10, gt=0)
    voices: int = Field(default=16, gt=0)

    model_config = ConfigDict(frozen=True)


def main(settings: Settings) -> None:
    document = _score()
    events = [
        Trigger(
            tick=0,
            ordinal=i,
            part="main",
            trigger_id=f"note-{i}",
            key=48 + i,
            pitch_hz=110 * 2 ** (i / 12),
        )
        for i in range(settings.voices)
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main",
            channels=["left", "right"],
            source=audio_effects.MainStream(),
        ),
        [audio_effects.Gain(name="trim", gain_db=-6)],
        settings.block_frames,
    )
    engine = live.LiveEngine(
        {"synth": synth.PersistentSynth(synth.prepare(document), settings.voices)},
        effects.OfflineEffects(effects.prepare(graph, 48000), "native"),
    )
    output = np.empty((settings.block_frames, 2))
    durations = []
    callbacks = settings.seconds * 48000 // settings.block_frames
    for i in range(callbacks):
        start = i * settings.block_frames
        before = perf_counter_ns()
        engine.advance_into(
            {"synth": actions if i == 0 else []},
            [],
            start,
            start + settings.block_frames,
            output,
        )
        durations.append((perf_counter_ns() - before) / 1000)
    ordered = sorted(durations[1:])
    budget = settings.block_frames / 48000 * 1e6
    p99 = ordered[int(0.99 * (len(ordered) - 1))]
    print(
        f"callbacks={callbacks} block={settings.block_frames} voices={settings.voices}"
    )
    print(f"median_us={median(ordered):.1f} p99_us={p99:.1f} max_us={max(ordered):.1f}")
    print(f"budget_us={budget:.1f} max_fraction={max(ordered) / budget:.3f}")


def _score() -> SynthInstrumentScore:
    return SynthInstrumentScore.model_validate(
        {
            "name": "benchmark",
            "title": "Benchmark",
            "kind": "synth_instrument",
            "timebases": [
                {"name": "output", "kind": "physical", "rate": {"numerator": 48000}}
            ],
            "body": {
                "voices": [
                    {
                        "name": "tone",
                        "mapping": {
                            "lowest_key": 0,
                            "highest_key": 127,
                            "reference_pitch_hz": 440,
                        },
                        "channels": [
                            {"input": "mono", "output": "left", "gain": 0.05},
                            {"input": "mono", "output": "right", "gain": 0.05},
                        ],
                        "oscillator": {"waveform": "triangle"},
                    }
                ]
            },
            "inputs": [
                {
                    "name": "performance",
                    "stream": {
                        "family": "event",
                        "timebase": "output",
                        "kinds": ["trigger", "release", "control_change"],
                    },
                    "binding": {"performance": True},
                }
            ],
            "outputs": [
                {
                    "name": "audio",
                    "stream": {
                        "family": "sampled",
                        "quantity": "audio_amplitude",
                        "unit": "full_scale",
                        "timebase": "output",
                        "channels": ["left", "right"],
                    },
                    "binding": {"audio": True},
                }
            ],
        }
    )


if __name__ == "__main__":
    main(tyro.cli(Settings))
