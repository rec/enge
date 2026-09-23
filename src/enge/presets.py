"""Small playable presets for offline MIDI examples and engine comparisons."""

from hashlib import sha256
from math import cos, pi, sin
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from ufor.samples.instrument import SampleInstrumentScore
from ufor.synth import SynthInstrumentScore


class Patch(BaseModel, frozen=True):
    engine: Literal["fm", "synth", "sample", "noise"]
    gain: float = Field(default=0.18, ge=0)
    pan: float = Field(default=0, ge=-1, le=1)
    ratio: float = Field(default=2, gt=0)
    index: float = Field(default=0.8, ge=0)

    def prepare_score(
        self, sample_rate: int
    ) -> tuple[SynthInstrumentScore | SampleInstrumentScore, dict[str, np.ndarray]]:
        """Build a stereo instrument with expression, velocity, bend and timbre."""
        routes = [
            {"input": "mono", "output": n, "gain": self.gain * g}
            for n, g in (
                ("left", cos((self.pan + 1) * pi / 4)),
                ("right", sin((self.pan + 1) * pi / 4)),
            )
        ]
        envelope = {
            "segments": [{"duration": "1/100", "target": 1}],
            "release": [{"duration": "2/25", "target": 0}],
        }
        settings = _settings(self.engine, self.index)
        voice = {
            **settings,
            "name": "voice",
            "mapping": {"lowest_key": 0, "highest_key": 127, "reference_pitch_hz": 220},
            "channels": routes,
        }
        controls = {
            "velocity": {"default": 1},
            "expression": {"default": 1},
            "modulation": {"default": 0.5},
            "bend": {"default": 0, "polarity": "bipolar"},
        }
        document: dict[str, object] = {
            "name": "midi",
            "title": "MIDI instrument",
            "timebases": [
                {
                    "name": "output",
                    "kind": "physical",
                    "rate": {"numerator": sample_rate},
                }
            ],
            "inputs": [
                {
                    "name": "performance",
                    "binding": {"performance": True},
                    "stream": {
                        "family": "event",
                        "timebase": "output",
                        "kinds": ["trigger", "release", "control_change"],
                    },
                }
            ],
            "outputs": [
                {
                    "name": "audio",
                    "binding": {"audio": True},
                    "stream": {
                        "family": "sampled",
                        "quantity": "audio_amplitude",
                        "unit": "full_scale",
                        "timebase": "output",
                        "channels": ["left", "right"],
                    },
                }
            ],
        }
        if self.engine == "fm":
            voice["fm"] = {
                "operators": [
                    {"name": "modulator", "ratio": self.ratio, "envelope": envelope},
                    {"name": "carrier", "envelope": envelope},
                ],
                "connection": {
                    "source": "modulator",
                    "destination": "carrier",
                    "index": self.index,
                },
            }
        elif self.engine == "synth":
            voice.update(oscillator={"waveform": "triangle"}, envelope=envelope)
        elif self.engine == "noise":
            voice.update(
                noise="white",
                envelope=envelope,
                mapping={"lowest_key": 0, "highest_key": 127, "pitch_tracking": False},
            )
        else:
            # One second contains exactly 220 periods; the wrap is continuous.
            # This deterministic asset exercises real sample interpolation/loops.
            phase = 2 * np.pi * 220 * np.arange(sample_rate) / sample_rate
            audio = (
                np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
            )[:, None] / 1.37
            voice.update(slice="tone", envelope=envelope)
            document.update(
                kind="instrument",
                assets=[
                    {
                        "name": "tone",
                        "location": {
                            "kind": "relative_file",
                            "path": "generated-harmonics.f64",
                        },
                        "encoding": "float64-le",
                        "content": {
                            "byte_length": audio.nbytes,
                            "sha256": sha256(audio.astype("<f8").tobytes()).hexdigest(),
                        },
                        "audio": {
                            "timebase": "output",
                            "channels": ["mono"],
                            "frames": len(audio),
                        },
                    }
                ],
                body={
                    "settings": {"controls": controls},
                    "slices": [
                        {
                            "name": "tone",
                            "asset": "tone",
                            "end_frame": len(audio),
                            "loop": {
                                "start_frame": 0,
                                "end_frame": len(audio),
                                "mode": "through_release",
                            },
                        }
                    ],
                    "slots": [voice],
                },
            )
            return SampleInstrumentScore.model_validate(document), {"tone": audio}
        document.update(
            kind="synth_instrument", body={"controls": controls, "voices": [voice]}
        )
        return SynthInstrumentScore.model_validate(document), {}


def _settings(engine: str, index: float) -> dict[str, object]:
    sources = [
        {
            "name": n,
            "scope": "trigger" if n == "velocity" else "part",
            "minimum": -1 if n == "bend" else 0,
            "maximum": 1,
        }
        for n in ("velocity", "expression", "bend", "modulation")
        if engine != "noise" or n != "bend"
    ]
    amplitude = {"name": "processing", "parameter": "amplitude"}
    tuning = {"name": "processing", "parameter": "tuning_cents"}
    color = (
        {"name": "fm", "parameter": "index"}
        if engine == "fm"
        else {"name": "filter-tone", "parameter": "cutoff_hz"}
    )
    unit, default, depth = (
        ("radians", index, 0.25) if engine == "fm" else ("hz", 4500, 1000)
    )
    parameters: list[dict[str, object]] = [
        {
            "target": amplitude,
            "unit": "ratio",
            "scope": "voice",
            "minimum": 0,
            "maximum": 1,
            "default": 1,
        },
        {
            "target": color,
            "unit": unit,
            "scope": "voice",
            "minimum": max(0, default - depth),
            "maximum": default + depth,
            "default": default,
        },
    ]
    if engine != "noise":
        parameters.insert(
            1,
            {
                "target": tuning,
                "unit": "cents",
                "scope": "voice",
                "minimum": -200,
                "maximum": 200,
                "default": 0,
            },
        )
    routes: list[dict[str, object]] = [
        {
            "name": n,
            "source": n,
            "target": amplitude,
            "operation": "multiply",
            "unit": "ratio",
            "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 1}],
        }
        for n in ("velocity", "expression")
    ]
    if engine != "noise":
        routes.append(
            {
                "name": "bend",
                "source": "bend",
                "target": tuning,
                "operation": "add",
                "unit": "cents",
                "points": [{"input": -1, "amount": -200}, {"input": 1, "amount": 200}],
            }
        )
    routes.append(
        {
            "name": "modulation",
            "source": "modulation",
            "target": color,
            "operation": "add",
            "unit": unit,
            "points": [
                {"input": 0, "amount": -min(default, depth)},
                {"input": 1, "amount": depth},
            ],
        }
    )
    return {
        "processing": {
            "filters": []
            if engine == "fm"
            else [{"name": "tone", "response": "lowpass", "cutoff_hz": 4500, "q": 0.7}]
        },
        "modulation": {"sources": sources, "parameters": parameters, "routes": routes},
        "bindings": [
            {
                "name": n,
                "kind": "control",
                "control": n,
                "smoothing": 0 if n == "velocity" else "1/10",
            }
            for n in ("velocity", "expression", "bend", "modulation")
            if engine != "noise" or n != "bend"
        ],
    }
