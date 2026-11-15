from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_effects import graph_input
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


@pytest.mark.parametrize("container", ["list", "tuple"])
def test_native_call_keeps_borrowed_host_sequences_and_audio_layout(
    tmp_path: Path,
    container: Literal["list", "tuple"],
) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [audio_effects.Gain(name="level", gain_db=-6)],
        48000,
    )
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(effects.prepare(graph, 48000)),
        2,
        input_routes=[[-2]],
    )
    time = np.arange(48000) / 48000
    source = np.column_stack([0.2 * np.sin(2 * np.pi * f * time) for f in (997, 7001)])
    actual = np.empty_like(source)
    inputs = [source] if container == "list" else (source,)
    runtime.process_into(actual, inputs)
    check_audio(tmp_path / "output.wav", actual, source * 10 ** (-6 / 20))
