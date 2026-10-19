"""Explicit, supervised CoreMIDI checks using only test-owned virtual endpoints."""

from contextlib import ExitStack
from os import environ, killpg
from pathlib import Path
from signal import SIGKILL
from subprocess import PIPE, Popen, TimeoutExpired
from sys import executable, platform
from time import monotonic, sleep
from uuid import uuid4

import pytest
from meedee import midi
from ufor import performance_binding
from ufor.events import ControlChange, PerformanceEvent, Release, Trigger

from enge.midi import LiveInput
from enge.performance import MidiConverter
from enge.presets import Patch


@pytest.mark.skipif(platform != "darwin", reason="CoreMIDI virtual-only qualification")
@pytest.mark.parametrize("isolated", [False, True], ids=["in_process", "isolated"])
def test_live_input(isolated: bool, request: pytest.FixtureRequest) -> None:
    if environ.get("ENGE_LIVE_INPUT_CHILD") == "1":
        exercise_input(isolated)
        return
    root = Path(__file__).resolve().parents[2]
    child = Popen(
        [executable, "-m", "pytest", "-q", "-s", "-n", "0", request.node.nodeid],
        cwd=root,
        env={**environ, "ENGE_LIVE_INPUT_CHILD": "1"},
        stdout=PIPE,
        stderr=PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(timeout=45)
    except TimeoutExpired:
        killpg(child.pid, SIGKILL)
        stdout, stderr = child.communicate(timeout=5)
        pytest.fail(f"native flow exceeded 45 seconds\n{stdout}\n{stderr}")
    print(stdout, end="")
    assert child.returncode == 0, stderr + stdout


def exercise_input(isolated: bool) -> None:
    score, _ = Patch(engine="synth").prepare_score(48000)
    binding = performance_binding.PerformanceBindingScore.model_validate(
        {
            "name": "native-input",
            "title": "Virtual MIDI qualification",
            "body": {
                "instrument": {"selector": "test:instrument"},
                "midi": [
                    {
                        "channels": [1],
                        "part": "main",
                        "repeated_key_release": "oldest",
                        "controllers": [{"number": 11, "control": "expression"}],
                        "pitch_bend": {"control": "bend"},
                    }
                ],
            },
        }
    )
    prefix = f"enge-live-{uuid4().hex}"
    with ExitStack() as stack:
        source = stack.enter_context(midi.Output(isolated=False).create_virtual(prefix))
        client = midi.TypedInput(isolated=isolated)
        port = next(p for p in client.ports() if p.name == prefix)
        connection = stack.enter_context(client.connect(port))
        converter = MidiConverter(source="keyboard", binding=binding, instrument=score)
        adapter = LiveInput(connection, converters=[converter], message_limit=2)
        for packet in [
            b"\x90\x45\x50",
            b"\xb0\x0b\x64",
            b"\xe0\x7f\x7f",
            b"\x80\x45\x00",
        ]:
            source.send(packet)
        received: list[PerformanceEvent] = []
        deadline = monotonic() + 5
        while len(received) < 4 and monotonic() < deadline:
            received.extend(adapter.drain(frame=480)["main"])
            sleep(0.01)
        assert len(received) == 4
        start, expression, bend, end = received
        assert isinstance(start, Trigger) and start.velocity == 80 / 127
        assert isinstance(expression, ControlChange) and expression.value == 100 / 127
        assert isinstance(bend, ControlChange) and bend.value == 1
        assert isinstance(end, Release) and end.trigger_id == start.trigger_id
        assert [e.tick for e in received] == [480] * 4
        assert [e.ordinal for e in received] == list(range(4))
        adapter.stop()
        assert not connection.closed

        # A too-small raw receive budget drops the final packet. No later MIDI
        # packet should be needed for the host to observe that loss.
        quiet_client = midi.TypedInput(isolated=isolated, receive_byte_limit=2)
        quiet_port = next(p for p in quiet_client.ports() if p.name == prefix)
        quiet = stack.enter_context(quiet_client.connect(quiet_port))
        quiet_converter = MidiConverter(
            source="quiet", binding=binding, instrument=score
        )
        quiet_adapter = LiveInput(quiet, converters=[quiet_converter])
        source.send(b"\x90\x45\x50")
        deadline = monotonic() + 5
        while quiet.dropped_packets == 0 and monotonic() < deadline:
            sleep(0.01)
        assert quiet.dropped_packets > 0
        with pytest.raises(RuntimeError, match="receive loss"):
            quiet_adapter.drain(frame=0)
        assert not quiet.closed
        assert not quiet_converter.ready
    print(f"Live conversion and silent loss passed: isolated={isolated}", flush=True)
