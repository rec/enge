from fractions import Fraction

import pytest
from envelfo.runtime import envelfo_at
from pydantic import ValidationError
from ufor.envelfo import Envelope
from ufor.instrument_trace import EnvelfoObservation
from ufor.segments import IndefiniteSegment, Segment

from enge.synth import PreparedVoiceEnvelope, VoiceEnvelopeRenderer


@pytest.mark.parametrize(
    "changes",
    [
        {"release": []},
        {"playback": "loop"},
        {"hold": False},
        {"initial": "current"},
        {"start": "event"},
        {"scope": "instrument"},
        {"segments": [Segment(duration=1, to=1, curve=2)]},
    ],
)
def test_general_envelope_features_are_rejected_by_voice_preparation(
    changes: dict[str, object],
) -> None:
    raw = (
        Envelope(
            segments=[Segment(duration=1, to=1)],
            release=[Segment(duration=1, to=0)],
        ).model_dump()
        | changes
    )
    envelope = Envelope.model_validate(raw)
    with pytest.raises(ValidationError):
        PreparedVoiceEnvelope(sample_rate=48000, envelope=envelope)


def test_voice_retirement_duration_does_not_require_a_zero_terminal_value() -> None:
    envelope = Envelope(
        segments=[Segment(duration=1, to=1)],
        release=[Segment(duration="1/3 s", to=0.25)],
    )
    voice = PreparedVoiceEnvelope(sample_rate=48000, envelope=envelope)
    assert voice.envelope == envelope
    renderer = VoiceEnvelopeRenderer(definition=voice)
    renderer.release()
    assert renderer.active_frames(48000) == 16000


def test_minimum_hold_defers_release_and_release_waits_for_advance() -> None:
    body = Envelope(
        initial=1,
        segments=[IndefiniteSegment()],
        release=[IndefiniteSegment(), Segment(duration="1/8 s", to=0)],
    )
    renderer = VoiceEnvelopeRenderer(
        definition=PreparedVoiceEnvelope(
            sample_rate=48000, envelope=body, minimum_hold_seconds=Fraction(1, 4)
        )
    )
    assert renderer.release()
    assert not renderer.release()
    renderer.settle_amplitudes(Fraction(6000))
    assert (
        envelfo_at(body, renderer.amplitude_states[""], Fraction(1, 8)).status == "held"
    )
    renderer.frame_count = 24000
    assert not renderer.complete
    assert renderer.active_frames(48000) == 48000
    renderer.amplitude_event(
        EnvelfoObservation(
            tick=24000,
            ordinal=0,
            part="main",
            trigger_id="note",
            target="envelope",
            action="advance",
        )
    )
    assert renderer.active_frames(48000) == 6000
    renderer.frame_count = 30000
    assert renderer.complete


def test_paused_release_keeps_voice_alive_until_resumed() -> None:
    body = Envelope(
        initial=1,
        segments=[IndefiniteSegment()],
        release=[Segment(duration="1/4 s", to=0)],
    )
    renderer = VoiceEnvelopeRenderer(
        definition=PreparedVoiceEnvelope(sample_rate=48000, envelope=body)
    )
    renderer.release()
    renderer.frame_count = 6000
    renderer.amplitude_event(
        EnvelfoObservation(
            tick=6000,
            ordinal=0,
            part="main",
            trigger_id="note",
            target="envelope",
            action="pause",
        )
    )
    renderer.frame_count = 24000
    assert renderer.active_frames(48000) == 48000
    assert not renderer.complete
    renderer.amplitude_event(
        EnvelfoObservation(
            tick=24000,
            ordinal=0,
            part="main",
            trigger_id="note",
            target="envelope",
            action="resume",
        )
    )
    assert renderer.active_frames(48000) == 6000
