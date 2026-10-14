import pytest
from pydantic import ValidationError
from ufor.envelfo import Envelope
from ufor.segments import Segment

from enge.synth import PreparedVoiceEnvelope


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
        {"segments": [{"duration": None}]},
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
    assert voice.release_frames == 16000
