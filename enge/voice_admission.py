"""Use Rust uFor to plan retirements during instrument trace preparation."""

from pydantic import TypeAdapter
from ufor.instrument_trace import VoiceRetirement
from ufor.voice_admission import VoiceAdmission

from . import _native


def native_retirements(batch: VoiceAdmission) -> list[VoiceRetirement]:
    return _RETIREMENTS.validate_json(
        _native.plan_voice_retirements(batch.model_dump_json())
    )


_RETIREMENTS = TypeAdapter(list[VoiceRetirement])
