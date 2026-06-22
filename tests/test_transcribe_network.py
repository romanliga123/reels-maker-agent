import pytest

from reels_agent.pipeline.audio_extract import extract_audio
from reels_agent.pipeline.transcribe import transcribe_wav
from tests.conftest import has_groq_key

pytestmark = pytest.mark.network
requires_groq = pytest.mark.skipif(not has_groq_key(), reason="требуется реальный GROQ_API_KEY")


@requires_groq
def test_transcribe_wav_real_api_call_succeeds(synth_video, tmp_path):
    """Не проверяем содержимое (synth_video — тон без речи), только то, что реальный
    вызов Groq API проходит без ошибок аутентификации/сети и возвращает список."""
    wav = tmp_path / "audio.wav"
    extract_audio(synth_video, wav)
    segments = transcribe_wav(wav, language="ru")
    assert isinstance(segments, list)
