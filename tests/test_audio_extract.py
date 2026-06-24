import subprocess

import pytest

from reels_agent.pipeline.audio_extract import extract_audio, AudioExtractError
from reels_agent import config


def test_extract_produces_16khz_mono_wav(synth_video, tmp_path):
    out_wav = tmp_path / "audio.wav"
    result = extract_audio(synth_video, out_wav)
    assert result == out_wav
    assert out_wav.exists()

    probe = subprocess.run(
        [config.FFPROBE_BIN, "-v", "error", "-show_entries",
         "stream=sample_rate,channels", "-of", "default=noprint_wrappers=1", str(out_wav)],
        capture_output=True, text=True, timeout=30,
    )
    assert "sample_rate=16000" in probe.stdout
    assert "channels=1" in probe.stdout


def test_extract_creates_parent_dirs(synth_video, tmp_path):
    out_wav = tmp_path / "nested" / "deep" / "audio.wav"
    extract_audio(synth_video, out_wav)
    assert out_wav.exists()


def test_extract_raises_on_missing_input(tmp_path):
    with pytest.raises(AudioExtractError):
        extract_audio(tmp_path / "missing.mp4", tmp_path / "out.wav")


def test_extract_reports_progress_up_to_full_duration(synth_video, tmp_path):
    fractions = []
    extract_audio(
        synth_video, tmp_path / "audio.wav",
        total_duration_sec=5.0, on_progress=fractions.append,
    )
    assert fractions  # ffmpeg -progress отдал хотя бы один отсчёт за 5с
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert fractions[-1] >= 0.9  # последний отсчёт должен быть близко к концу
