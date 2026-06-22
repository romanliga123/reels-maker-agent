import pytest

from reels_agent.pipeline.probe import probe_video, ProbeError
from reels_agent import config


def test_probe_reads_correct_dimensions_and_audio(synth_video):
    result = probe_video(synth_video)
    assert result.width == 1280
    assert result.height == 720
    assert result.has_audio is True
    assert 4.5 < result.duration < 5.5


def test_probe_detects_missing_audio(silent_video):
    result = probe_video(silent_video)
    assert result.has_audio is False


def test_probe_rejects_corrupt_file(corrupt_video):
    with pytest.raises(ProbeError):
        probe_video(corrupt_video)


def test_probe_rejects_missing_file(tmp_path):
    with pytest.raises(ProbeError):
        probe_video(tmp_path / "does_not_exist.mp4")


def test_probe_rejects_too_long_video(synth_video, monkeypatch):
    monkeypatch.setattr(config, "MAX_VIDEO_DURATION_SEC", 1.0)  # 5с видео > 1с лимита
    with pytest.raises(ProbeError):
        probe_video(synth_video)
