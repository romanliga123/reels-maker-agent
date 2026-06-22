import subprocess

import pytest

from reels_agent.pipeline.audio_energy import detect_energy_spans
from reels_agent import config


@pytest.fixture(scope="module")
def burst_wav(tmp_path_factory):
    """Тихие тона с двумя громкими 'смехоподобными' всплесками — детерминированный сигнал
    для проверки, что detect_energy_spans находит именно эти всплески и игнорирует тишину."""
    out_dir = tmp_path_factory.mktemp("burst_fixture")
    parts = {
        "q1.wav": ["sine=frequency=220:duration=4", "volume=0.05"],
        "burst1.wav": ["anoisesrc=color=white:duration=2", "volume=0.8,tremolo=f=8:d=0.9"],
        "q2.wav": ["sine=frequency=220:duration=5", "volume=0.05"],
        "burst2.wav": ["anoisesrc=color=white:duration=2", "volume=0.8,tremolo=f=8:d=0.9"],
        "q3.wav": ["sine=frequency=220:duration=4", "volume=0.05"],
    }
    for name, (src, af) in parts.items():
        subprocess.run([
            config.FFMPEG_BIN, "-y", "-f", "lavfi", "-i", src,
            "-af", af, "-ar", "16000", "-ac", "1", str(out_dir / name),
        ], capture_output=True, check=True, timeout=30)

    list_file = out_dir / "list.txt"
    list_file.write_text("\n".join(f"file '{name}'" for name in parts), encoding="utf-8")
    combined = out_dir / "combined.wav"
    subprocess.run([
        config.FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(combined),
    ], capture_output=True, check=True, timeout=30)
    return combined


def test_detects_bursts_at_expected_timestamps(burst_wav):
    spans = detect_energy_spans(burst_wav)
    assert len(spans) >= 2

    # ожидаем всплески около [4-6с] и [11-13с] (после двух тихих блоков 4с и 5с)
    starts = sorted(s.start for s in spans)
    assert any(3.5 <= s <= 6.5 for s in starts)
    assert any(10.0 <= s <= 13.5 for s in starts)


def test_quiet_only_audio_finds_nothing(tmp_path):
    quiet = tmp_path / "quiet.wav"
    subprocess.run([
        config.FFMPEG_BIN, "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=5",
        "-af", "volume=0.05", "-ar", "16000", "-ac", "1", str(quiet),
    ], capture_output=True, check=True, timeout=30)
    spans = detect_energy_spans(quiet)
    assert spans == []


def test_spans_sorted_by_score_descending(burst_wav):
    spans = detect_energy_spans(burst_wav)
    scores = [s.score for s in spans]
    assert scores == sorted(scores, reverse=True)
