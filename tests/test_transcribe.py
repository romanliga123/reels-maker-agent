import subprocess

import pytest

from reels_agent.pipeline.transcribe import (
    _segment_index_for, _wav_duration, split_audio_chunks, transcribe_long_audio,
    transcribe_wav, TranscribeError,
)
from reels_agent.models import TranscriptSegment, Word
from reels_agent import config


class TestSegmentIndexFor:
    def test_finds_containing_segment(self):
        segs = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0}]
        assert _segment_index_for(segs, 5.0) == 1

    def test_returns_none_when_no_match(self):
        segs = [{"start": 0.0, "end": 4.0}]
        assert _segment_index_for(segs, 10.0) is None


@pytest.fixture(scope="module")
def long_wav(tmp_path_factory):
    """20с тоновый wav, режем на чанки по 5с — без сети, чисто проверка нарезки/слияния."""
    out_dir = tmp_path_factory.mktemp("long_wav_fixture")
    path = out_dir / "long.wav"
    subprocess.run([
        config.FFMPEG_BIN, "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=20",
        "-ar", "16000", "-ac", "1", str(path),
    ], capture_output=True, check=True, timeout=30)
    return path


class TestWavDuration:
    def test_reports_correct_duration(self, long_wav):
        assert 19.5 < _wav_duration(long_wav) < 20.5

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(TranscribeError):
            _wav_duration(tmp_path / "missing.wav")


class TestSplitAudioChunks:
    def test_produces_expected_chunk_count(self, long_wav, tmp_path):
        chunks = split_audio_chunks(long_wav, tmp_path, chunk_sec=5, overlap_sec=1)
        # 20с / 5с шагом без хвоста короче чанка => 4 чанка
        assert len(chunks) == 4
        for path, offset in chunks:
            assert path.exists()

    def test_chunk_offsets_increase_by_chunk_sec(self, long_wav, tmp_path):
        chunks = split_audio_chunks(long_wav, tmp_path, chunk_sec=5, overlap_sec=1)
        offsets = [offset for _, offset in chunks]
        assert offsets == [0.0, 5.0, 10.0, 15.0]

    def test_chunks_include_overlap_duration(self, long_wav, tmp_path):
        chunks = split_audio_chunks(long_wav, tmp_path, chunk_sec=5, overlap_sec=1)
        first_chunk_path, _ = chunks[0]
        dur = _wav_duration(first_chunk_path)
        assert 5.5 < dur < 6.5  # chunk_sec + overlap_sec


class TestTranscribeWav:
    def test_rejects_file_over_size_limit(self, long_wav, monkeypatch):
        monkeypatch.setattr(config, "WHISPER_CHUNK_LIMIT_BYTES", 10)  # заведомо меньше файла
        with pytest.raises(TranscribeError):
            transcribe_wav(long_wav)

    def test_raises_without_api_key(self, long_wav, monkeypatch):
        monkeypatch.setattr(config, "GROQ_API_KEY", "")
        with pytest.raises(TranscribeError):
            transcribe_wav(long_wav)


class TestTranscribeLongAudioMerge:
    def test_merges_chunks_with_offset_and_drops_overlap(self, long_wav, tmp_path, monkeypatch):
        """Подменяем transcribe_wav на детерминированный фейк (без сети) и проверяем,
        что итоговые сегменты сдвинуты на смещение чанка и не дублируются на стыке."""
        monkeypatch.setattr(config, "WHISPER_CHUNK_LIMIT_BYTES", 1)  # форсируем чанкинг
        monkeypatch.setattr(config, "WHISPER_CHUNK_TARGET_SEC", 5)  # 20с/5с => 4 чанка

        def fake_transcribe_wav(chunk_path, language="ru"):
            # каждый чанк "слышит" один сегмент длиной 3с, начинающийся в локальном времени 0
            # (т.е. внутри overlap-зоны для всех чанков кроме первого — должен быть отброшен)
            return [TranscriptSegment(
                text=f"chunk {chunk_path.name}", start=0.0, end=3.0,
                words=[Word("word", 0.0, 3.0)],
            )]

        monkeypatch.setattr(
            "reels_agent.pipeline.transcribe.transcribe_wav", fake_transcribe_wav
        )

        segments = transcribe_long_audio(long_wav, tmp_path, overlap_sec=1)
        # только чанк 0 проходит (его сегмент начинается до overlap_sec, но i==0 не фильтруется);
        # чанки 1..3 отбрасывают сегмент, т.к. seg.start(0.0) < overlap_sec(1.0)
        assert len(segments) == 1
        assert segments[0].start == 0.0  # сдвиг чанка 0 равен 0, без изменений

    def test_offsets_are_applied_to_words_and_segments(self, long_wav, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "WHISPER_CHUNK_LIMIT_BYTES", 1)
        monkeypatch.setattr(config, "WHISPER_CHUNK_TARGET_SEC", 5)

        def fake_transcribe_wav(chunk_path, language="ru"):
            # сегмент начинается ПОСЛЕ overlap-зоны (2.0 > overlap_sec=1.0) — не должен отбрасываться
            return [TranscriptSegment(
                text="ok", start=2.0, end=3.0, words=[Word("ok", 2.0, 3.0)],
            )]

        monkeypatch.setattr(
            "reels_agent.pipeline.transcribe.transcribe_wav", fake_transcribe_wav
        )

        segments = transcribe_long_audio(long_wav, tmp_path, overlap_sec=1)
        starts = sorted(s.start for s in segments)
        # 4 чанка co смещениями 0,5,10,15 — каждый добавляет локальный сегмент start=2.0
        assert starts == [2.0, 7.0, 12.0, 17.0]
        # таймкоды слов тоже должны быть сдвинуты, не только сегментов
        for seg in segments:
            assert seg.words[0].start == seg.start

    def test_short_file_skips_chunking_entirely(self, long_wav, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "reels_agent.pipeline.transcribe.transcribe_wav",
            lambda path, language="ru": (calls.append(path), [])[1],
        )
        transcribe_long_audio(long_wav, long_wav.parent)
        assert len(calls) == 1
        assert calls[0] == long_wav
