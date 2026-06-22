"""Транскрипция аудио через Groq hosted Whisper (word-level таймкоды).

Для файлов в пределах лимита размера используется transcribe_wav() напрямую.
Для длинных файлов transcribe_long_audio() режет WAV на ~10-минутные чанки
с небольшим перекрытием (чтобы не терять слова на стыке), транскрибирует
каждый чанк отдельно и сшивает результат, сдвигая таймкоды на смещение чанка.
"""
import subprocess
from pathlib import Path

from groq import Groq

from .. import config
from ..models import Word, TranscriptSegment


class TranscribeError(Exception):
    pass


def _client() -> Groq:
    if not config.GROQ_API_KEY:
        raise TranscribeError("GROQ_API_KEY не задан")
    return Groq(api_key=config.GROQ_API_KEY, timeout=300.0)


def transcribe_wav(wav_path: Path, language: str = "ru") -> list[TranscriptSegment]:
    """Транскрибирует один WAV-файл (должен быть в пределах лимита размера Groq API)."""
    size = wav_path.stat().st_size
    if size > config.WHISPER_CHUNK_LIMIT_BYTES:
        raise TranscribeError(
            f"Файл {size/1024/1024:.1f}MB превышает лимит без чанкинга "
            f"({config.WHISPER_CHUNK_LIMIT_BYTES/1024/1024:.0f}MB)"
        )

    client = _client()
    with open(wav_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=(wav_path.name, f.read()),
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    segments_raw = getattr(resp, "segments", None) or []
    words_raw = getattr(resp, "words", None) or []

    words_by_segment: list[list[Word]] = [[] for _ in segments_raw]
    for w in words_raw:
        ws, we = float(w["start"]), float(w["end"])
        word = Word(text=w["word"], start=ws, end=we)
        idx = _segment_index_for(segments_raw, ws)
        if idx is not None:
            words_by_segment[idx].append(word)

    segments: list[TranscriptSegment] = []
    for i, seg in enumerate(segments_raw):
        segments.append(TranscriptSegment(
            text=seg["text"].strip(),
            start=float(seg["start"]),
            end=float(seg["end"]),
            words=words_by_segment[i],
        ))

    if not segments and getattr(resp, "text", ""):
        # Фолбэк: модель не вернула сегменты (короткий клип) — один сегмент на весь текст
        segments.append(TranscriptSegment(text=resp.text.strip(), start=0.0, end=0.0, words=[]))

    return segments


def _segment_index_for(segments_raw: list[dict], word_start: float) -> int | None:
    for i, seg in enumerate(segments_raw):
        if seg["start"] <= word_start <= seg["end"]:
            return i
    return None


def _wav_duration(wav_path: Path) -> float:
    cmd = [
        config.FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise TranscribeError(f"ffprobe не смог определить длительность wav: {out.stderr.strip()[:200]}")
    return float(out.stdout.strip())


def split_audio_chunks(wav_path: Path, chunk_dir: Path,
                        chunk_sec: int = None, overlap_sec: int = 5) -> list[tuple[Path, float]]:
    """Режет WAV на чанки [(путь, смещение_начала_сек), ...] с overlap_sec перекрытием."""
    chunk_sec = chunk_sec or config.WHISPER_CHUNK_TARGET_SEC
    chunk_dir.mkdir(parents=True, exist_ok=True)
    total = _wav_duration(wav_path)

    chunks: list[tuple[Path, float]] = []
    start = 0.0
    i = 0
    while start < total:
        read_len = min(chunk_sec + overlap_sec, total - start)
        out_path = chunk_dir / f"chunk_{i:03d}.wav"
        cmd = [
            config.FFMPEG_BIN, "-y", "-ss", str(start), "-t", str(read_len),
            "-i", str(wav_path), "-ac", "1", "-ar", "16000", "-f", "wav", str(out_path),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            raise TranscribeError(f"ffmpeg не смог нарезать чанк {i}: {out.stderr.strip()[:200]}")
        chunks.append((out_path, start))
        start += chunk_sec
        i += 1

    return chunks


def transcribe_long_audio(wav_path: Path, work_dir: Path, language: str = "ru",
                           overlap_sec: int = 5) -> list[TranscriptSegment]:
    """Транскрибирует длинный WAV через чанкинг и сшивает результат с офсетом таймкодов."""
    size = wav_path.stat().st_size
    if size <= config.WHISPER_CHUNK_LIMIT_BYTES:
        return transcribe_wav(wav_path, language)

    chunk_dir = work_dir / "chunks"
    chunks = split_audio_chunks(wav_path, chunk_dir, overlap_sec=overlap_sec)

    merged: list[TranscriptSegment] = []
    for i, (chunk_path, offset) in enumerate(chunks):
        chunk_segments = transcribe_wav(chunk_path, language)
        for seg in chunk_segments:
            # Перекрытие отрезаем только у не-первых чанков: всё, что начинается
            # внутри overlap_sec от начала чанка, уже покрыто хвостом предыдущего.
            if i > 0 and seg.start < overlap_sec:
                continue
            shifted_words = [Word(text=w.text, start=w.start + offset, end=w.end + offset) for w in seg.words]
            merged.append(TranscriptSegment(
                text=seg.text,
                start=seg.start + offset,
                end=seg.end + offset,
                words=shifted_words,
            ))

    return merged
