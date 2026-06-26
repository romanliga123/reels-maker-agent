"""Эвристика "пик смеха / эмоций" по аудио: RMS-энергия + zero-crossing-rate,
z-нормированные относительно самого файла (своя база сравнения для каждой
записи, а не глобальный порог).

Это не ML-классификатор смеха — это эвристика для v1: бурст громкости +
высокая ZCR (шум/хрипота смеха) часто сопровождают смех и эмоциональные пики
речи. Достаточно для ранжирования кандидатов в клипы, не для точной
классификации "это именно смех".

Раньше сюда же добавлялась нестабильность высоты тона через librosa.yin, но
yin использует numba (@stencil/@guvectorize) — JIT-компиляция при первом
вызове в процессе разово съедает много памяти и роняла Render free tier
(512MB) по OOM именно на этом шаге. RMS/ZCR заменили на чистый numpy.

Сам librosa тоже убрали целиком (даже под ленивым импортом) — простой
`import librosa` тянет numba/llvmlite транзитивно через другие подмодули
librosa, и память от JIT-движка остаётся в процессе навсегда (gc/malloc_trim
её не освобождают, это не "висящая" куча, а реально загруженный код). На
коротком (4.3 мин) тестовом видео это поднимало RSS до ~500MB ещё до старта
рендера. Читаем WAV сами через stdlib `wave` (файл — наш собственный, мы
точно знаем формат: mono 16-bit PCM 16kHz, как и просили у ffmpeg в
audio_extract.py), framing/RMS/ZCR — через numpy stride tricks.
"""
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class EnergySpan:
    start: float
    end: float
    score: float  # z-нормированная интенсивность (для ранжирования между собой)


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def _frame_features(y: np.ndarray, frame_length: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    """RMS и zero-crossing-rate по фреймам — эквивалент librosa.feature.rms/
    zero_crossing_rate с center=False, но без зависимости от librosa/numba."""
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    signs = np.sign(frames)
    signs[signs == 0] = 1  # тишина (0) не считаем переходом через ноль
    zcr = np.mean(signs[:, :-1] != signs[:, 1:], axis=1)
    return rms, zcr


def detect_energy_spans(
    wav_path: Path,
    min_span_sec: float = 1.5,
    merge_gap_sec: float = 1.0,
    z_threshold: float = 1.3,
    chunk_sec: float = 180.0,
    on_progress: Callable[[float], None] | None = None,
) -> list[EnergySpan]:
    sr = 16000
    frame_length = int(sr * 0.05)   # 50ms
    hop_length = int(sr * 0.025)    # 25ms

    rms_parts, zcr_parts, time_parts = [], [], []
    with wave.open(str(wav_path), "rb") as wf:
        total_frames = wf.getnframes()
        total_duration = total_frames / sr
        if total_duration <= 0:
            return []

        chunk_samples = int(chunk_sec * sr)
        offset_samples = 0
        while offset_samples < total_frames:
            raw = wf.readframes(chunk_samples)
            if not raw:
                break
            y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if len(y) < frame_length:
                break
            rms_chunk, zcr_chunk = _frame_features(y, frame_length, hop_length)
            n_chunk = min(len(rms_chunk), len(zcr_chunk))
            if n_chunk > 0:
                rms_parts.append(rms_chunk[:n_chunk])
                zcr_parts.append(zcr_chunk[:n_chunk])
                frame_offset = offset_samples / sr
                time_parts.append(frame_offset + np.arange(n_chunk) * hop_length / sr)
            offset_samples += len(y)
            if on_progress:
                on_progress(min(1.0, offset_samples / total_frames))

    if not rms_parts:
        return []

    rms = np.concatenate(rms_parts)
    zcr = np.concatenate(zcr_parts)
    times = np.concatenate(time_parts)
    n = len(rms)

    composite = 0.55 * _zscore(rms) + 0.45 * _zscore(zcr)

    smooth_window = max(1, int(0.4 / (hop_length / sr)))  # ~0.4s
    composite = _smooth(composite, smooth_window)

    above = composite > z_threshold
    spans: list[EnergySpan] = []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n:
            if above[j]:
                j += 1
                continue
            # допускаем короткий провал ниже порога внутри одного всплеска
            gap_end = j
            while gap_end < n and not above[gap_end] and (times[gap_end] - times[j - 1]) <= merge_gap_sec:
                gap_end += 1
            if gap_end < n and above[gap_end]:
                j = gap_end
                continue
            break
        start, end = times[i], times[min(j, n - 1)]
        if end - start >= min_span_sec:
            score = float(composite[i:j].max())
            spans.append(EnergySpan(start=float(start), end=float(end), score=score))
        i = j

    spans.sort(key=lambda s: s.score, reverse=True)
    return spans
