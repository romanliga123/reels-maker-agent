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
(512MB) по OOM именно на этом шаге. RMS/ZCR — чистый numpy, без JIT.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
# librosa тянет numba (JIT-компиляция при импорте — медленно на слабом CPU,
# особенно на Render free tier). Импортируем лениво внутри функции, чтобы
# веб-сервер стартовал мгновенно и не падал по health-check таймауту.


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


def detect_energy_spans(
    wav_path: Path,
    min_span_sec: float = 1.5,
    merge_gap_sec: float = 1.0,
    z_threshold: float = 1.3,
    chunk_sec: float = 180.0,
    on_progress: Callable[[float], None] | None = None,
) -> list[EnergySpan]:
    import librosa

    sr = 16000
    frame_length = int(sr * 0.05)   # 50ms
    hop_length = int(sr * 0.025)    # 25ms

    total_duration = librosa.get_duration(path=str(wav_path))
    if total_duration <= 0:
        return []

    # rms/zcr framируют сигнал в память (frame_length x n_frames) — на длинном аудио
    # это растёт пропорционально длительности. Читаем и считаем фичи по чанкам
    # (librosa.load с offset/duration не грузит файл целиком), склеивая только лёгкие
    # 1-D результаты — поведение алгоритма (z-score по всему файлу, сглаживание,
    # склейка всплесков) не меняется.
    rms_parts, zcr_parts, time_parts = [], [], []
    offset = 0.0
    while offset < total_duration:
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True, offset=offset, duration=chunk_sec)
        if len(y) < frame_length:
            break
        rms_chunk = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=False)[0]
        zcr_chunk = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length, center=False)[0]
        n_chunk = min(len(rms_chunk), len(zcr_chunk))
        if n_chunk > 0:
            rms_parts.append(rms_chunk[:n_chunk])
            zcr_parts.append(zcr_chunk[:n_chunk])
            time_parts.append(offset + librosa.frames_to_time(np.arange(n_chunk), sr=sr, hop_length=hop_length))
        offset += len(y) / sr  # реальная длина чанка (последний может быть короче chunk_sec)
        if on_progress:
            on_progress(min(1.0, offset / total_duration))

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
