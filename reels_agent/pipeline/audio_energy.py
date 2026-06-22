"""Эвристика "пик смеха / эмоций" по аудио: RMS-энергия + zero-crossing-rate
+ нестабильность высоты тона, z-нормированные относительно самого файла
(своя база сравнения для каждой записи, а не глобальный порог).

Это не ML-классификатор смеха — это эвристика для v1: бурст громкости +
высокая ZCR (шум/хрипота смеха) + скачущий pitch часто сопровождают смех и
эмоциональные пики речи. Достаточно для ранжирования кандидатов в клипы,
не для точной классификации "это именно смех".
"""
from dataclasses import dataclass
from pathlib import Path

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
) -> list[EnergySpan]:
    import librosa

    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    if len(y) == 0:
        return []

    frame_length = int(sr * 0.05)   # 50ms
    hop_length = int(sr * 0.025)    # 25ms

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    f0 = librosa.yin(y, fmin=80, fmax=500, sr=sr, frame_length=frame_length * 2, hop_length=hop_length)

    n = min(len(rms), len(zcr), len(f0))
    rms, zcr, f0 = rms[:n], zcr[:n], f0[:n]

    pitch_var = np.abs(np.diff(f0, prepend=f0[0]))
    pitch_var = np.nan_to_num(pitch_var, nan=0.0, posinf=0.0, neginf=0.0)

    composite = 0.4 * _zscore(rms) + 0.3 * _zscore(zcr) + 0.3 * _zscore(pitch_var)

    smooth_window = max(1, int(0.4 / (hop_length / sr)))  # ~0.4s
    composite = _smooth(composite, smooth_window)

    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)

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
