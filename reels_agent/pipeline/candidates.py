"""Сборка финальных ClipCandidate из аудио-эвристики + LLM-хуков (+ ручных меток).

Алгоритм:
1. Каждый сигнал (хук/шутка/тезис от LLM, эмоциональный всплеск от аудио)
   превращается в "сырой" временной интервал.
2. Интервалы, которые пересекаются по времени, схлопываются в один кандидат
   (двойное подтверждение сигнала повышает его итоговый score).
3. Каждый итоговый интервал привязывается к границам предложений по
   транскрипту (не режем по середине фразы) и подгоняется под допустимую
   длительность клипа [CLIP_MIN_SEC, CLIP_MAX_SEC].
4. Ручные метки добавляются отдельно, без слияния — это явный выбор пользователя.
"""
import uuid
from dataclasses import dataclass, field

from .. import config
from ..models import ClipCandidate, TranscriptSegment
from .audio_energy import EnergySpan
from .hook_analysis import HookSpan

KIND_LABELS = {
    "hook": "🪝 Цепляющий момент",
    "joke": "😂 Шутка",
    "thesis": "💡 Ключевая мысль",
}
KIND_SCORE = {"hook": 2.5, "joke": 2.5, "thesis": 2.0}
AUDIO_LABEL = "🔥 Пик эмоций/смеха"

ENERGY_PAD_TARGET_SEC = 20.0


@dataclass
class _RawSpan:
    start: float
    end: float
    score: float
    reasons: list[str] = field(default_factory=list)
    sources: set = field(default_factory=set)


def _snap_to_sentences(start: float, end: float, segments: list[TranscriptSegment]) -> tuple[float, float]:
    if not segments:
        return start, end
    new_start = start
    for seg in segments:
        if seg.end > start:
            new_start = seg.start
            break
    new_end = end
    for seg in reversed(segments):
        if seg.start < end:
            new_end = seg.end
            break
    if new_end <= new_start:
        return start, end
    return new_start, new_end


def _clamp_duration(start: float, end: float, segments: list[TranscriptSegment]) -> tuple[float, float]:
    duration = end - start

    if duration > config.CLIP_MAX_SEC:
        max_end = start + config.CLIP_MAX_SEC
        snapped_end = None
        for seg in segments:
            if seg.start < start:
                continue
            if seg.end <= max_end:
                snapped_end = seg.end
            else:
                break
        end = snapped_end if snapped_end and snapped_end > start else max_end
        duration = end - start

    if duration < config.CLIP_MIN_SEC:
        min_end = start + config.CLIP_MIN_SEC
        for seg in segments:
            if seg.end <= end:
                continue
            end = seg.end
            if end >= min_end:
                break
        duration = end - start
        if duration < config.CLIP_MIN_SEC:
            # совсем короткий хвост видео — тянем начало назад
            min_start = end - config.CLIP_MIN_SEC
            for seg in reversed(segments):
                if seg.start >= start:
                    continue
                start = seg.start
                if start <= min_start:
                    break

    return max(start, 0.0), end


def _transcript_excerpt(start: float, end: float, segments: list[TranscriptSegment], limit: int = 300) -> str:
    parts = [seg.text for seg in segments if seg.start < end and seg.end > start]
    text = " ".join(parts).strip()
    return text[:limit]


def _merge_overlapping(raw: list[_RawSpan]) -> list[_RawSpan]:
    raw = sorted(raw, key=lambda r: r.start)
    merged: list[_RawSpan] = []
    for r in raw:
        if merged and r.start <= merged[-1].end:
            last = merged[-1]
            last.end = max(last.end, r.end)
            last.start = min(last.start, r.start)
            last.score += r.score
            for reason in r.reasons:
                if reason not in last.reasons:
                    last.reasons.append(reason)
            last.sources |= r.sources
        else:
            merged.append(_RawSpan(start=r.start, end=r.end, score=r.score,
                                    reasons=list(r.reasons), sources=set(r.sources)))
    return merged


def build_candidates(
    transcript: list[TranscriptSegment],
    energy_spans: list[EnergySpan],
    hook_spans: list[HookSpan],
    manual: list[tuple[float, float]] = (),
    refined_joke_spans: list[HookSpan | None] = (),
) -> list[ClipCandidate]:
    """refined_joke_spans — по одному элементу на energy_spans (тот же порядок),
    от hook_analysis.refine_laughter_spans: LLM смотрит транскрипт ПЕРЕД всплеском
    смеха/аплодисментов и ищет настоящий сетап шутки/истории вместо симметричного
    отступа от самого всплеска. None — там, где LLM не нашла ясной причины реакции,
    для них остаётся старый отступ ENERGY_PAD_TARGET_SEC."""
    raw: list[_RawSpan] = []

    for h in hook_spans:
        label = KIND_LABELS.get(h.kind, "💡 Интересный момент")
        reason = f"{label}: {h.reason}" if h.reason else label
        raw.append(_RawSpan(start=h.start, end=h.end, score=KIND_SCORE.get(h.kind, 2.0),
                             reasons=[reason], sources={"llm"}))

    refined = list(refined_joke_spans) if refined_joke_spans else [None] * len(energy_spans)
    for e, joke in zip(energy_spans, refined):
        if joke is not None:
            label = KIND_LABELS.get("joke", "😂 Шутка")
            reason = f"{label} (смех + сетап по тексту): {joke.reason}" if joke.reason else label
            raw.append(_RawSpan(start=joke.start, end=joke.end, score=KIND_SCORE.get("joke", 2.5) + e.score,
                                 reasons=[reason], sources={"audio", "llm"}))
            continue
        center = (e.start + e.end) / 2
        half = max((e.end - e.start) / 2, ENERGY_PAD_TARGET_SEC / 2)
        raw.append(_RawSpan(start=max(center - half, 0.0), end=center + half, score=e.score,
                             reasons=[AUDIO_LABEL], sources={"audio"}))

    merged = _merge_overlapping(raw)

    candidates: list[ClipCandidate] = []
    for m in merged:
        start, end = _snap_to_sentences(m.start, m.end, transcript)
        start, end = _clamp_duration(start, end, transcript)
        if end <= start:
            continue
        score = m.score + (1.0 if len(m.sources) > 1 else 0.0)
        candidates.append(ClipCandidate(
            id=str(uuid.uuid4())[:8],
            start=start,
            end=end,
            reason=" + ".join(m.reasons),
            score=round(score, 2),
            source="+".join(sorted(m.sources)),
            transcript_excerpt=_transcript_excerpt(start, end, transcript),
        ))

    for ms, me in manual:
        candidates.append(make_manual_candidate(ms, me, transcript))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def make_manual_candidate(start: float, end: float, transcript: list[TranscriptSegment]) -> ClipCandidate:
    start, end = _clamp_duration(start, end, transcript)
    return ClipCandidate(
        id=str(uuid.uuid4())[:8],
        start=start,
        end=end,
        reason="✍️ Ручная отметка",
        score=999.0,
        source="manual",
        transcript_excerpt=_transcript_excerpt(start, end, transcript),
    )
