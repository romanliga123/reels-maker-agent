"""LLM-анализ транскрипта: ищем хуки, шутки и ключевые тезисы — кандидатов в Reels.

Транскрипт режется на окна ~window_sec, каждое окно отдельно отправляется в Groq
с таймкодами строкой за строкой; модель обязана вернуть строгий JSON-массив.
Никакой арифметики моделью не требуется — она только классифицирует моменты,
сами тайминги уже посчитаны нами (см. feedback "llm_cant_do_math" — не доверяем
LLM числа, которые можно посчитать самим; здесь числа просто переписываются из
входных данных, а не вычисляются).
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from groq import Groq

from .. import config
from ..models import TranscriptSegment


class HookAnalysisError(Exception):
    pass


@dataclass
class HookSpan:
    start: float
    end: float
    reason: str
    kind: str  # "hook" | "joke" | "thesis"


GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "gemma2-9b-it",
]

SYSTEM_PROMPT = (
    "Ты — опытный редактор коротких видео (Reels/TikTok/Shorts) с 5-летним стажем. "
    "Тебе дают фрагмент транскрипта подкаста/стрима с таймкодами в секундах. "
    "Найди моменты, которые стоит вырезать в отдельный клип: "
    "hook — цепляющая фраза или неожиданный поворот, joke — шутка/смешной момент, "
    "thesis — самостоятельная законченная мысль или ценный вывод. "
    "Каждый момент должен быть смысловым отрывком длиной от 15 до 90 секунд. "
    "Используй ТОЛЬКО таймкоды, присутствующие в исходном тексте — не придумывай свои. "
    "Ответь СТРОГО JSON-массивом без пояснений и без markdown, формат каждого элемента: "
    '{"start": <число>, "end": <число>, "kind": "hook|joke|thesis", "reason": "<короткое объяснение на русском>"}. '
    "Если в отрывке ничего интересного нет — верни пустой массив []."
)


def _format_window(segments: list[TranscriptSegment]) -> str:
    lines = [f"[{seg.start:.1f}] {seg.text}" for seg in segments]
    return "\n".join(lines)


def _extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return []


def _call_groq_json(client: Groq, prompt: str) -> list[dict]:
    last_error = None
    for model in GROQ_FALLBACK_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.2,
            )
            return _extract_json_array(response.choices[0].message.content)
        except Exception as e:
            error_str = str(e).lower()
            if any(k in error_str for k in ("rate", "429", "503", "capacity", "model", "overloaded", "unavailable")):
                last_error = e
                continue
            raise HookAnalysisError(f"Groq API Error: {e}")
    raise HookAnalysisError(f"Groq: все модели недоступны. {last_error}")


def _windows(segments: list[TranscriptSegment], window_sec: float):
    if not segments:
        return
    window: list[TranscriptSegment] = []
    window_start = segments[0].start
    for seg in segments:
        if seg.start - window_start > window_sec and window:
            yield window
            window = []
            window_start = seg.start
        window.append(seg)
    if window:
        yield window


def analyze_hooks(segments: list[TranscriptSegment], window_sec: float = 240) -> list[HookSpan]:
    if not config.GROQ_API_KEY:
        raise HookAnalysisError("GROQ_API_KEY не задан")

    client = Groq(api_key=config.GROQ_API_KEY, timeout=120.0)
    spans: list[HookSpan] = []

    for window in _windows(segments, window_sec):
        window_text = _format_window(window)
        if not window_text.strip():
            continue
        prompt = f"Транскрипт (таймкоды в секундах от начала видео):\n{window_text}"
        items = _call_groq_json(client, prompt)

        win_start, win_end = window[0].start, window[-1].end
        for item in items:
            try:
                start = float(item["start"])
                end = float(item["end"])
                kind = str(item.get("kind", "thesis"))
                reason = str(item.get("reason", "")).strip()
            except (KeyError, TypeError, ValueError):
                continue
            # модель иногда чуть выходит за границы окна — обрезаем по факту, не доверяем слепо
            start = max(start, win_start)
            end = min(end, win_end)
            if end - start < 5:
                continue
            spans.append(HookSpan(start=start, end=end, reason=reason, kind=kind))

    return spans
