"""Генерация .ass субтитров для клипа: статичные построчно или караоке по словам.

Шрифт Arial выбран намеренно — он почти всегда есть в системе и корректно
рендерит кириллицу через libass (ffmpeg `ass=` фильтр). Если шрифта Arial нет
в системе рендеринга, libass подставит похожий моноширинный — кириллица
не должна превращаться в "коробки", это и есть риск, который проверяем на шаге
рендера (день один — Cyrillic smoke-test).
"""
from ..models import TranscriptSegment

FONT_NAME = "Arial"

_HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{FONT_NAME},72,&H00FFFFFF,&H0000FFFF,&H00000000,&H96000000,1,0,0,0,100,100,0,0,1,4,2,2,60,60,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _fmt_time(sec: float) -> str:
    # Считаем всё в целых сантисекундах — иначе округление при sec вида X.999
    # переносит секунды в "60" без переноса в минуты/часы (невалидный таймкод ASS).
    total_cs = int(round(max(sec, 0.0) * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clip_segments(segments: list[TranscriptSegment], clip_start: float, clip_end: float):
    for seg in segments:
        if seg.end <= clip_start or seg.start >= clip_end:
            continue
        yield seg


def build_static_ass(segments: list[TranscriptSegment], clip_start: float, clip_end: float) -> str:
    lines = []
    for seg in _clip_segments(segments, clip_start, clip_end):
        start = max(seg.start, clip_start) - clip_start
        end = min(seg.end, clip_end) - clip_start
        text = seg.text.replace("\n", " ").strip()
        if not text:
            continue
        lines.append(f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},Default,,0,0,0,,{text}")
    return _HEADER + "\n".join(lines) + ("\n" if lines else "")


def build_karaoke_ass(segments: list[TranscriptSegment], clip_start: float, clip_end: float) -> str:
    lines = []
    for seg in _clip_segments(segments, clip_start, clip_end):
        start = max(seg.start, clip_start) - clip_start
        end = min(seg.end, clip_end) - clip_start
        words = [w for w in seg.words if w.end > clip_start and w.start < clip_end]
        if not words:
            text = seg.text.replace("\n", " ").strip()
            if not text:
                continue
            lines.append(f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},Default,,0,0,0,,{text}")
            continue

        prev_end = max(seg.start, clip_start)
        parts = []
        for w in words:
            w_start = max(w.start, clip_start)
            w_end = min(w.end, clip_end)
            k_cs = max(1, round((w_end - prev_end) * 100))
            parts.append(f"{{\\k{k_cs}}}{w.text.strip()} ")
            prev_end = w_end
        text = "".join(parts).strip()
        lines.append(f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},Default,,0,0,0,,{text}")
    return _HEADER + "\n".join(lines) + ("\n" if lines else "")


def build_ass(segments: list[TranscriptSegment], clip_start: float, clip_end: float, style: str = "dynamic") -> str:
    if style == "static":
        return build_static_ass(segments, clip_start, clip_end)
    return build_karaoke_ass(segments, clip_start, clip_end)
