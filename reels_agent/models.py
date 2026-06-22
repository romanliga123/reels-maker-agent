"""Dataclasses, передаваемые между стадиями пайплайна."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class ClipCandidate:
    id: str
    start: float
    end: float
    reason: str          # человекочитаемое объяснение, напр. "🔥 пик смеха"
    score: float
    source: str           # "audio" | "llm" | "manual" | "audio+llm"
    transcript_excerpt: str = ""
    approved: bool = False
    subtitle_style: str = "dynamic"  # "dynamic" | "static"


@dataclass
class RenderResult:
    clip_id: str
    output_path: str
    duration: float
    error: str | None = None
