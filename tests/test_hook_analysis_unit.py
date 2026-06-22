from unittest.mock import MagicMock

from reels_agent.pipeline.hook_analysis import (
    _extract_json_array, _windows, analyze_hooks, HookAnalysisError,
)
from reels_agent.models import TranscriptSegment


class TestExtractJsonArray:
    def test_plain_json(self):
        assert _extract_json_array('[{"start": 1, "end": 2}]') == [{"start": 1, "end": 2}]

    def test_json_wrapped_in_prose(self):
        text = 'Вот результат:\n[{"start": 1, "end": 2}]\nНадеюсь, помогло!'
        assert _extract_json_array(text) == [{"start": 1, "end": 2}]

    def test_json_in_markdown_fence(self):
        text = '```json\n[{"start": 1, "end": 2}]\n```'
        assert _extract_json_array(text) == [{"start": 1, "end": 2}]

    def test_empty_array(self):
        assert _extract_json_array("[]") == []

    def test_garbage_returns_empty_list(self):
        assert _extract_json_array("это не json вообще") == []


class TestWindows:
    def test_groups_by_window_sec(self, fake_transcript):
        windows = list(_windows(fake_transcript, window_sec=5))
        # сегменты на 0.0, 4.0, 8.0 — окно 5с должно разбить на минимум 2 группы
        assert len(windows) >= 2

    def test_single_window_when_window_sec_covers_all(self, fake_transcript):
        windows = list(_windows(fake_transcript, window_sec=1000))
        assert len(windows) == 1
        assert len(windows[0]) == len(fake_transcript)

    def test_empty_segments_yields_nothing(self):
        assert list(_windows([], window_sec=240)) == []


class TestAnalyzeHooks:
    def test_raises_without_api_key(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "")
        try:
            analyze_hooks(fake_transcript)
            assert False, "должно было выбросить HookAnalysisError"
        except HookAnalysisError:
            pass

    def test_clamps_items_outside_window(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(
            "reels_agent.pipeline.hook_analysis._call_groq_json",
            lambda client, prompt: [
                {"start": -100, "end": 999, "kind": "hook", "reason": "вышел за границы окна"},
            ],
        )
        spans = analyze_hooks(fake_transcript, window_sec=1000)
        assert len(spans) == 1
        assert spans[0].start >= fake_transcript[0].start
        assert spans[0].end <= fake_transcript[-1].end

    def test_drops_too_short_items(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(
            "reels_agent.pipeline.hook_analysis._call_groq_json",
            lambda client, prompt: [
                {"start": 0, "end": 1, "kind": "joke", "reason": "слишком короткий"},
            ],
        )
        spans = analyze_hooks(fake_transcript, window_sec=1000)
        assert spans == []

    def test_skips_malformed_items(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(
            "reels_agent.pipeline.hook_analysis._call_groq_json",
            lambda client, prompt: [
                {"start": "not-a-number", "end": 5, "kind": "hook", "reason": "битый"},
                {"start": 0, "end": 10, "kind": "thesis", "reason": "нормальный"},
            ],
        )
        spans = analyze_hooks(fake_transcript, window_sec=1000)
        assert len(spans) == 1
        assert spans[0].reason == "нормальный"
