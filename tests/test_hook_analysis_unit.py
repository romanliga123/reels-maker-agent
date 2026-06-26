from unittest.mock import MagicMock

from reels_agent.pipeline.hook_analysis import (
    _extract_json_array, _extract_json_object, _windows, analyze_hooks, refine_laughter_spans,
    refine_joke_text_boundaries, HookSpan, HookAnalysisError,
)
from reels_agent.pipeline.audio_energy import EnergySpan
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

    def test_on_progress_reaches_100_percent(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr("reels_agent.pipeline.hook_analysis._call_groq_json", lambda client, prompt: [])

        fractions = []
        analyze_hooks(fake_transcript, window_sec=5, on_progress=fractions.append)

        assert fractions[-1] == 1.0
        assert all(0.0 <= f <= 1.0 for f in fractions)


class TestExtractJsonObject:
    def test_plain_json(self):
        assert _extract_json_object('{"start": 1, "end": 2}') == {"start": 1, "end": 2}

    def test_null_returns_none(self):
        assert _extract_json_object("null") is None

    def test_json_wrapped_in_prose(self):
        text = 'Вот результат:\n{"start": 1, "end": 2}\nНадеюсь, помогло!'
        assert _extract_json_object(text) == {"start": 1, "end": 2}

    def test_garbage_returns_none(self):
        assert _extract_json_object("это не json вообще") is None


class TestRefineLaughterSpans:
    def test_no_api_key_returns_all_none(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "")
        energy = [EnergySpan(start=5.0, end=6.0, score=2.0)]
        assert refine_laughter_spans(energy, fake_transcript) == [None]

    def test_empty_energy_spans_returns_empty(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        assert refine_laughter_spans([], fake_transcript) == []

    def test_llm_finds_joke_boundaries(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(
            "reels_agent.pipeline.hook_analysis._call_groq_object",
            lambda client, system_prompt, prompt: {"start": 0.0, "end": 8.0, "reason": "сетап про второй сегмент"},
        )
        energy = [EnergySpan(start=8.5, end=9.0, score=2.0)]
        results = refine_laughter_spans(energy, fake_transcript)
        assert len(results) == 1
        assert results[0] is not None
        assert results[0].kind == "joke"
        assert results[0].start == 0.0
        assert results[0].end == 8.0

    def test_llm_returns_null_keeps_none(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr("reels_agent.pipeline.hook_analysis._call_groq_object", lambda client, sp, prompt: None)
        energy = [EnergySpan(start=8.5, end=9.0, score=2.0)]
        assert refine_laughter_spans(energy, fake_transcript) == [None]

    def test_too_short_result_dropped(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(
            "reels_agent.pipeline.hook_analysis._call_groq_object",
            lambda client, system_prompt, prompt: {"start": 8.0, "end": 8.5, "reason": "слишком коротко"},
        )
        energy = [EnergySpan(start=8.5, end=9.0, score=2.0)]
        assert refine_laughter_spans(energy, fake_transcript) == [None]

    def test_preserves_order_and_length(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        calls = []

        def fake_call(client, system_prompt, prompt):
            calls.append(prompt)
            if len(calls) == 1:
                return {"start": 0.0, "end": 4.0, "reason": "первая"}
            return None

        monkeypatch.setattr("reels_agent.pipeline.hook_analysis._call_groq_object", fake_call)
        energy = [EnergySpan(start=2.0, end=2.5, score=1.0), EnergySpan(start=9.0, end=9.5, score=1.5)]
        results = refine_laughter_spans(energy, fake_transcript)
        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is None

    def test_on_progress_reaches_100_percent(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr("reels_agent.pipeline.hook_analysis._call_groq_object", lambda client, sp, prompt: None)
        energy = [EnergySpan(start=2.0, end=2.5, score=1.0), EnergySpan(start=9.0, end=9.5, score=1.5)]

        fractions = []
        refine_laughter_spans(energy, fake_transcript, on_progress=fractions.append)
        assert fractions[-1] == 1.0


class TestRefineJokeTextBoundaries:
    def test_no_api_key_returns_unchanged(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "")
        hooks = [HookSpan(start=4.0, end=8.0, reason="грубая оценка", kind="joke")]
        assert refine_joke_text_boundaries(hooks, fake_transcript) == hooks

    def test_empty_input_returns_empty(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        assert refine_joke_text_boundaries([], fake_transcript) == []

    def test_llm_widens_setup_backwards(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr(
            "reels_agent.pipeline.hook_analysis._call_groq_object",
            lambda client, system_prompt, prompt: {"start": 0.0, "end": 9.5, "reason": "нашёлся сетап раньше"},
        )
        hooks = [HookSpan(start=8.0, end=9.5, reason="грубая оценка", kind="joke")]
        results = refine_joke_text_boundaries(hooks, fake_transcript)
        assert len(results) == 1
        assert results[0].start == 0.0
        assert results[0].end == 9.5
        assert results[0].reason == "нашёлся сетап раньше"

    def test_llm_returns_null_keeps_original(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr("reels_agent.pipeline.hook_analysis._call_groq_object", lambda client, sp, prompt: None)
        original = HookSpan(start=4.0, end=8.0, reason="оригинал", kind="joke")
        results = refine_joke_text_boundaries([original], fake_transcript)
        assert results == [original]

    def test_on_progress_reaches_100_percent(self, fake_transcript, monkeypatch):
        from reels_agent import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key-for-test")
        monkeypatch.setattr("reels_agent.pipeline.hook_analysis._call_groq_object", lambda client, sp, prompt: None)
        hooks = [
            HookSpan(start=0.0, end=4.0, reason="a", kind="joke"),
            HookSpan(start=4.0, end=8.0, reason="b", kind="joke"),
        ]
        fractions = []
        refine_joke_text_boundaries(hooks, fake_transcript, on_progress=fractions.append)
        assert fractions[-1] == 1.0
