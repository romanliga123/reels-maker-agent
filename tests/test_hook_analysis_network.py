import pytest

from reels_agent.pipeline.hook_analysis import analyze_hooks, HookSpan
from tests.conftest import has_groq_key

pytestmark = pytest.mark.network
requires_groq = pytest.mark.skipif(not has_groq_key(), reason="требуется реальный GROQ_API_KEY")


@requires_groq
def test_analyze_hooks_real_api_returns_valid_spans(fake_transcript):
    """fake_transcript содержит настоящие русские предложения — это реальный smoke-тест
    того, что Groq возвращает валидный JSON, парсящийся в HookSpan."""
    spans = analyze_hooks(fake_transcript, window_sec=1000)
    assert isinstance(spans, list)
    for span in spans:
        assert isinstance(span, HookSpan)
        assert span.start >= fake_transcript[0].start
        assert span.end <= fake_transcript[-1].end
        assert span.kind in ("hook", "joke", "thesis")
