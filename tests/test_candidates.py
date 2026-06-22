from reels_agent.pipeline.candidates import build_candidates, make_manual_candidate
from reels_agent.pipeline.audio_energy import EnergySpan
from reels_agent.pipeline.hook_analysis import HookSpan
from reels_agent import config


class TestBuildCandidatesMerging:
    def test_hook_only_produces_llm_candidate(self, fake_transcript):
        hooks = [HookSpan(start=0.0, end=4.0, reason="вступление", kind="hook")]
        candidates = build_candidates(fake_transcript, [], hooks)
        assert len(candidates) == 1
        assert candidates[0].source == "llm"
        assert "вступление" in candidates[0].reason

    def test_energy_only_produces_audio_candidate(self, fake_transcript):
        energy = [EnergySpan(start=5.0, end=6.0, score=2.0)]
        candidates = build_candidates(fake_transcript, energy, [])
        assert len(candidates) == 1
        assert candidates[0].source == "audio"

    def test_overlapping_hook_and_energy_merge(self, fake_transcript):
        hooks = [HookSpan(start=4.0, end=8.0, reason="тезис", kind="thesis")]
        energy = [EnergySpan(start=5.0, end=6.0, score=2.0)]
        candidates = build_candidates(fake_transcript, energy, hooks)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.source == "audio+llm"
        assert "тезис" in c.reason
        # бонус за двойное подтверждение сигнала
        assert c.score > 2.0

    def test_non_overlapping_signals_stay_separate(self, long_transcript):
        # энергетический всплеск паддится до ENERGY_PAD_TARGET_SEC(20с) вокруг центра,
        # поэтому сигналы должны быть существенно дальше друг от друга, чем 20с, чтобы не слиться
        hooks = [HookSpan(start=0.0, end=2.0, reason="A", kind="hook")]
        energy = [EnergySpan(start=35.0, end=36.0, score=2.0)]
        candidates = build_candidates(long_transcript, energy, hooks)
        assert len(candidates) == 2

    def test_candidates_sorted_by_score_descending(self, fake_transcript):
        hooks = [
            HookSpan(start=0.0, end=2.0, reason="joke", kind="joke"),
            HookSpan(start=8.0, end=10.0, reason="thesis", kind="thesis"),
        ]
        candidates = build_candidates(fake_transcript, [], hooks)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_clip_duration_clamped_to_min(self, long_transcript):
        hooks = [HookSpan(start=4.0, end=4.5, reason="короткий", kind="hook")]
        candidates = build_candidates(long_transcript, [], hooks)
        assert candidates[0].end - candidates[0].start >= config.CLIP_MIN_SEC - 0.01

    def test_clip_duration_clamped_to_max(self, fake_transcript):
        # растягиваем транскрипт искусственно длинным хуком
        hooks = [HookSpan(start=0.0, end=500.0, reason="очень длинный", kind="hook")]
        candidates = build_candidates(fake_transcript, [], hooks)
        assert candidates[0].end - candidates[0].start <= config.CLIP_MAX_SEC + 0.01

    def test_empty_inputs_produce_no_candidates(self, fake_transcript):
        assert build_candidates(fake_transcript, [], []) == []


class TestManualCandidate:
    def test_manual_candidate_has_max_score(self, fake_transcript):
        c = make_manual_candidate(4.0, 8.0, fake_transcript)
        assert c.score == 999.0
        assert c.source == "manual"

    def test_manual_candidate_appears_first_after_merge(self, fake_transcript):
        hooks = [HookSpan(start=0.0, end=2.0, reason="hook", kind="hook")]
        candidates = build_candidates(fake_transcript, [], hooks, manual=[(4.0, 8.0)])
        assert candidates[0].source == "manual"

    def test_manual_candidate_excerpt_matches_window(self, long_transcript):
        c = make_manual_candidate(4.0, 8.0, long_transcript)
        assert "второй сегмент" in c.transcript_excerpt
        assert "первый сегмент" not in c.transcript_excerpt
