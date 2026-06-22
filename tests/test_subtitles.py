from reels_agent.pipeline.subtitles import _fmt_time, build_static_ass, build_karaoke_ass, build_ass


class TestFmtTime:
    def test_zero(self):
        assert _fmt_time(0) == "0:00:00.00"

    def test_simple(self):
        assert _fmt_time(61.5) == "0:01:01.50"

    def test_hours(self):
        assert _fmt_time(3661.2) == "1:01:01.20"

    def test_negative_clamped_to_zero(self):
        assert _fmt_time(-5) == "0:00:00.00"

    def test_carry_seconds_into_minutes(self):
        # 59.999 округляется к 60.00с — должно перенестись в минуты, а не остаться "60"
        assert _fmt_time(59.999) == "0:01:00.00"

    def test_carry_minutes_into_hours(self):
        assert _fmt_time(3599.999) == "1:00:00.00"


class TestStaticAss:
    def test_includes_only_segments_in_range(self, fake_transcript):
        ass = build_static_ass(fake_transcript, clip_start=4.0, clip_end=8.0)
        assert "первый сегмент" not in ass
        assert "второй сегмент" in ass
        assert "третий" not in ass

    def test_times_are_clip_relative(self, fake_transcript):
        ass = build_static_ass(fake_transcript, clip_start=4.0, clip_end=8.0)
        # сегмент [4.0-8.0] в исходном транскрипте должен стать [0.00-4.00] относительно клипа
        assert "Dialogue: 0,0:00:00.00,0:00:04.00" in ass

    def test_empty_when_no_segments_overlap(self, fake_transcript):
        ass = build_static_ass(fake_transcript, clip_start=100.0, clip_end=110.0)
        assert "Dialogue:" not in ass
        assert "[Events]" in ass  # заголовок всё равно присутствует


class TestKaraokeAss:
    def test_has_k_tags(self, fake_transcript):
        ass = build_karaoke_ass(fake_transcript, clip_start=0.0, clip_end=12.0)
        assert "\\k" in ass

    def test_k_durations_are_positive_ints(self, fake_transcript):
        import re
        ass = build_karaoke_ass(fake_transcript, clip_start=0.0, clip_end=12.0)
        durations = [int(d) for d in re.findall(r"\\k(\d+)", ass)]
        assert durations, "должен быть хотя бы один \\k тег"
        assert all(d >= 1 for d in durations)

    def test_falls_back_to_plain_text_when_no_words(self):
        from reels_agent.models import TranscriptSegment
        segs = [TranscriptSegment(text="Без слов вообще.", start=0.0, end=2.0, words=[])]
        ass = build_karaoke_ass(segs, clip_start=0.0, clip_end=2.0)
        assert "Без слов вообще." in ass
        assert "\\k" not in ass


class TestBuildAssDispatch:
    def test_dynamic_style_uses_karaoke(self, fake_transcript):
        assert "\\k" in build_ass(fake_transcript, 0.0, 12.0, style="dynamic")

    def test_static_style_has_no_karaoke_tags(self, fake_transcript):
        assert "\\k" not in build_ass(fake_transcript, 0.0, 12.0, style="static")
