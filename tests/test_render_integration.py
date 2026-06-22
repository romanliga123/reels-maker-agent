import subprocess

from reels_agent.models import ClipCandidate
from reels_agent.pipeline.face_track import compute_crop_plan
from reels_agent.pipeline.render import render_clip
from reels_agent import config


def test_render_clip_produces_valid_vertical_mp4(synth_video, fake_transcript, tmp_path):
    candidate = ClipCandidate(
        id="rendertest", start=0.0, end=4.0, reason="test", score=1.0,
        source="manual", subtitle_style="dynamic",
    )
    crop = compute_crop_plan(synth_video, candidate.start, candidate.end, 1280, 720)
    output_path = tmp_path / "out.mp4"

    result = render_clip(synth_video, candidate, fake_transcript, crop, tmp_path, output_path)

    assert result.error is None
    assert output_path.exists()
    assert abs(result.duration - 4.0) < 0.01

    probe = subprocess.run(
        [config.FFPROBE_BIN, "-v", "error", "-show_entries", "stream=width,height,duration",
         "-of", "default=noprint_wrappers=1", str(output_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert f"width={config.OUTPUT_WIDTH}" in probe.stdout
    assert f"height={config.OUTPUT_HEIGHT}" in probe.stdout


def test_render_clip_with_static_subtitles(synth_video, fake_transcript, tmp_path):
    candidate = ClipCandidate(
        id="rendertest2", start=0.0, end=4.0, reason="test", score=1.0,
        source="manual", subtitle_style="static",
    )
    crop = compute_crop_plan(synth_video, candidate.start, candidate.end, 1280, 720)
    output_path = tmp_path / "out_static.mp4"

    result = render_clip(synth_video, candidate, fake_transcript, crop, tmp_path, output_path)
    assert result.error is None
    assert output_path.exists()


def test_render_clip_writes_ass_file_to_work_dir(synth_video, fake_transcript, tmp_path):
    candidate = ClipCandidate(
        id="abc123", start=0.0, end=4.0, reason="t", score=1.0, source="manual",
    )
    crop = compute_crop_plan(synth_video, candidate.start, candidate.end, 1280, 720)
    render_clip(synth_video, candidate, fake_transcript, crop, tmp_path, tmp_path / "out.mp4")
    assert (tmp_path / "abc123.ass").exists()


def test_render_clip_reports_error_for_invalid_source(fake_transcript, tmp_path):
    from reels_agent.pipeline.face_track import CropPlan
    candidate = ClipCandidate(
        id="badsrc", start=0.0, end=4.0, reason="t", score=1.0, source="manual",
    )
    crop = CropPlan(x=0, y=0, width=405, height=720)
    result = render_clip(tmp_path / "missing.mp4", candidate, fake_transcript, crop,
                          tmp_path, tmp_path / "out.mp4")
    assert result.error is not None
    assert not (tmp_path / "out.mp4").exists() or result.error
