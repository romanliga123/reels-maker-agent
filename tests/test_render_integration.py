import subprocess

import pytest

from reels_agent.models import ClipCandidate
from reels_agent.pipeline.face_track import compute_crop_plan
from reels_agent.pipeline.render import render_clip, extract_clip_segment
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


@pytest.fixture(scope="session")
def long_synth_video(tmp_path_factory):
    """30-секундное видео — достаточно длинное, чтобы вырезать сегмент из СЕРЕДИНЫ
    и проверить, что abs/local пересчёт времени в extract_clip_segment верный."""
    out_dir = tmp_path_factory.mktemp("fixtures_long")
    path = out_dir / "long_synth.mp4"
    cmd = [
        config.FFMPEG_BIN, "-y",
        "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=30",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=30",
        "-shortest", "-c:v", "libx264", "-g", "12", "-c:a", "aac", str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=60)
    return path


def test_extract_clip_segment_then_render_uses_correct_local_offset(
    long_synth_video, fake_transcript, tmp_path,
):
    """Воспроизводит реальный путь job_loop._run_render_pipeline: сначала вырезаем
    локальный сегмент вокруг клипа из середины длинного видео, потом считаем
    crop/рендерим клип на этом сегменте с cut_start = candidate.start - seg_start.
    Если -avoid_negative_ts make_zero и пересчёт сдвига сломаны — длительность
    итогового клипа или его наличие пострадают."""
    candidate = ClipCandidate(
        id="segtest", start=15.0, end=19.0, reason="test", score=1.0,
        source="manual", subtitle_style="dynamic",
    )
    segment_path = tmp_path / "segment.mp4"
    seg_start = extract_clip_segment(long_synth_video, candidate.start, candidate.end, segment_path, pad_sec=5.0)
    assert segment_path.exists()
    assert seg_start == pytest.approx(10.0)

    local_start = candidate.start - seg_start
    local_end = candidate.end - seg_start
    crop = compute_crop_plan(segment_path, local_start, local_end, 1280, 720)

    output_path = tmp_path / "out_segment.mp4"
    result = render_clip(
        segment_path, candidate, fake_transcript, crop, tmp_path, output_path,
        cut_start=local_start,
    )

    assert result.error is None
    assert output_path.exists()
    assert abs(result.duration - (candidate.end - candidate.start)) < 0.01

    probe = subprocess.run(
        [config.FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", str(output_path)],
        capture_output=True, text=True, timeout=30,
    )
    actual_duration = float(probe.stdout.strip().split("=")[1])
    assert abs(actual_duration - (candidate.end - candidate.start)) < 0.3


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
