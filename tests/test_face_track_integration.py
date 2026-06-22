from reels_agent.pipeline.face_track import compute_crop_plan
from reels_agent import config


def test_no_face_falls_back_to_center_crop(synth_video):
    plan = compute_crop_plan(synth_video, 0, 4, 1280, 720)
    expected_width = round(720 * config.OUTPUT_WIDTH / config.OUTPUT_HEIGHT)
    assert plan.width == expected_width
    assert plan.height == 720
    center_x = plan.x + plan.width / 2
    assert abs(center_x - 1280 / 2) < 5  # центрировано по кадру при отсутствии лица


def test_crop_window_stays_within_frame_bounds(synth_video):
    plan = compute_crop_plan(synth_video, 0, 4, 1280, 720)
    assert plan.x >= 0
    assert plan.x + plan.width <= 1280
