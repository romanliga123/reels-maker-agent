from reels_agent.pipeline.face_track import compute_crop_plan
from reels_agent import config


def test_crop_centered_horizontally():
    plan = compute_crop_plan(1280, 720)
    expected_width = round(720 * config.OUTPUT_WIDTH / config.OUTPUT_HEIGHT)
    assert plan.width == expected_width
    assert plan.height == 720
    center_x = plan.x + plan.width / 2
    assert abs(center_x - 1280 / 2) < 1


def test_crop_window_stays_within_frame_bounds():
    plan = compute_crop_plan(1280, 720)
    assert plan.x >= 0
    assert plan.x + plan.width <= 1280


def test_crop_width_clamped_to_source_width_when_narrower_than_target_ratio():
    plan = compute_crop_plan(400, 720)
    assert plan.width == 400
    assert plan.x == 0
