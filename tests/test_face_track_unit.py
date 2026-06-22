from reels_agent.pipeline.face_track import _trim_outliers


def test_short_list_returned_unchanged():
    assert _trim_outliers([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_outliers_trimmed_from_both_ends():
    values = [100.0, 10.0, 11.0, 12.0, 13.0, 14.0, -50.0, 15.0, 16.0, 17.0]
    trimmed = _trim_outliers(values, trim_frac=0.1)
    assert 100.0 not in trimmed
    assert -50.0 not in trimmed


def test_trim_preserves_middle_values():
    values = [float(i) for i in range(20)]
    trimmed = _trim_outliers(values, trim_frac=0.1)
    assert 0.0 not in trimmed
    assert 19.0 not in trimmed
    assert 10.0 in trimmed
