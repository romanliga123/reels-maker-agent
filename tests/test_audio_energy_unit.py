import numpy as np

from reels_agent.pipeline.audio_energy import _zscore, _smooth


class TestZscore:
    def test_zero_std_returns_zeros(self):
        x = np.array([5.0, 5.0, 5.0])
        result = _zscore(x)
        assert np.allclose(result, [0.0, 0.0, 0.0])

    def test_normalizes_to_zero_mean_unit_std(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _zscore(x)
        assert abs(result.mean()) < 1e-9
        assert abs(result.std() - 1.0) < 1e-9

    def test_peak_has_highest_zscore(self):
        x = np.array([1.0, 1.0, 10.0, 1.0, 1.0])
        result = _zscore(x)
        assert np.argmax(result) == 2


class TestSmooth:
    def test_window_one_is_noop(self):
        x = np.array([1.0, 5.0, 2.0])
        assert np.allclose(_smooth(x, 1), x)

    def test_smoothing_reduces_single_spike(self):
        x = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        smoothed = _smooth(x, 3)
        assert smoothed.max() < x.max()
        assert len(smoothed) == len(x)
