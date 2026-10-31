import matplotlib

matplotlib.use("Agg")

import pytest

from bike_sim.env import BikeWindEnv
from bike_sim.visualize import FrameState, Visualizer, course_profile, rider_points


def test_tuck_lowers_torso_and_hands():
    up = rider_points(0.0, 0.0)
    tk = rider_points(1.0, 0.0)
    assert tk["shoulder"][1] < up["shoulder"][1]
    assert tk["head"][1] < up["head"][1]
    assert tk["hands"][1] < up["hands"][1]
    assert tk["hip"][0] > up["hip"][0]


def test_reward_matches_pace_formula():
    s = FrameState(t=400.0, x=4000.0, v=10.0, posture=0.5, power=250.0, w_bal=10_000.0, headwind=1.0, policy="test")
    assert s.episode_return == pytest.approx((4000.0 - 8.0 * 400.0) / 16.0)
    assert s.step_reward == pytest.approx((10.0 - 8.0) / 8.0)


def test_render_rgb_shape():
    env = BikeWindEnv()
    xs, hw = course_profile(env)
    viz = Visualizer(xs, hw)
    viz.update(FrameState(t=10.0, x=2500.0, v=8.0, posture=0.7, power=300.0, w_bal=12_000.0, headwind=7.0, policy="test"))
    rgb = viz.rgb()
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert rgb.shape[0] > 100 and rgb.shape[1] > 100
