import numpy as np
import pytest

from bike_sim.baselines import BASELINES
from bike_sim.env import BikeWindEnv
from bike_sim.evaluate import rollout


def test_reset_and_step_shapes():
    env = BikeWindEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(r)
    assert not term and not trunc
    assert info["t"] == pytest.approx(env.p.dt_control)


def test_action_round_trip():
    u_p, u_a = BikeWindEnv.decode_action(BikeWindEnv.encode_action(0.3, 0.8))
    assert u_p == pytest.approx(0.3) and u_a == pytest.approx(0.8)


def test_posture_is_rate_limited():
    env = BikeWindEnv()
    env.reset()
    env.step(BikeWindEnv.encode_action(0.5, 1.0))
    assert env.state.posture == pytest.approx(env.p.posture_rate * env.p.dt_control)


def test_return_equals_minus_time_offset():
    env = BikeWindEnv(course_length=500.0)
    res = rollout(env, lambda e, o: BASELINES["upright_cp"](e))
    assert res["finished"]
    expected = (env.course_length - env.v_ref * res["time"]) / (env.v_ref * env.p.dt_control)
    assert res["return"] == pytest.approx(expected, rel=1e-6)


def test_fast_finisher_has_positive_return():
    env = BikeWindEnv()
    res = rollout(env, lambda e, o: BASELINES["myopic"](e))
    assert res["finished"]
    assert res["mean_speed"] > env.v_ref
    assert res["return"] > 0


def test_zero_power_truncates():
    env = BikeWindEnv(course_length=2000.0)
    res = rollout(env, lambda e, o: BikeWindEnv.encode_action(0.0, 0.0))
    assert not res["finished"]
    assert res["time"] >= env.max_time
