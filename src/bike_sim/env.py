"""Gymnasium environment: ride a fixed-length course through a wind field as fast as possible.

Actions (both in [-1, 1], mapped to [0, 1]):
    0: power fraction u_p of the posture-dependent peak power
    1: posture target u_a (0 upright, 1 full tuck), rate-limited

Reward per step is (dx - v_ref * dt) / (v_ref * dt_control). The undiscounted
return equals L / (v_ref * dt_control) - T / dt_control, so maximizing it
minimizes finishing time T, i.e. maximizes mean velocity L / T. A pure
progress reward would sum to L for every policy that finishes.

v_ref defaults to 8 m/s (28.8 km/h), just below the mean speed of a decent
rider on this course, so finishing policies that beat that pace collect a
positive return.
"""

from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import rider
from .aero import aero
from .config import DEFAULT, Params
from .dynamics import RiderState, flat, substep
from .wind import HeadwindPatch, WindField

PREVIEW_OFFSETS = (100.0, 300.0, 600.0)  # m ahead
V_SCALE = 15.0
WIND_SCALE = 10.0


class BikeWindEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        wind_field: WindField | None = None,
        course_length: float = 4000.0,
        v_ref: float = 8.0,
        v0: float = 5.0,
        max_time_factor: float = 3.0,
        grade_fn=flat,
        params: Params = DEFAULT,
    ):
        self.wind_field = wind_field if wind_field is not None else HeadwindPatch()
        self.course_length = course_length
        self.v_ref = v_ref
        self.v0 = v0
        self.max_time = max_time_factor * course_length / v_ref
        self.grade_fn = grade_fn
        self.p = params

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        n_obs = 6 + len(PREVIEW_OFFSETS)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        self.state = RiderState()

    @staticmethod
    def decode_action(action) -> tuple[float, float]:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return 0.5 * (a[0] + 1.0), 0.5 * (a[1] + 1.0)

    @staticmethod
    def encode_action(u_p: float, u_a: float) -> np.ndarray:
        return np.array([2.0 * u_p - 1.0, 2.0 * u_a - 1.0], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        s = self.state
        wx, wy = self.wind_field(s.x)
        preview = [self.wind_field(s.x + d)[0] / WIND_SCALE for d in PREVIEW_OFFSETS]
        return np.array(
            [
                s.x / self.course_length,
                s.v / V_SCALE,
                s.w_bal / self.p.w_prime_max,
                s.posture,
                wx / WIND_SCALE,
                wy / WIND_SCALE,
                *preview,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = RiderState(x=0.0, v=self.v0, w_bal=self.p.w_prime_max, posture=0.0, t=0.0)
        return self._obs(), {}

    def step(self, action):
        u_p, u_a_target = self.decode_action(action)
        s0 = self.state
        n_sub = int(round(self.p.dt_control / self.p.dt_sim))
        energy = 0.0
        finished = False
        s = s0
        for _ in range(n_sub):
            s_next, power = substep(s, u_p, u_a_target, self.wind_field, self.grade_fn, self.p)
            energy += power * self.p.dt_sim
            if s_next.x >= self.course_length:
                frac = (self.course_length - s.x) / max(s_next.x - s.x, 1e-9)
                s = replace(s_next, x=self.course_length, t=s.t + frac * self.p.dt_sim)
                finished = True
                break
            s = s_next
        self.state = s

        dx = s.x - s0.x
        dt = s.t - s0.t
        reward = (dx - self.v_ref * dt) / (self.v_ref * self.p.dt_control)
        truncated = (not finished) and s.t >= self.max_time

        wx, wy = self.wind_field(s.x)
        a = aero(s.v, wx, wy, rider.cda(s.posture, self.p), self.p)
        info = {
            "t": s.t,
            "x": s.x,
            "v": s.v,
            "w_bal": s.w_bal,
            "posture": s.posture,
            "mean_power": energy / max(dt, 1e-9),
            "headwind": -wx,
            "reynolds": a.reynolds,
            "drag": -a.drag_x,
            "finished": finished,
        }
        return self._obs(), float(reward), finished, truncated, info
