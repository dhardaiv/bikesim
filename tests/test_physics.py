import numpy as np
import pytest

from bike_sim import rider
from bike_sim.aero import aero
from bike_sim.config import DEFAULT
from bike_sim.dynamics import forces
from bike_sim.steady import equilibrium_speed, optimal_posture


def test_drag_opposes_motion_in_still_air():
    assert aero(10.0, 0.0, 0.0, 0.3).drag_x < 0.0


def test_headwind_increases_drag():
    still = aero(10.0, 0.0, 0.0, 0.3).drag_x
    head = aero(10.0, -5.0, 0.0, 0.3).drag_x
    assert head < still < 0.0


def test_tailwind_faster_than_rider_pushes():
    a = aero(5.0, 8.0, 0.0, 0.3)
    assert a.v_rel < 0.0
    assert a.drag_x > 0.0


def test_drag_magnitude_matches_dynamic_pressure():
    a = aero(10.0, -2.0, 0.0, 0.3)
    assert a.airspeed == pytest.approx(12.0)
    assert a.q == pytest.approx(0.5 * DEFAULT.rho * 144.0)
    assert -a.drag_x == pytest.approx(a.q * 0.3)
    assert a.pressure_drag + a.friction_drag == pytest.approx(-a.drag_x)


def test_crosswind_adds_yaw_and_drag():
    no_cross = aero(10.0, 0.0, 0.0, 0.3)
    cross = aero(10.0, 0.0, 4.0, 0.3)
    assert cross.yaw > 0.0
    assert cross.cda > no_cross.cda
    assert cross.drag_x < no_cross.drag_x


def test_reynolds_in_bluff_body_range():
    re = aero(10.0, 0.0, 0.0, 0.3).reynolds
    assert 1e5 < re < 1e6


def test_power_balance_root_has_zero_net_force():
    for headwind in (-4.0, 0.0, 6.0):
        v = equilibrium_speed(0.5, headwind)
        power = rider.critical_power(0.5)
        net = sum(forces(v, power, 0.5, (-headwind, 0.0)))
        assert abs(net) < 1e-6


def test_headwind_lowers_equilibrium_speed():
    assert equilibrium_speed(0.5, 8.0) < equilibrium_speed(0.5, 0.0) < equilibrium_speed(0.5, -4.0)


def test_steady_optimum_is_interior_and_deepens_with_headwind():
    u_tail, v_tail = optimal_posture(-6.0)
    u_head, v_head = optimal_posture(10.0)
    assert 0.0 < u_tail < 1.0 and 0.0 < u_head < 1.0
    assert u_head > u_tail
    for headwind, v_opt in ((-6.0, v_tail), (10.0, v_head)):
        assert v_opt >= equilibrium_speed(0.0, headwind) - 1e-9
        assert v_opt >= equilibrium_speed(1.0, headwind) - 1e-9


def test_posture_coupling():
    us = np.linspace(0.0, 1.0, 11)
    cdas = [rider.cda(u) for u in us]
    caps = [rider.p_cap(u) for u in us]
    cps = [rider.critical_power(u) for u in us]
    assert np.all(np.diff(cdas) < 0)
    assert np.all(np.diff(caps) < 0)
    assert np.all(np.diff(cps) < 0)
    assert cdas[0] == pytest.approx(DEFAULT.cda_upright)
    assert cdas[-1] == pytest.approx(DEFAULT.cda_tuck)


def test_w_bal_depletes_above_cp_and_recovers_below():
    cp = rider.critical_power(0.0)
    w = rider.update_w_bal(DEFAULT.w_prime_max, cp + 100.0, 0.0, 10.0)
    assert w == pytest.approx(DEFAULT.w_prime_max - 1000.0)
    w_rec = rider.update_w_bal(w, cp - 100.0, 0.0, 10.0)
    assert w < w_rec <= DEFAULT.w_prime_max


def test_power_clamped_to_cp_when_w_bal_empty():
    assert rider.deliverable_power(1.0, 0.3, 0.0) == pytest.approx(rider.critical_power(0.3))
    assert rider.deliverable_power(1.0, 0.3, 1000.0) == pytest.approx(rider.p_cap(0.3))
