"""Rider posture and physiology.

Posture u_a in [0, 1] runs from upright (0) to full tuck (1). Drag savings
saturate (the first part of the tuck saves the most CdA) while the power
penalty grows quadratically. With linear maps for both, steady speed would be
a monotone function of posture and the optimum would always be an endpoint.
"""

import numpy as np

from .config import DEFAULT, Params


def cda(u_a: float, p: Params = DEFAULT) -> float:
    gain = 2.0 * u_a - u_a**2
    return p.cda_upright - (p.cda_upright - p.cda_tuck) * gain


def p_cap(u_a: float, p: Params = DEFAULT) -> float:
    return p.p_peak * (1.0 - p.alpha * u_a**2)


def critical_power(u_a: float, p: Params = DEFAULT) -> float:
    return p.cp0 * (1.0 - p.beta * u_a**2)


def deliverable_power(u_p: float, u_a: float, w_bal: float, p: Params = DEFAULT) -> float:
    """Commanded power, clamped to critical power once W' is exhausted."""
    power = float(np.clip(u_p, 0.0, 1.0)) * p_cap(u_a, p)
    if w_bal <= 0.0:
        power = min(power, critical_power(u_a, p))
    return power


def update_w_bal(w_bal: float, power: float, u_a: float, dt: float, p: Params = DEFAULT) -> float:
    """Differential W' balance (Skiba): linear depletion above CP, recovery
    proportional to the deficit below CP."""
    cp = critical_power(u_a, p)
    if power > cp:
        w_bal -= (power - cp) * dt
    else:
        w_bal += (cp - power) * (p.w_prime_max - w_bal) / p.w_prime_max * dt
    return float(np.clip(w_bal, 0.0, p.w_prime_max))
