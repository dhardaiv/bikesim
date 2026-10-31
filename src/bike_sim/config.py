from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # Environment / air
    rho: float = 1.2  # kg/m^3
    mu: float = 1.81e-5  # Pa s, dynamic viscosity of air
    g: float = 9.81

    # Rider + bike
    mass: float = 80.0  # kg
    crr: float = 0.004
    eta: float = 0.97  # drivetrain efficiency
    v_eps: float = 1.0  # m/s, floor on speed when converting power to force
    f_prop_max: float = 400.0  # N, traction / pedal-force limit at low speed

    # Aerodynamics
    cda_upright: float = 0.40  # m^2
    cda_tuck: float = 0.25  # m^2
    char_length: float = 1.0  # m, torso-scale length for Reynolds number
    k_yaw: float = 0.3  # CdA growth with sin^2(yaw)
    re_ref: float = 5e5
    cd_re_slope: float = 0.0  # optional Cd(Re) knob, fractional change per decade of Re
    skin_friction_fraction: float = 0.1  # share of drag from skin friction (reported only)

    # Physiology
    p_peak: float = 350.0  # W, max instantaneous power (upright)
    cp0: float = 250.0  # W, critical power (upright)
    alpha: float = 0.20  # peak-power loss at full tuck (quadratic in posture)
    beta: float = 0.20  # critical-power loss at full tuck (quadratic in posture)
    w_prime_max: float = 20_000.0  # J, anaerobic work capacity

    # Integration / control
    dt_sim: float = 0.1  # s, physics substep
    dt_control: float = 2.0  # s, policy step
    posture_rate: float = 0.5  # 1/s, max change in posture per second


DEFAULT = Params()
