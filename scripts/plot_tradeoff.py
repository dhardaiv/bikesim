"""Plot steady speed vs headwind for upright, full tuck, and the optimal posture."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bike_sim.steady import tradeoff_table

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    headwinds = np.linspace(-8.0, 12.0, 41)
    t = tradeoff_table(headwinds)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.plot(headwinds, t["v_upright"] * 3.6, label="upright (u_a=0)")
    ax1.plot(headwinds, t["v_tuck"] * 3.6, label="full tuck (u_a=1)")
    ax1.plot(headwinds, t["v_opt"] * 3.6, "k--", label="optimal mix")
    ax1.set_ylabel("steady speed at CP [km/h]")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(headwinds, t["u_opt"], "k")
    ax2.set_xlabel("headwind [m/s] (negative = tailwind)")
    ax2.set_ylabel("optimal posture u_a")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff.png", dpi=130)

    for w, uo, vu, vt, vo in zip(t["headwind"], t["u_opt"], t["v_upright"], t["v_tuck"], t["v_opt"]):
        if w in (-8.0, -4.0, 0.0, 4.0, 8.0, 12.0):
            print(f"headwind {w:+5.1f} m/s: u*={uo:.3f}  v_up={vu*3.6:5.2f}  v_tuck={vt*3.6:5.2f}  v*={vo*3.6:5.2f} km/h")
    print(f"saved {OUT / 'tradeoff.png'}")


if __name__ == "__main__":
    main()
