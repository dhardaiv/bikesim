"""Roll out the trained SAC policy and baselines on the same course and compare."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .baselines import BASELINES
from .env import BikeWindEnv

RESULTS = Path(__file__).resolve().parents[2] / "results"


def rollout(env: BikeWindEnv, act) -> dict:
    """act(env, obs) -> action. Returns per-step traces and summary."""
    obs, _ = env.reset()
    trace = {k: [] for k in ("x", "t", "v", "posture", "mean_power", "w_bal", "headwind")}
    total_reward = 0.0
    while True:
        obs, r, term, trunc, info = env.step(act(env, obs))
        total_reward += r
        for k in trace:
            trace[k].append(info[k])
        if term or trunc:
            break
    trace = {k: np.array(v) for k, v in trace.items()}
    t_final = trace["t"][-1]
    return {
        "trace": trace,
        "finished": bool(info["finished"]),
        "time": t_final,
        "mean_speed": trace["x"][-1] / t_final,
        "return": total_reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RESULTS / "sac_bike.zip"))
    args = parser.parse_args()

    env = BikeWindEnv()
    policies = {name: (lambda e, o, f=f: f(e)) for name, f in BASELINES.items()}

    model_path = Path(args.model)
    if model_path.exists():
        from stable_baselines3 import SAC

        model = SAC.load(model_path, device="cpu")
        policies["sac"] = lambda e, o: model.predict(o, deterministic=True)[0]
    else:
        print(f"no model at {model_path}; evaluating baselines only")

    results = {name: rollout(env, act) for name, act in policies.items()}

    print(f"{'policy':<16}{'time [s]':>10}{'mean km/h':>11}{'return':>9}  finished")
    for name, r in results.items():
        print(f"{name:<16}{r['time']:>10.1f}{r['mean_speed']*3.6:>11.2f}{r['return']:>9.2f}  {r['finished']}")

    RESULTS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    first = next(iter(results.values()))["trace"]
    axes[0].plot(first["x"], first["headwind"], "k")
    axes[0].set_ylabel("headwind [m/s]")
    for name, r in results.items():
        tr = r["trace"]
        axes[1].plot(tr["x"], tr["v"] * 3.6, label=name)
        axes[2].plot(tr["x"], tr["mean_power"])
        axes[3].plot(tr["x"], tr["posture"])
    axes[1].set_ylabel("speed [km/h]")
    axes[1].legend(fontsize=8)
    axes[2].set_ylabel("power [W]")
    axes[3].set_ylabel("posture u_a")
    axes[3].set_xlabel("distance [m]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "eval_traces.png", dpi=130)
    print(f"saved {RESULTS / 'eval_traces.png'}")


if __name__ == "__main__":
    main()
