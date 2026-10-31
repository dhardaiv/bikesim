"""Train SAC on the headwind-patch course."""

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from .env import BikeWindEnv

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(RESULTS / "sac_bike.zip"))
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    env = Monitor(BikeWindEnv())
    model = SAC(
        "MlpPolicy",
        env,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        learning_starts=2_000,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        device="cpu",
        verbose=1,
    )
    model.learn(total_timesteps=args.steps, log_interval=20)
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
