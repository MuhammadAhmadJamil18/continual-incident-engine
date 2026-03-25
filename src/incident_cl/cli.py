from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .experiment import run_comparison, save_results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train naive vs replay continual incident classifier."
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/results.json"),
        help="Where to write JSON results for the dashboard.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eras", type=int, default=5)
    p.add_argument("--steps-per-era", type=int, default=120)
    p.add_argument("--replay-capacity", type=int, default=500)
    args = p.parse_args()

    cfg = ExperimentConfig(
        seed=args.seed,
        num_eras=args.eras,
        steps_per_era=args.steps_per_era,
        replay_capacity=args.replay_capacity,
    )
    payload = run_comparison(cfg)
    save_results(payload, args.output)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
