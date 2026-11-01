"""Optuna hyper-parameter search for EMGNet (validation participants only; resumable)."""

from __future__ import annotations

import argparse

from myo_emg.training.hpo import run_hpo
from myo_emg.utils.config import add_config_args, artifacts_dir, config_from_args
from myo_emg.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--n-trials", type=int, default=None, help="trials to run now (default: hpo.n_trials)")
    parser.add_argument("--timeout-h", type=float, default=None, help="stop after this many hours")
    args = parser.parse_args()
    config = config_from_args(args)
    setup_logging(args.log_level, artifacts_dir(config) / "logs" / "hpo.log")
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # our own per-epoch logs are enough
    run_hpo(config, args.n_trials, args.timeout_h)


if __name__ == "__main__":
    main()
