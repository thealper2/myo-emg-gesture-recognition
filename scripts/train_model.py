"""Train the final EMGNet on all training participants with the best HPO parameters."""

from __future__ import annotations

import argparse

from myo_emg.training.train import refinalize, run
from myo_emg.utils.config import add_config_args, artifacts_dir, config_from_args
from myo_emg.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--no-hpo", action="store_true", help="ignore artifacts/hpo/best_params.yaml")
    parser.add_argument("--finalize-only", action="store_true",
                        help="no training: re-run the checks/reports on the saved final_model")
    args = parser.parse_args()
    config = config_from_args(args)
    setup_logging(args.log_level, artifacts_dir(config) / "logs" / "train.log")
    if args.finalize_only:
        refinalize(config)
    else:
        run(config, use_hpo=not args.no_hpo, overrides=args.overrides)


if __name__ == "__main__":
    main()
