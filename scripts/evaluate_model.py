"""Deploy the trained EMGNet and evaluate it once on the held-out test participants."""

from __future__ import annotations

import argparse

from myo_emg.training.evaluate import evaluate_test
from myo_emg.utils.config import add_config_args, artifacts_dir, config_from_args
from myo_emg.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--force", action="store_true", help="repeat the test evaluation of the same model")
    args = parser.parse_args()
    config = config_from_args(args)
    setup_logging(args.log_level, artifacts_dir(config) / "logs" / "evaluate.log")
    evaluate_test(config, force=args.force)


if __name__ == "__main__":
    main()
