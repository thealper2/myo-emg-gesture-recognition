"""Parse EMG-EPN-612 into compact caches and create the participant-level split."""

from __future__ import annotations

import argparse

from myo_emg.data.prepare import prepare_dataset
from myo_emg.utils.config import add_config_args, config_from_args
from myo_emg.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-users", type=int, default=None, help="only the first N users per partition (smoke runs)")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config = config_from_args(args, ("configs/dataset.yaml",))
    prepare_dataset(config, workers=args.workers or int(config["prepare"]["workers"]), max_users=args.max_users)


if __name__ == "__main__":
    main()
