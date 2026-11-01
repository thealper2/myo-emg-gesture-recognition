"""Audit the local EMG-EPN-612 dataset (read-only) and write JSON/Markdown reports."""

from __future__ import annotations

import argparse

from myo_emg.data.inspect import run_audit
from myo_emg.utils.config import add_config_args, artifacts_dir, config_from_args
from myo_emg.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--workers", type=int, default=None, help="parallel parser processes")
    parser.add_argument("--max-users", type=int, default=None, help="audit only the first N users per partition")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config = config_from_args(args, ("configs/dataset.yaml",))
    workers = args.workers or int(config["prepare"]["workers"])
    run_audit(config, artifacts_dir(config) / "reports" / "dataset", workers=workers, max_users=args.max_users)


if __name__ == "__main__":
    main()
