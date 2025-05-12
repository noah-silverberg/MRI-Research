"""
main.py

Entry‑point script for the MRI reconstruction workflow.
Reads a YAML configuration, builds a `MRIPipeline` object, and executes the
pipeline.  Designed so that advanced users can subclass `MRIPipeline` or
replace particular steps via dependency injection.
"""

from pathlib import Path
import argparse
from typing import Any

from mri_pipeline.config import load_config
from mri_pipeline.pipeline import MRIPipeline


def _cli() -> Any:
    """Parse CLI arguments (config path, optional overrides)."""
    parser = argparse.ArgumentParser(description="Run MRI reconstruction pipeline")
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="Path to YAML config"
    )
    # Future: allow `--set section.key=value` style overrides
    return parser.parse_args()


def main() -> None:
    """Top‑level entrypoint executed when running `python -m main`."""
    args = _cli()
    cfg = load_config(args.config)
    pipeline = MRIPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
