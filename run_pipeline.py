import sys
import argparse
import logging
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import AutoEITPipeline


def setup_logging(debug_mode: bool) -> logging.Logger:
    """
    Configure root logging for the CLI session.
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO

    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger("AutoEIT_CLI")


def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="AutoEIT Spanish Scoring Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the master YAML configuration file.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    return parser.parse_args()


def resolve_config_path(config_arg: str) -> Path:
    """
    Resolve config path safely for both relative and absolute input paths.
    """
    candidate = Path(config_arg)

    if candidate.is_absolute():
        return candidate.resolve()

    return (PROJECT_ROOT / candidate).resolve()


def main() -> int:
    """
    Main CLI execution flow with graceful error handling.
    """
    args = parse_args()
    logger = setup_logging(args.debug)

    config_path = resolve_config_path(args.config)

    if not config_path.exists():
        logger.critical(f"Configuration file not found: {config_path}")
        return 1

    if not config_path.is_file():
        logger.critical(f"Provided config path is not a file: {config_path}")
        return 1

    try:
        logger.info(f"Starting AutoEIT Pipeline with config: {config_path}")
        pipeline = AutoEITPipeline(config_path=str(config_path))
        pipeline.run_experiment()
        logger.info("Pipeline execution completed successfully.")
        return 0

    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user.")
        return 130

    except Exception as exc:
        logger.critical("A fatal error occurred during pipeline execution.")
        logger.error(str(exc))

        if args.debug:
            logger.debug("Full traceback follows:\n%s", traceback.format_exc())
        else:
            logger.info("Re-run with --debug to view the full traceback.")

        return 1


if __name__ == "__main__":
    sys.exit(main())