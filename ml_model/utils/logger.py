"""
Logging Configuration
======================

Provides a consistent logging setup across the entire ml_model package.
Logs go to both the console (with colour-coded levels) and a rotating
file in ``outputs/logs/``.

Usage::

    from ml_model.utils import setup_logger, get_logger

    setup_logger()  # call once at startup
    logger = get_logger(__name__)
    logger.info("Training started")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


# ANSI colour codes for console output
_COLOURS = {
    "DEBUG": "\033[36m",      # cyan
    "INFO": "\033[32m",       # green
    "WARNING": "\033[33m",    # yellow
    "ERROR": "\033[31m",      # red
    "CRITICAL": "\033[1;31m", # bold red
    "RESET": "\033[0m",
}


class _ColourFormatter(logging.Formatter):
    """Log formatter that adds ANSI colours to level names on TTY outputs.

    Falls back to plain text when the output is not a terminal (e.g.
    when redirected to a file or piped).
    """

    def __init__(self, fmt: str, datefmt: Optional[str] = None) -> None:
        """Store the base format string."""
        super().__init__(fmt, datefmt)
        self._is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Apply colour to the level name if outputting to a terminal.

        Args:
            record: Log record to format.

        Returns:
            Formatted log string.
        """
        if self._is_tty:
            colour = _COLOURS.get(record.levelname, "")
            reset = _COLOURS["RESET"]
            record.levelname = f"{colour}{record.levelname}{reset}"
        return super().format(record)


def setup_logger(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_filename: str = "training.log",
) -> None:
    """Configure the root logger with console and file handlers.

    Safe to call multiple times — clears existing handlers first.

    Args:
        level: Minimum log level (e.g. ``logging.INFO``).
        log_dir: Directory for the log file.  Defaults to
            ``ml_model/outputs/logs/``.
        log_filename: Name of the log file.
    """
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_fmt = _ColourFormatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # --- File handler ---
    log_path = log_dir / log_filename
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info("Logging initialised → %s", log_path)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger (convenience wrapper).

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        Configured :class:`logging.Logger`.
    """
    return logging.getLogger(name)
