"""
Verbose logging configuration for Cyber-Inference.

Provides comprehensive logging with:
- Console output with rich formatting
- File logging with rotation for persistence
- Colored output for different log levels
- Timestamp and source tracking
- Automatic log rotation to prevent unlimited growth
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom cyberpunk theme for console output
CYBER_THEME = Theme({
    "info": "bright_green",
    "warning": "bright_yellow",
    "error": "bright_red",
    "debug": "cyan",
    "success": "bold bright_green",
    "highlight": "bold magenta",
    "timestamp": "dim cyan",
    "module": "bright_blue",
})

console = Console(theme=CYBER_THEME)

# Global log level - can be adjusted via config
_log_level = logging.DEBUG

# Log directory
_log_dir: Optional[Path] = None


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.DEBUG) -> None:
    """
    Initialize the logging system with verbose console and file output.

    Args:
        log_dir: Directory for log files. If None, only console logging is enabled.
        level: Logging level (default: DEBUG for verbose output)
    """
    global _log_level, _log_dir
    _log_level = level
    _log_dir = log_dir

    # Create log directory if specified
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Rich console handler with verbose format
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        show_level=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
        log_time_format="[%Y-%m-%d %H:%M:%S.%f]",
    )
    rich_handler.setLevel(level)
    rich_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
    root_logger.addHandler(rich_handler)

    # File handler with rotation if log directory is specified
    if log_dir:
        log_file = log_dir / "cyber-inference.log"

        # RotatingFileHandler: max 10MB per file, keep 5 backup files (50MB total max)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        # File logging at INFO level to reduce noise (DEBUG only to console)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        root_logger.addHandler(file_handler)

        # Log startup info
        root_logger.info(f"[highlight]Log file initialized:[/highlight] {log_file} (rotating, max 10MB x 5)")

    root_logger.info("[success]Cyber-Inference logging system initialized[/success]")
    root_logger.debug(f"Log level set to: {logging.getLevelName(level)}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Ensure basic setup if not already done
    if not logging.getLogger().handlers:
        setup_logging(level=_log_level)

    return logger


class LogContext:
    """Context manager for logging operation blocks with timing."""

    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time: Optional[datetime] = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"[highlight]Starting:[/highlight] {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = datetime.now() - self.start_time
        elapsed_ms = elapsed.total_seconds() * 1000

        if exc_type is None:
            self.logger.log(
                self.level,
                f"[success]Completed:[/success] {self.operation} ({elapsed_ms:.2f}ms)"
            )
        else:
            self.logger.error(
                f"[error]Failed:[/error] {self.operation} ({elapsed_ms:.2f}ms) - {exc_val}"
            )
        return False


def log_startup_banner() -> None:
    """Display the Cyber-Inference startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗██╗   ██╗██████╗ ███████╗██████╗       ██╗███╗   ██╗███████╗        ║
║  ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗      ██║████╗  ██║██╔════╝        ║
║  ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝█████╗██║██╔██╗ ██║█████╗          ║
║  ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗╚════╝██║██║╚██╗██║██╔══╝          ║
║  ╚██████╗   ██║   ██████╔╝███████╗██║  ██║      ██║██║ ╚████║██║             ║
║   ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝      ╚═╝╚═╝  ╚═══╝╚═╝             ║
║                                                                               ║
║                    Edge Inference Server Management                           ║
║                         v0.1.0 | GPLv3 License                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bright_green")

