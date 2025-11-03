"""
Simple logging utility for reactive diffusion policy project.
"""

import datetime
import sys
from pathlib import Path

from loguru import logger


class LoggingManager:
    """
    A class-based logging manager that provides organized logging functionality
    for single-process and multi-process applications.
    """
    
    def __init__(self, log_dir: str = "logs", log_file_prefix: str = ""):
        """
        Initialize the logging manager.
        
        Args:
            log_dir: Directory for log files (default: "logs")
            log_file_prefix: Prefix for log file names (default: "")
        """
        self.log_dir = Path(log_dir)
        self.log_file_prefix = log_file_prefix
        self._console_format = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>[{extra[component]}]</cyan> - <level>{message}</level>"
        self._file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | [{extra[component]}] - {message}"
        
    def setup_logging(self):
        """
        Set up logging that captures all terminal output to a timestamped file.
        Creates a new log file, removing any existing one with the same name.
        """
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)

        # Remove existing handlers
        logger.remove()

        # Add console output (terminal)
        logger.add(
            sys.stderr,
            format=self._console_format,
            level="DEBUG",
        )

        # Add timestamped file output
        timestamp = ""
        log_file = self.log_dir / f"{self.log_file_prefix}{timestamp}.log"
        
        if log_file.exists() and log_file.is_file():
            log_file.unlink()
            
        logger.add(
            str(log_file),
            format=self._file_format,
            level="DEBUG",
        )

    def setup_multiprocess_logging(self):
        """
        Set up logging for child processes - appends to existing log file
        without deleting it. Thread-safe for multi-process environments.
        """
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)

        # Remove existing handlers
        logger.remove()

        # Add console output (terminal)
        logger.add(
            sys.stderr,
            format=self._console_format,
            level="DEBUG",
        )

        # Add file output in APPEND mode (don't delete existing file)
        log_files = sorted(self.log_dir.glob(f"{self.log_file_prefix}*.log"))
        if log_files:
            last_log_file = log_files[-1]
            log_file = self.log_dir / last_log_file.name
            
            logger.add(
                str(log_file),
                format=self._file_format,
                level="DEBUG",
                mode="a",  # Append mode
                enqueue=True,  # Thread-safe writing
            )

    def get_logger(self, component: str):
        """
        Get a logger bound to a specific component.
        
        Args:
            component: Component name for logging context
            
        Returns:
            Logger instance bound to the component
        """
        return logger.bind(component=component)

    def safe_shutdown_log(
        self,
        component: str,
        message: str,
        level: str = "INFO",
    ):
        """
        Log a final message before process shutdown that bypasses loguru queues.
        Guarantees the message reaches the log file even if background threads are killed.

        Args:
            component: Component name for logging
            message: Message to log
            level: Log level (default: "INFO")
        """
        log_files = sorted(self.log_dir.glob(f"{self.log_file_prefix}*.log"))
        if not log_files:
            return
            
        last_log_file = log_files[-1]
        log_file = self.log_dir / last_log_file.name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} | {level: <8} | [{component}] - {message}\n"
        
        try:
            with open(log_file, "a") as f:
                f.write(log_line)
                f.flush()
        except Exception:
            pass  # Fail silently if file write fails


# Convenience functions for backward compatibility
def setup_logging(log_dir: str = "logs", log_file_prefix: str = ""):
    """
    Single function to set up logging that captures all terminal output
    to a timestamped file.

    Args:
        log_dir: Directory for log files (default: "logs")
        log_file_prefix: Prefix for log file names (default: "")
    """
    logging_manager = LoggingManager(log_dir, log_file_prefix)
    logging_manager.setup_logging()


def setup_multiprocess_logging(log_dir: str = "logs", log_file_prefix: str = ""):
    """
    Set up logging for child processes - appends to existing log file
    without deleting it.

    Args:
        log_dir: Directory for log files (default: "logs")
        log_file_prefix: Prefix for log file names (default: "")
    """
    logging_manager = LoggingManager(log_dir, log_file_prefix)
    logging_manager.setup_multiprocess_logging()


def get_logger(component: str):
    """Get a logger for a component."""
    return logger.bind(component=component)


def safe_shutdown_log(
    component: str,
    message: str,
    level: str = "INFO",
    log_dir: str = "logs",
    log_file_prefix: str = "",
):
    """
    Log a final message before process shutdown that bypasses loguru queues.
    Guarantees the message reaches the log file even if background threads are killed.

    Args:
        component: Component name for logging
        message: Message to log
        level: Log level (default: "INFO")
        log_dir: Directory for log files (default: "logs")
        log_file_prefix: Prefix for log file names (default: "")
    """
    logging_manager = LoggingManager(log_dir, log_file_prefix)
    logging_manager.safe_shutdown_log(component, message, level)
