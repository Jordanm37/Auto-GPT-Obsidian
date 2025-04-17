"""Basic logging configuration for the Bidian agent."""

import logging
import json
import os
from typing import Dict, Any


class JsonFormatter(logging.Formatter):
    """Formats log records into JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            A JSON string representation of the log record.
        """
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)
        # Add any extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                # Avoid adding default fields or internal attributes
                if key not in {'name', 'msg', 'args', 'levelname', 'levelno',
                               'pathname', 'filename', 'module', 'exc_info',
                               'exc_text', 'stack_info', 'lineno', 'funcName',
                               'created', 'msecs', 'relativeCreated', 'thread',
                               'threadName', 'processName', 'process', 'message',
                               'asctime'}:
                    log_entry[key] = value

        return json.dumps(log_entry)


def setup_logging(log_level_str: str = "INFO") -> None:
    """Configures the root logger.

    Sets up a console handler with JSON formatting.

    Args:
        log_level_str: The minimum log level to output (e.g., "DEBUG", "INFO").
                       Defaults to "INFO". Reads from LOG_LEVEL env var if set.
    """
    log_level_env = os.environ.get("LOG_LEVEL", log_level_str).upper()
    log_level = getattr(logging, log_level_env, logging.INFO)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates if called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create formatter and add it to the handler
    formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z")
    console_handler.setFormatter(formatter)

    # Add the handler to the root logger
    root_logger.addHandler(console_handler)

    logging.info(f"Logging configured with level: {log_level_env}")

# Example usage (optional, can be removed or put under if __name__ == "__main__":)
# if __name__ == "__main__":
#     setup_logging(log_level_str="DEBUG")
#     logging.debug("This is a debug message.")
#     logging.info("This is an info message.")
#     logging.warning("This is a warning message.")
#     try:
#         1 / 0
#     except ZeroDivisionError:
#         logging.error("This is an error message with exception info.", exc_info=True)
#     logging.critical("This is a critical message.")
