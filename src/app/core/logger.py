from loguru import logger
from src.app.core.settings import settings
from datetime import datetime
import sys


import sys
from datetime import datetime


def configure_logger():
    """
    Configures the logger with specified settings.

    This function configures the logger with settings specified in the settings
    module. It adds a handler to log messages to the console with colorized
    output and a specified log level and format. It also adds a handler to log
    messages to a file with colorized output, a specified log level and format,
    rotation, retention, and compression settings.

    Parameters:
    None

    Returns:
    None

    Raises:
    None
    """
    logger.remove()

    logger.add(
        sys.stdout,
        colorize=True,
        level=settings.LOGGER.LOG_LEVEL.value,
        format=settings.LOGGER.LOG_FORMAT,
    )

    log_file_path = settings.LOGGER.LOG_FILE.split("/")[:-1]
    log_file_name = settings.LOGGER.LOG_FILE.split("/")[-1].split(".")[0]
    log_file = "/".join(
        log_file_path
    ) + f"/{log_file_name}/{datetime.now()}.log".replace(" ", "_")
    print(log_file)

    logger.add(
        log_file,
        colorize=True,
        level=settings.LOGGER.LOG_LEVEL.value,
        format=settings.LOGGER.LOG_FORMAT,
        rotation=settings.LOGGER.ROTATION,
        retention=settings.LOGGER.RETENTION,
        compression=settings.LOGGER.COMPRESSION,
    )

    logger.debug("Logger configured with settings")


configure_logger()
