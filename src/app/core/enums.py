from enum import Enum


class ModeEnum(str, Enum):
    """
    An enumeration class representing different modes of operation.

    Attributes:
    - DEV: Development mode
    - PROD: Production mode
    - TEST: Testing mode
    """

    DEV = "DEV"
    PROD = "PROD"
    TEST = "TEST"


class LoggingLevelEnum(Enum):
    """
    An enumeration class representing different logging levels.

    Attributes:
    DEBUG (str): The DEBUG logging level.
    INFO (str): The INFO logging level.
    WARNING (str): The WARNING logging level.
    ERROR (str): The ERROR logging level.
    CRITICAL (str): The CRITICAL logging level.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AppTags(Enum):
    MONITOR: str = "MONITOR"
    SERVICE: str = "SERVICE"
    TEST: str = "TEST"


class RequestMethod(Enum):
    GET: str = "GET"
    POST: str = "POST"
    PUT: str = "PUT"
    DELETE: str = "DELETE"
    PATCH: str = "PATCH"
    OPTIONS: str = "OPTIONS"
    HEAD: str = "HEAD"
