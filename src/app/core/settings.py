from pydantic_settings import BaseSettings, SettingsConfigDict

from src.app.core.enums import LoggingLevelEnum, ModeEnum


class FastAPISettings(BaseSettings):
    """
    A class to store settings for a FastAPI application.

    Attributes:
    - APP_NAME (str): The name of the FastAPI application.
    - FAST_HOST (str): The host address for the FastAPI application.
    - FAST_PORT (int): The port number for the FastAPI application.
    - API_VERSION (str): The version of the API.
    - model_config (SettingsConfigDict): A dictionary to store model configuration settings.
    """

    APP_NAME: str = "FastAPI"
    FAST_HOST: str = "localhost"
    FAST_PORT: int = 8888
    FAST_ENDPOINT: str = "googly"
    API_VERSION: str = "v1"

    model_config = SettingsConfigDict(extra="ignore")


class LoggerSettings(BaseSettings):
    """
    A class to define settings for logging configuration.

    Attributes:
    - LOG_LEVEL: LoggingLevelEnum - the log level to be used (default: DEBUG)
    - LOG_FILE: str - the file path for the log file (default: logs/app.log)
    - LOG_FORMAT: str - the format for log messages (default: specified format)
    - ROTATION: str - the rotation interval for log files (default: 1 week)
    - RETENTION: str - the retention period for log files (default: 1 month)
    - COMPRESSION: str - the compression method for log files (default: zip)
    """

    LOG_LEVEL: LoggingLevelEnum = LoggingLevelEnum.DEBUG
    LOG_FILE: str = "logs/app.log"
    LOG_FORMAT: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    ROTATION: str = "1 week"
    RETENTION: str = "1 month"
    COMPRESSION: str = "zip"

    model_config = SettingsConfigDict(extra="ignore")


class Settings(BaseSettings):
    """
    A class to store settings for a project.

    Attributes:
    - PROJECT_NAME (str): The name of the project.
    - MODE (ModeEnum): The mode of the project.
    - FASTAPI (FastAPISettings): Settings for FastAPI.
    - LOGGER (LoggerSettings): Settings for the logger.
    - SERVE_HOST (str): The host for serving the project.
    - SERVE_ENDPOINT (str): The endpoint for serving the project.
    - SERVE_PORT (int): The port for serving the project.
    - GOOGLY_PATH (str): The path to the googly image.

    Configurations:
    - env_file (str): The path to the environment file.
    - env_file_encoding (str): The encoding of the environment file.
    - extra (str): How to handle extra fields in the environment file.

    Methods:
    - __init__: Initializes the Settings object with optional environment file.

    Usage:
    settings = Settings()
    """

    PROJECT_NAME: str
    MODE: ModeEnum

    FASTAPI: FastAPISettings = FastAPISettings()
    LOGGER: LoggerSettings = LoggerSettings()

    SERVE_HOST: str = "http://localhost"
    SERVE_ENDPOINT: str = "googly_serve"
    SERVE_PORT: int = 8000

    GOOGLY_PATH: str = "assets/googly1.png"

    class Config:
        env_file: str = ".env"
        env_file_encoding: str = "utf-8"
        extra = "ignore"

    def __init__(self, *args, **kwargs):
        """
        Initializes the Settings object.

        Args:
        - _env_file (str): Optional path to an environment file.

        Returns:
        - None

        Raises:
        - None
        """
        super().__init__(*args, **kwargs)
        self._env_file = kwargs.get("_env_file", None)
        if self._env_file:
            print(f"Using environment file: {self._env_file}")
            kwargs["FASTAPI"] = FastAPISettings(_env_file=self._env_file)
            kwargs["LOGGER"] = LoggerSettings(_env_file=self._env_file)

            super().__init__(*args, **kwargs)


settings = Settings(_env_file="config/local.env")
