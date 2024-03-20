from pydantic_settings import BaseSettings, SettingsConfigDict
from src.app.core.enums import LoggingLevelEnum, ModeEnum


class FastAPISettings(BaseSettings):
    """
    A class to define settings for FastAPI configuration.
    Attributes:
    - APP_NAME: str - the name of the FastAPI application (default: FastAPI)
    - HOST: str - the host for the FastAPI application (default: localhost)
    - PORT: int - the port for the FastAPI application (default: 8000)
    """

    APP_NAME: str = "FastAPI"
    HOST: str = "localhost"
    PORT: int = 8000
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


class RayServeSettings(BaseSettings):
    SERVE_URL: str = "http://localhost:8000"
    SERVE_ENDPOINT: str = "googly"


class Settings(BaseSettings):
    """
    A class representing the settings for the application.

    Attributes:
    - PROJECT_NAME (str): The name of the project.
    - MODE (ModeEnum): The mode in which the application is running (DEV, PROD, etc.).
    - API_VERSION (str): The version of the API.
    - FASTAPI (FastAPISettings): Settings related to the FastAPI framework.
    - LOGGER (LoggerSettings): Settings related to logging.
    - RAY_SERVE (RayServeSettings): Settings related to Ray Serve.

    Configurations:
    - env_file (str): The path to the .env file.
    - env_file_encoding (str): The encoding of the .env file.

    """

    PROJECT_NAME: str
    MODE: ModeEnum

    FASTAPI: FastAPISettings = FastAPISettings()
    LOGGER: LoggerSettings = LoggerSettings()
    RAY_SERVE: RayServeSettings = RayServeSettings()

    class Config:
        env_file: str = ".env"
        env_file_encoding: str = "utf-8"
        extra = "ignore"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_file = kwargs.get("_env_file", None)
        if self._env_file:
            print(f"Using environment file: {self._env_file}")
            kwargs["FASTAPI"] = FastAPISettings(_env_file=self._env_file)
            kwargs["LOGGER"] = LoggerSettings(_env_file=self._env_file)
            super().__init__(*args, **kwargs)


settings = Settings(_env_file="config/local.env")
