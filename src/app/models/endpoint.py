from importlib import import_module
from types import ModuleType
from typing import Callable, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from src.app.core.enums import AppTags, RequestMethod
from src.app.core.logger import logger


class EndpointModel(BaseModel):
    """
    Model class for defining an endpoint in an API application.

    Attributes:
    - name (str): Endpoint path
    - endpoint_prefix (str): Endpoint prefix
    - method (RequestMethod): HTTP methods
    - app_dir (str): Application directory
    - endpoint_module (str | ModuleType): Endpoint module
    - endpoint_func (str | Callable): Endpoint function
    - endpoint_path (str): Endpoint path
    - api_prefix (Optional[str]): API prefix
    - api_version (Optional[str]): API version
    - module_prefix (Optional[str]): Endpoint module prefix
    - tag (AppTags): Endpoint tag
    - model_config (ConfigDict): Configuration dictionary

    Methods:
    - validate_and_import_module(values): Validates and imports the module for the endpoint
    """

    # Required Fields
    name: str = Field(..., description="Endpoint path")
    endpoint_prefix: str = Field(default="/", description="Endpoint prefix")
    method: RequestMethod = Field(..., description="HTTP methods")
    app_dir: str = Field(..., description="Application directory")
    endpoint_module: str | ModuleType = Field(..., description="Endpoint module")
    endpoint_func: str | Callable = Field(..., description="Endpoint function")
    endpoint_path: str = Field(..., description="Endpoint path")
    router_config: dict = Field(default={}, description="Router configuration")

    # Optional Fields based on endpoint function path
    api_prefix: Optional[str] = Field(default="api", description="API prefix")
    api_version: Optional[str] = Field(default="v1", description="API version")
    module_prefix: Optional[str] = Field(
        default="endpoints", description="Endpoint module prefix"
    )

    # Extra endpoint args
    tag: AppTags = Field(default=None, description="Endpoint tag")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_and_import_module(self, values):
        """
        Validates and imports the module for the endpoint.

        Args:
        - values: Values to validate

        Returns:
        - self: The EndpointModel instance

        Raises:
        - ValueError: If the module or function could not be imported or found
        """

        logger.debug(f"Validating and importing module for endpoint {self.name}")
        if callable(self.endpoint_func):
            logger.debug(
                f"Endpoint function {self.endpoint_func.__name__} is already callable."
            )
            return self

        # Define required module fields
        module_fields = [
            self.app_dir,
            self.api_prefix,
            self.api_version,
            self.module_prefix,
            self.endpoint_module,
        ]

        module_path = ".".join(filter(None, [field for field in module_fields]))

        # Attempt to construct and import the module path
        try:
            self.endpoint_module = import_module(module_path)
            logger.debug(f"Module {module_path} imported successfully.")
        except ImportError:
            logger.error(f"Module {module_path} could not be imported.")
            raise ValueError(f"Module {module_path} could not be imported.")

        # Attempt to import the endpoint function
        try:
            self.endpoint_func = getattr(self.endpoint_module, self.endpoint_func)
            logger.debug(
                f"Function {self.endpoint_func.__name__} found in {module_path}."
            )
        except AttributeError:
            logger.error(
                f"Function {self.endpoint_func} could not be found in {module_path}."
            )
            raise ValueError(
                f"Function {self.endpoint_func} could not be found in {module_path}."
            )

        logger.debug(f"Endpoint {self.name} validated and imported successfully.")
        return self
