from fastapi import APIRouter

from src.app.models.endpoint import EndpointModel
from src.app.core.logger import logger


class AppRouter(APIRouter):
    def __init__(
        self,
        endpoints: list[EndpointModel],
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.endpoints = endpoints
        self.setup_endpoints()

    def setup_endpoints(
        self,
    ):
        for endpoint in self.endpoints:
            logger.debug(
                f"Setting up endpoint: {endpoint.name} at {endpoint.endpoint_path}"
            )
            self.add_api_route(
                endpoint.endpoint_path,
                endpoint.endpoint_func,
                methods=[endpoint.method.value],
                tags=[endpoint.tag],
                **endpoint.router_config,
            )
