from src.app.core.enums import AppTags, RequestMethod

from src.app.models.endpoint import EndpointModel
from src.app.core.router import AppRouter
from src.app.core.app import GooglyApp
from src.app.core.settings import settings


e1 = EndpointModel(
    name="health",
    app_dir="src.app",
    endpoint_module="health",
    endpoint_func="health_check",
    endpoint_path="/health",
    method=RequestMethod.GET,
    tag=AppTags.MONITOR,
)


e2 = EndpointModel(
    name="googly",
    app_dir="src.app",
    endpoint_module="image",
    endpoint_func="googly",
    endpoint_path="/googly",
    method=RequestMethod.POST,
    tag=AppTags.SERVICE,
)

endpoints = [e1, e2]

router = AppRouter(endpoints)
app = GooglyApp(
    title=settings.PROJECT_NAME,
    routers=[router],
)
