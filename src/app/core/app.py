from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.core.router import AppRouter


class GooglyApp(FastAPI):

    def __init__(self, routers: list[AppRouter], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_middleware(
            CORSMiddleware,
            allow_credentials=True,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._routers_list = routers
        self._setup_routers()

    def _setup_routers(self):
        for rt in self._routers_list:
            self.include_router(rt)
        print("###############")
