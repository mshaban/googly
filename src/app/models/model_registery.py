from typing import Any, Optional


# Define a generic class for model registry
class ModelRegistry:
    models: dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def register_model(cls, model_name: str, *args) -> None:
        cls.models[model_name] = args

    @classmethod
    def get_model(cls, model_name: str):
        return cls.models.get(model_name, None)
