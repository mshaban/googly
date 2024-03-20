from typing import Any, Optional
from pathlib import Path
import json
from openvino.runtime import Core


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


class ModelArtifacts:
    def __init__(self, config_path: str):
        self.model_artifacts = self.load_config(config_path)

    def load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as config_file:
            return json.load(config_file)

    def setup_artifacts(self) -> tuple:
        ie = Core()
        model_path = Path(self.model_artifacts["path"])
        precision = self.model_artifacts["precision"]
        xml_path = model_path / precision / self.model_artifacts["xml"]
        bin_path = model_path / precision / self.model_artifacts["bin"]

        model = ie.read_model(model=xml_path, weights=bin_path)
        compiled_model = ie.compile_model(model=model)

        input_layer = next(iter(compiled_model.inputs))
        output_layer = next(iter(compiled_model.outputs))

        return input_layer, output_layer, compiled_model
