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
    def __init__(self, config_path: str, batch_size: int):
        self.model_artifacts = self.load_config(config_path)
        self.batch_size = batch_size
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
        self.setup_artifacts()

    def load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as config_file:
            return json.load(config_file)

    def setup_artifacts(self):
        ie = Core()
        model_path = Path(self.model_artifacts["path"])
        precision = self.model_artifacts["precision"]
        xml_path = model_path / precision / self.model_artifacts["xml"]
        bin_path = model_path / precision / self.model_artifacts["bin"]

        model = ie.read_model(model=xml_path, weights=bin_path)

        input_layer = next(iter(model.inputs))
        input_shape = input_layer.shape
        # Convert input_shape to a list and update the batch size
        input_shape = list(input_shape)
        input_shape[0] = self.batch_size
        # Reshape the model using a compatible type for the new shape
        model.reshape({input_layer.any_name: input_shape})
        self.compiled_model = ie.compile_model(model=model)

        self.input_layer = next(iter(self.compiled_model.inputs))
        self.output_layer = next(iter(self.compiled_model.outputs))
