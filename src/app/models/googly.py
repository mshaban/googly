from pydantic import BaseModel, model_validator

from src.app.models.features import EyeModel, FaceModel
from src.app.models.image import ImageModel
from src.app.core.logger import logger


class GooglyModel(BaseModel):
    faces: list[FaceModel]
    eyes: list[EyeModel]
    input_image: ImageModel
    googly: ImageModel

    def model_post_init(self, *args, **kwargs):
        self.eyes = [
            EyeModel(points=self.normalize_eye_points(eye)) for eye in self.eyes
        ]

        super().model_post_init(*args, **kwargs)

    def normalize_eye_points(self, eye_model):
        """
        Sorts the eye points in the given eye model by their x-coordinate in ascending order.

        Args:
        eye_model (EyeModel): The EyeModel object containing the points to be normalized.

        Returns:
        list: A list of Point objects sorted by their x-coordinate in ascending order.

        Raises:
        TypeError: If the input eye_model is not of type EyeModel.
        """

        if not isinstance(eye_model, EyeModel):
            raise TypeError("Input must be of type EyeModel")

        sorted_points = sorted(eye_model.points, key=lambda point: point.x)
        return sorted_points
