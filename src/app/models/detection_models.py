from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import numpy as np
from pydantic import BaseModel

from src.app.core.logger import logger
from src.app.models.features import EyeModel, FaceModel
from src.app.models.image import ImageModel
from src.app.utils.image_utils import load_image_from_bytes


class InferenceModel(ABC, BaseModel):
    name: str
    artifacts: Optional[Any]

    class Config:
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def input_schema(self) -> Type[BaseModel]:
        """Define the input schema for the model"""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Type[BaseModel]:
        """Define the output schema for the model"""
        pass

    @abstractmethod
    def predict(self, request: Any, *args) -> Any | list[Any]:
        """Define the prediction logic for the model"""
        pass


class FaceInferModel(InferenceModel):

    @property
    def input_schema(self) -> Type[ImageModel]:
        return ImageModel

    @property
    def output_schema(self) -> Type[FaceModel]:
        return FaceModel

    def predict(self, request: ImageModel, *args) -> list[FaceModel]:
        """
        Predicts the faces present in the given image data.

        Parameters:
        - request (ImageModel): An ImageModel object containing the image data to be processed.

        Returns:
        - list[FaceModel]: A list of FaceModel objects representing the detected faces in the image.

        Raises:
        - ValueError: If the image data is not in a valid format.
        - RuntimeError: If there is an issue with detecting faces in the image.
        """

        image = load_image_from_bytes(request.data)
        logger.info(f"Detecting faces on {request.filename}")

        try:
            faces = self.detect(image)
            logger.info(f"Detected {len(faces)} faces in the {request.filename}.")
            return faces
        except Exception as e:
            logger.error(f"Error detecting faces in the image {request.filename}: {e}")
            raise RuntimeError("Error detecting faces in the image") from e

    @abstractmethod
    def detect(self, img: np.ndarray) -> list[FaceModel]:
        pass


class EyeInferModel(InferenceModel, ABC):

    @property
    def input_schema(self) -> Type[ImageModel]:
        return ImageModel

    @property
    def output_schema(self) -> Type[EyeModel]:
        return EyeModel

    def predict(self, request: ImageModel, faces: list[FaceModel]) -> list[EyeModel]:
        """
        Predicts the location of eyes in the given image.

        Parameters:
        - request (ImageModel): An ImageModel object containing the image data to be processed.

        Returns:
        - list[EyeModel]: A list of EyeModel objects representing the predicted locations of eyes in the image.

        Raises:
        - ValueError: If the image data is not in a valid format.
        - RuntimeError: If there is an issue with detecting eyes in the image.
        """
        image = load_image_from_bytes(request.data)
        logger.info(f"Detecting eyes on {request.filename}")
        try:
            eyes = self.detect(image, faces)
            logger.info(f"Detected {len(eyes)} eyes in the {request.filename}.")
            return eyes
        except Exception as e:
            logger.error(f"Error detecting eyes in the image {request.filename}: {e}")
            raise RuntimeError("Error detecting eyes in the image") from e

    @abstractmethod
    def detect(
        self, img: np.ndarray, face_coordinates: list[FaceModel]
    ) -> list[EyeModel]:
        pass
