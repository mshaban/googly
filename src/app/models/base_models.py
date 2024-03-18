from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import numpy as np
from pydantic import BaseModel

from src.app.core.logger import logger
from src.app.models.features import EyeModel, FaceModel
from src.app.models.image import ImageModel
from src.app.models.model_registery import ModelRegistry


class InferenceModel(ABC, BaseModel):
    name: str
    model_artifacts: Optional[Any] = None
    registery: ModelRegistry = ModelRegistry()

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, *args, **kwargs):
        """
        Initializes the model by fetching model artifacts and calling the
        parent class's model_post_init method.

        Parameters:
        - self: the instance of the class
        - *args: variable length argument list
        - **kwargs: keyword arguments

        Returns:
        - None

        Raises:
        - No specific exceptions are raised by this method

        This method initializes the model by fetching model artifacts using the
        fetch_model method and then calls the model_post_init method of the
        parent class using super().
        """
        self.model_artifacts = self.fetch_model()
        super().model_post_init(*args, **kwargs)

    def fetch_model(self):
        """
        Fetches a model from the registry based on the name provided.

        Parameters:
        - self: The instance of the class.

        Returns:
        - model: The model fetched from the registry based on the name.

        Raises:
        - ValueError: If the model with the provided name is not found in the registry.

        This method fetches a model from the registry based on the name
        provided. It first checks if the model exists in the registry. If the
        model is not found, it logs an error message and raises a ValueError.
        Otherwise, it returns the fetched model.
        """
        model = self.registery.get_model(self.name)
        if not model:
            logger.error(f"Model {self.name} not found in registry.")
            raise ValueError("Model not found in registry.")
        return model

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
    def predict(self, data: Any) -> Any | list[Any]:
        """Define the prediction logic for the model"""
        pass


class FaceInferModel(InferenceModel):

    @property
    def input_schema(self) -> Type[ImageModel]:
        return ImageModel

    @property
    def output_schema(self) -> Type[FaceModel]:
        return FaceModel

    def predict(self, data: ImageModel) -> list[FaceModel]:
        """
        Predicts the faces present in the given image data.

        Parameters:
        - data (ImageModel): An ImageModel object containing the image data to be processed.

        Returns:
        - list[FaceModel]: A list of FaceModel objects representing the detected faces in the image.

        Raises:
        - ValueError: If the image data is not in a valid format.
        - RuntimeError: If there is an issue with detecting faces in the image.
        """

        image = np.array(data.image)
        logger.info(f"Detecting faces on {data.filename}")

        try:
            return self.detect(image)
        except Exception as e:
            logger.error(f"Error detecting faces in the image {data.filename}: {e}")
            raise RuntimeError("Error detecting faces in the image") from e

    @abstractmethod
    def detect(self, img: np.ndarray) -> list[FaceModel]:
        pass


class EyeInferModel(InferenceModel, ABC):
    faces: list[FaceModel]

    @property
    def input_schema(self) -> Type[ImageModel]:
        return ImageModel

    @property
    def output_schema(self) -> Type[EyeModel]:
        return EyeModel

    def predict(
        self,
        data: ImageModel,
    ) -> list[EyeModel]:
        """
        Predicts the location of eyes in the given image.

        Parameters:
        - data (ImageModel): An ImageModel object containing the image data to be processed.

        Returns:
        - list[EyeModel]: A list of EyeModel objects representing the predicted locations of eyes in the image.

        Raises:
        - ValueError: If the image data is not in a valid format.
        - RuntimeError: If there is an issue with detecting eyes in the image.
        """

        image = np.array(data.image)
        logger.info(f"Detecting eyes on {data.filename}")
        try:
            return self.detect(image, self.faces)
        except Exception as e:
            logger.error(f"Error detecting eyes in the image {data.filename}: {e}")
            raise RuntimeError("Error detecting eyes in the image") from e

    @abstractmethod
    def detect(
        self, img: np.ndarray, face_coordinates: list[FaceModel]
    ) -> list[EyeModel]:
        pass
