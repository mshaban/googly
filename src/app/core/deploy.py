import json
import logging

from app.models.features import EyeModel, FaceModel
from app.models.image import ImageModel
from fastapi.requests import Request
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse

from src.app.models.model_registery import ModelArtifacts
from src.app.models.vino_models import VinoEyeInferModel, VinoFaceInferModel


@serve.deployment
class VinoFaceInferModelServe:
    def __init__(self, config_path: str):
        artifacts = ModelArtifacts(config_path=config_path).setup_artifacts()
        self.model = VinoFaceInferModel(artifacts=artifacts)

    async def __call__(self, image: ImageModel) -> list[FaceModel]:
        result = self.model.predict(image)
        return result


@serve.deployment
class VinoEyeInferModelServe:
    def __init__(self, config_path: str):
        artifacts = ModelArtifacts(config_path=config_path).setup_artifacts()
        self.model = VinoEyeInferModel(artifacts=artifacts)

    async def __call__(self, image, face_data) -> list[EyeModel]:
        result = self.model.predict(image, face_data)
        return result


@serve.deployment(name="composite-model-service", route_prefix="/googly_serve")
class CompositeModelServe:
    def __init__(
        self,
        face_model: DeploymentHandle,
        eye_model: DeploymentHandle,
    ):
        self._face_model = face_model
        self._eye_model = eye_model
        self._logger = logging.getLogger("ray.serve")

    async def __call__(self, request) -> dict:
        """
        Perform face and eye detection on the input image.

        Args:
            image (ImageModel): The input image to perform face and eye detection on.

        Returns:
            dict: A dictionary containing the results of face and eye detection.
                The keys are "face_detection_results" and "eye_detection_results",
                and the values are DeploymentResponse objects.

        Raises:
            None

        This method asynchronously calls the face detection model and the eye detection model
        to perform face and eye detection on the input image. It returns a dictionary containing
        the results of both detections.
        """
        if isinstance(request, str) or isinstance(request, dict):
            image = ImageModel.model_validate(request)
        elif isinstance(request, Request):
            json_request = await request.json()
            json_object = json.loads(json_request)
            image = ImageModel.model_validate(json_object)
        elif isinstance(request, ImageModel):
            image = request
        else:
            raise ValueError(
                f"Invalid request type: {type(request)}. Request must be a string, dict, or ImageModel."
            )
        self._logger.info(f"Processing image: {image.filename}")
        face_detection_results: DeploymentResponse = await self._face_model.remote(
            image
        )
        eye_detection_results: DeploymentResponse = await self._eye_model.remote(
            image, face_detection_results
        )
        self._logger.info(f"Image processed: {image.filename}")
        return {
            "face_detection_results": face_detection_results,
            "eye_detection_results": eye_detection_results,
        }
