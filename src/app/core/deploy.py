from app.models.features import EyeModel, FaceModel
from app.models.image import ImageModel
from ray import serve
from src.app.models.model_registery import ModelArtifacts
from src.app.models.vino_models import VinoEyeInferModel, VinoFaceInferModel

from ray.serve.handle import DeploymentHandle, DeploymentResponse


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


@serve.deployment
class CompositeModelServe:
    def __init__(
        self,
        face_model: DeploymentHandle,
        eye_model: DeploymentHandle,
    ):
        self._face_model = face_model
        self._eye_model = eye_model

    async def __call__(self, image: ImageModel) -> dict:
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
        face_detection_results: DeploymentResponse = await self._face_model.remote(
            image
        )
        eye_detection_results: DeploymentResponse = await self._eye_model.remote(
            image, face_detection_results
        )

        return {
            "face_detection_results": face_detection_results,
            "eye_detection_results": eye_detection_results,
        }
