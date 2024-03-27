from abc import ABC
from typing import Any

import cv2
import numpy as np

from src.app.core.logger import logger
from src.app.models.detection_models import EyeInferModel, FaceInferModel
from src.app.models.features import EyeModel, FaceModel, PointModel
from src.app.utils.model_utils import batch_process, batch_vector_generator


class BaseVinoModel(ABC):
    def preprocess_images(
        self, images: list[np.ndarray], input_shape: tuple[int, int]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Preprocesses a list of images for input into a neural network model in parallel.
        Args:
            images (list[np.ndarray]): The list of input images to be
            preprocessed. Each should be a 3D numpy array representing an image.
            input_shape (Tuple[int, int]): The desired input shape of the
            images in the format (height, width).
        Returns:
            list[np.ndarray]: The list of preprocessed images ready for input
            into a neural network model. Each will be a 4D numpy array with
            shape (1, channels, height, width).
        Raises:
            ValueError: If any input image is not a 3D numpy array.
        """

        def preprocess_single_image(image: np.ndarray) -> np.ndarray:
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            if image.ndim != 3:
                raise ValueError("Image must be 3D")
            input_height, input_width = input_shape
            image = cv2.resize(image, (input_width, input_height))
            image = image.transpose((2, 0, 1))
            return image

        results = np.array(
            batch_process(images, preprocess_single_image), dtype=np.float32
        )

        return results


class VinoFaceInferModel(FaceInferModel, BaseVinoModel):
    name: str = "face-detection-adas-0001"

    def detect(self, img: np.ndarray) -> list[FaceModel]:

        # Preprocess the image for the face detection model
        face_input_layer, face_output_layer, compiled_face_model = (
            self.artifacts.input_layer,
            self.artifacts.output_layer,
            self.artifacts.compiled_model,
        )
        target_shape = (face_input_layer.shape[2], face_input_layer.shape[3])
        preprocessed_images = self.preprocess_images([img], target_shape)

        # Perform inference
        batch_size = face_input_layer.shape[0]
        n_samples = 1
        # padded_images = pad_vector(preprocessed_images, batch_size - n_samples)
        results = compiled_face_model(preprocessed_images)[face_output_layer]
        results = self.filter_faces(results[0][0])

        # Calculate the bounding box coordinates of the detected faces
        height, width = img.shape[:2]
        faces = [
            FaceModel(
                xmin=int(max(0, (face[-4] * width))),
                ymin=int(max(0, (face[-3] * height))),
                xmax=int(min(width, (face[-2] * width))),
                ymax=int(min(height, (face[-1] * height))),
            )
            for face in results
        ]
        return faces

    def filter_faces(self, faces: list, threshold=0.5):
        """
        Filter out faces with confidence scores below a given threshold.
        Parameters:
        - faces: The detected faces from the model's output.
        - threshold: The confidence threshold below which faces will be filtered out.
        Returns:
        - filtered_faces: The faces that passed the confidence threshold.
        """
        filtered_faces = [face for face in faces if face[2] > threshold]
        return np.array(filtered_faces)


class VinoEyeInferModel(EyeInferModel, BaseVinoModel):
    name: str = "facial-landmarks-35-adas-0002"

    def detect(self, img: np.ndarray, faces: list[FaceModel]) -> list[EyeModel]:
        """
        Detects eyes in the given image based on the provided face coordinates.

        Parameters:
        img (np.ndarray): The input image in which to detect eyes.
        face_coordinates (list[FaceModel]): A list of FaceModel objects
        representing the coordinates of faces.

        Returns:
        list[EyeModel]: A list of EyeModel objects representing the detected eyes.

        Raises:
        None

        This method takes an input image and a list of face coordinates,
        extracts the face regions from the image, preprocesses them, and then
        predicts the eyes for each face in a single call. The predicted eyes
        are returned as a list of EyeModel objects.
        """
        eye_input_layer, eye_output_layer, compiled_eye_model = (
            self.artifacts.input_layer,
            self.artifacts.output_layer,
            self.artifacts.compiled_model,
        )
        batch_size = eye_input_layer.shape[0]
        target_shape = (eye_input_layer.shape[2], eye_input_layer.shape[3])

        # Collect all face regions
        face_regions = self.extract_face_region(img, faces)

        # Preprocess all face regions in a batch
        preprocessed_faces = self.preprocess_images(face_regions, target_shape)

        # Patch detected faces
        batched_results = []
        for ixb, batch in enumerate(
            batch_vector_generator(preprocessed_faces, batch_size)
        ):
            batch_faces = faces[ixb * batch_size : (ixb + 1) * batch_size]
            eyes_batch = self.get_eyes(
                batch, eye_output_layer, compiled_eye_model, batch_faces
            )
            batched_results.extend(eyes_batch)

        return batched_results

    def extract_face_region(
        self, img: np.ndarray, faces: list[FaceModel]
    ) -> list[np.ndarray]:
        def get_face(img, face_coordinates):
            xmin, ymin, xmax, ymax = face_coordinates
            return img[ymin:ymax, xmin:xmax]

        face_images = batch_process([(img, fc.coordinates) for fc in faces], get_face)

        return face_images

    def get_eyes(
        self,
        preprocessed_faces: np.ndarray,
        eye_output_layer: int,
        compiled_eye_model,
        faces: list[FaceModel],
    ) -> list[EyeModel]:

        results = compiled_eye_model(preprocessed_faces)[eye_output_layer]
        results = results.reshape(results.shape[0], -1, 2)
        eye_batches = batch_process(
            [(rs, fc) for rs, fc in zip(results, faces)],
            self.create_eye_model,
        )

        return eye_batches

    def create_eye_model(self, eyes_result, face: FaceModel):

        def eye_model(indices):
            logger.debug(f"Indices: {indices}")
            logger.debug(f"Eyes Result: {eyes_result[indices[0]]}")
            points = [
                PointModel(
                    x=int(eyes_result[j][0] * face_width + xmin),
                    y=int(eyes_result[j][1] * face_height + ymin),
                )
                for j in indices
            ]
            return EyeModel(points=points)

        xmin, ymin, xmax, ymax = face.coordinates
        face_width = face.width
        face_height = face.height
        # Define indices for left and right eyes and eyebrows
        eye_and_eyebrow_indices = {
            "left": [0, 1, 12, 13, 14],  # 0, 1 for eyes and 12-14 for eyebrows
            "right": [2, 3, 15, 16, 17],  # 2, 3 for eyes and 15-17 for eyebrows
        }

        left_eye_model = eye_model(eye_and_eyebrow_indices["left"])
        right_eye_model = eye_model(eye_and_eyebrow_indices["right"])
        # Log everything in this func to debug
        logger.debug(f"Face: {face}")
        logger.debug(f"Left Eye: {left_eye_model}")
        logger.debug(f"Right Eye: {right_eye_model}")
        logger.debug(f"Face Width: {face_width}")
        logger.debug(f"Face Height: {face_height}")

        return left_eye_model, right_eye_model, face

    # Function to create PointModels for eyes and eyebrows
