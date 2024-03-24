from typing import Any
from src.app.models.features import FaceModel, EyeModel, PointModel
from src.app.models.detection_models import EyeInferModel, FaceInferModel
from src.app.utils.image_utils import batch_process
from abc import ABC
import cv2

import numpy as np


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
        face_input_layer, face_output_layer, compiled_face_model = self.artifacts
        target_shape = (face_input_layer.shape[2], face_input_layer.shape[3])
        preprocessed_images = self.preprocess_images([img], target_shape)
        # logger.info(f"preprocess shape {preprocessed_image.shape}")

        # Perform inference
        results = compiled_face_model(preprocessed_images)[face_output_layer]
        print(results.shape)
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

    def filter_faces(self, faces, threshold=0.5):
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

    def detect(
        self, img: np.ndarray, face_coordinates: list[FaceModel]
    ) -> list[EyeModel]:
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
        eye_input_layer, eye_output_layer, compiled_eye_model = self.artifacts
        target_shape = (eye_input_layer.shape[2], eye_input_layer.shape[3])

        # Collect all face regions
        face_regions = self.extracti
        # Preprocess all face regions in a batch
        preprocessed_faces = self.preprocess_images(face_regions, target_shape)
        faces = self.preprocess_images(process_,)

        # Predict eyes for all faces in a single call
        eyes_batch = self.get_eyes(
            preprocessed_faces, eye_output_layer, compiled_eye_model, face_coordinates
        )
        return eyes_batch

    def extract_face_region(self, img: np.ndarray, fc: FaceModel) -> np.ndarray:
        def get_face()
        batch_process(
            [(img, fc) for fc in face_coordinates], self.extract_face_region
        )

        xmin, ymin, xmax, ymax = fc.coordinates
        return img[ymin:ymax, xmin:xmax]

    def get_eyes(
        self,
        preprocessed_faces: np.ndarray,
        eye_output_layer: int,
        compiled_eye_model,
        face_coordinates: list[FaceModel],
    ) -> list[EyeModel]:
        """
        Extracts eye landmarks from preprocessed faces using a compiled eye model.

        Parameters:
        - preprocessed_faces (np.ndarray): An array of preprocessed face images.
        - eye_output_layer (int): The output layer index for the eye landmarks in the compiled eye model.
        - compiled_eye_model: The compiled eye model used for landmark extraction.
        - face_coordinates (list[FaceModel]): A list of FaceModel objects containing face coordinates.

        Returns:
        - list[EyeModel]: A list of EyeModel objects representing the extracted eye landmarks.

        Raises:
        - None

        This function processes preprocessed face images to extract eye
        landmarks using a compiled eye model. It iterates over the face
        coordinates provided, calculates the eye landmarks based on the model
        output, and constructs EyeModel objects for the left and right eyes.
        The resulting EyeModel objects are then returned as a list.
        """

        results = compiled_eye_model(preprocessed_faces)[eye_output_layer]
        print(results.shape)

        results = results.reshape(results.shape[0], -1, 2)
        print(results.shape)
        print(len(face_coordinates))
        eye_batches = batch_process(
            [(rs, fc.coordinates) for rs, fc in zip(results, face_coordinates)],
            self.create_eye_model,
        )
        return eye_batches

    def create_eye_model(self, result, face_coords):

        def eye_model(indices):
            points = [
                PointModel(
                    x=int(result[j][0] * face_width + xmin),
                    y=int(result[j][1] * face_height + ymin),
                )
                for j in indices
            ]

            return EyeModel(points=points)

        result = result.reshape(-1, 2)
        xmin, ymin, xmax, ymax = face_coords
        face_width = xmax - xmin
        face_height = ymax - ymin
        # Define indices for left and right eyes and eyebrows
        eye_and_eyebrow_indices = {
            "left": [0, 1, 12, 13, 14],  # 0, 1 for eyes and 12-14 for eyebrows
            "right": [2, 3, 15, 16, 17],  # 2, 3 for eyes and 15-17 for eyebrows
        }

        left_eye_model = eye_model(eye_and_eyebrow_indices["left"])
        right_eye_model = eye_model(eye_and_eyebrow_indices["right"])

        return left_eye_model, right_eye_model

    # Function to create PointModels for eyes and eyebrows
