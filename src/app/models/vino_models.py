from src.app.models.features import FaceModel, EyeModel
from src.app.models.base_models import EyeInferModel, FaceInferModel
from abc import ABC
import cv2

import numpy as np


class BaseVinoModel(ABC):
    def preprocess_image(
        self, image: np.ndarray, input_shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Preprocesses an image for input into a neural network model.

        Args:
            image (np.ndarray): The input image to be preprocessed. It should be a 3D numpy array representing an image.
            input_shape (Tuple[int, int]): The desired input shape of the image in the format (height, width).

        Returns:
            np.ndarray: The preprocessed image ready for input into a neural
            network model. It will be a 4D numpy array with shape (1, channels,
            height, width).

        Raises:
            ValueError: If the input image is not a 3D numpy array.
        """

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.ndim != 3:
            raise ValueError("Image must be 3D")

        input_height, input_width = input_shape
        image = cv2.resize(image, (input_width, input_height))
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image


class VinoFaceInferModel(FaceInferModel, BaseVinoModel):
    name: str = "face-detection-adas-0001"

    def detect(self, img: np.ndarray) -> list[FaceModel]:
        """
        Detect faces in the input image using a pre-trained face detection model.

        Parameters:
        img (np.ndarray): The input image in the form of a NumPy array.

        Returns:
        list[FaceModel]: A list of FaceModel objects representing the detected faces in the image.

        Raises:
        ValueError: If the input image is not a valid NumPy array.
        RuntimeError: If there is an issue with the face detection model or inference process.

        This method takes an input image and preprocesses it for the face
        detection model. It then performs inference using the model and filters
        the detected faces. The bounding box coordinates of the detected faces
        are calculated based on the input image dimensions and returned as a
        list of FaceModel objects.
        """

        # Preprocess the image for the face detection model
        face_input_layer, face_output_layer, compiled_face_model = self.model_artifacts
        target_shape = (face_input_layer.shape[2], face_input_layer.shape[3])
        preprocessed_image = self.preprocess_image(img, target_shape)

        # Perform inference
        results = compiled_face_model([preprocessed_image])[face_output_layer]
        results = self.filter_faces(results[0][0])

        # Calculate the bounding box coordinates of the detected faces
        height, width = img.shape[:2]
        faces = [
            FaceModel(
                xmin=int(face[-4] * width),
                ymin=int(face[-3] * height),
                xmax=int(face[-2] * width),
                ymax=int(face[-1] * height),
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
        img (np.ndarray): The input image in which eyes are to be detected.
        face_coordinates (list[FaceModel]): A list of FaceModel objects representing the coordinates of faces in the image.

        Returns:
        list[EyeModel]: A list of EyeModel objects representing the detected eyes in the image.

        Raises:
        None

        This method takes an input image and a list of face coordinates, and
        uses a pre-trained eye detection model to detect eyes in the image. It
        returns a list of EyeModel objects representing the detected eyes. The
        method internally uses helper functions to extract eyes from the faces
        based on the provided face coordinates.
        """

        eye_input_layer, eye_output_layer, compiled_eye_model = self.model_artifacts
        target_shape = (eye_input_layer.shape[2], eye_input_layer.shape[3])

        # Get the eyes for each face
        eyes = [
            face_eyes
            for fc in face_coordinates
            for face_eyes in self.get_eyes(
                img, fc, eye_output_layer, compiled_eye_model, target_shape
            )
        ]

        return eyes

    def get_eyes(
        self,
        img: np.ndarray,
        fc: FaceModel,
        eye_output_layer: int,
        compiled_eye_model,
        target_shape: tuple[int, int],
    ) -> tuple[EyeModel, EyeModel]:
        """
        Extracts the coordinates of the left and right eyes from a given image.

        Parameters:
        - img: The input image as a numpy array.
        - fc: The coordinates of the detected face in the image.
        - eye_output_layer: The index of the output layer for the eye model.
        - compiled_eye_model: The compiled eye model for predicting eye coordinates.
        - target_shape: The target shape for preprocessing the face region.

        Returns:
        - Tuple[EyeModel, EyeModel]: A tuple containing the left and right EyeModel objects representing the detected eyes.

        Raises:
        - None

        This function takes the input image, extracts the face region based on the detected face coordinates,
        preprocesses the face region, and uses the compiled eye model to predict the eye coordinates.
        The predicted coordinates are then transformed to the actual image coordinates and used to create
        EyeModel objects for the left and right eyes, which are returned as a tuple.
        """
        # Extract the face region from the image
        xmin, ymin, xmax, ymax = fc.coordinates
        face_width = xmax - xmin
        face_height = ymax - ymin
        face_region = img[ymin:ymax, xmin:xmax]

        # Preprocess the face region for the eye model
        preprocessed_face = self.preprocess_image(face_region, target_shape)
        results = compiled_eye_model([preprocessed_face])[eye_output_layer]
        results = results[0].reshape(-1, 2)

        # Filter landmarks for the eyes and extract left and right eye coordinates
        eyes_only = results[:4]
        left_eye = eyes_only[:2]
        right_eye = eyes_only[2:]

        left_eye_model = EyeModel(
            xmin=int(left_eye[0][0] * face_width + xmin),
            ymin=int(left_eye[0][1] * face_height + ymin),
            xmax=int(left_eye[1][0] * face_width + xmin),
            ymax=int(left_eye[1][1] * face_height + ymin),
        )
        right_eye_model = EyeModel(
            xmin=int(right_eye[0][0] * face_width + xmin),
            ymin=int(right_eye[0][1] * face_height + ymin),
            xmax=int(right_eye[1][0] * face_width + xmin),
            ymax=int(right_eye[1][1] * face_height + ymin),
        )

        return left_eye_model, right_eye_model
