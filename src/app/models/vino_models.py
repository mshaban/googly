from abc import ABC
from typing import Any

import cv2
import numpy as np
from openvino.runtime import ConstOutput, CompiledModel

from src.app.core.logger import logger
from src.app.models.detection_models import EyeInferModel, FaceInferModel
from src.app.models.features import EyeModel, FaceModel, PointModel
from src.app.utils.model_utils import (
    batch_process,
    batch_vector_generator,
)


class BaseVinoModel(ABC):
    def _resize_images_for_target(
        self, images: list[np.ndarray], input_shape: tuple[int, int]
    ) -> np.ndarray:

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

    def _init_detection_config(self):
        """Initialize and return detection configuration."""
        face_input_layer = self.artifacts.input_layer
        target_shape = (face_input_layer.shape[2], face_input_layer.shape[3])
        return (
            face_input_layer,
            self.artifacts.output_layer,
            self.artifacts.compiled_model,
            target_shape,
        )


class VinoFaceInferModel(FaceInferModel, BaseVinoModel):
    name: str = "face-detection-adas-0001"

    def detect(self, img: np.ndarray) -> list[FaceModel]:
        # Initialize detection configuration
        face_input_layer, face_output_layer, compiled_face_model, target_shape = (
            self._init_detection_config()
        )
        preprocessed_images = self._resize_images_for_target([img], target_shape)

        # Perform inference and process results
        return self._infer_and_process(
            preprocessed_images,
            face_input_layer,
            face_output_layer,
            compiled_face_model,
            img,
        )

    def _infer_and_process(
        self,
        preprocessed_images: np.ndarray,
        face_input_layer: ConstOutput,
        face_output_layer: ConstOutput,
        compiled_face_model: CompiledModel,
        original_img: np.ndarray,
    ):
        """Infer and process the face detection in batches."""
        # Calculate batch size and initialize final faces list
        batch_size = face_input_layer.shape[0]
        final_faces = []

        # Calculate the number of preprocessed images and the required batch information
        n_preprocessed_images = preprocessed_images.shape[0]
        n_missing_samples, total_batches = self._calculate_batch_info(
            n_preprocessed_images, batch_size
        )

        # Iterate over batches and process each
        for ixb, batch in enumerate(
            batch_vector_generator(preprocessed_images, batch_size)
        ):
            # Infer faces from the current batch
            results = compiled_face_model(batch)[face_output_layer]

            # Extract valid detections from the results
            valid_image_detections = self._extract_valid_detections(
                results, ixb, total_batches, n_missing_samples, batch_size
            )

            # Filter faces from the valid detections
            filtered_faces = self.filter_faces(valid_image_detections)

            # Create face models from the filtered faces
            face_models = self.create_face_model(
                filtered_faces, original_img.shape[1], original_img.shape[0]
            )

            # Extend the final faces list with the current face models
            final_faces.extend(face_models)

        # Return the final list of face models
        return final_faces

    def _calculate_batch_info(self, n_preprocessed_images: int, batch_size: int):
        """Calculate and return batch processing information."""

        # Calculate the number of missing samples to complete the last batch
        n_missing_samples = (
            batch_size - (n_preprocessed_images % batch_size)
        ) % batch_size

        # Calculate the total number of batches needed
        total_batches = (n_preprocessed_images + batch_size - 1) // batch_size

        # Return the calculated values
        return n_missing_samples, total_batches

    def _extract_valid_detections(
        self,
        results: np.ndarray,
        current_batch_idx: int,
        total_batches: int,
        n_missing_samples: int,
        batch_size: int,
    ):
        """Extract valid detections from results considering the batch context."""

        # Check if it's the last batch and there are missing samples
        if current_batch_idx + 1 == total_batches and n_missing_samples > 0:
            # Filter detections based on the adjusted batch size
            return [
                detection
                for detection in results[0, 0]
                if detection[0] < (batch_size - n_missing_samples)
            ]
        else:
            # Return all detections for non-final or complete batches
            return results[0, 0]

    def filter_faces(self, faces: list[np.ndarray], threshold: float = 0.5):
        """Filter faces based on a confidence threshold."""
        return np.array([face for face in faces if face[2] > threshold])

    def create_face_model(self, results: np.ndarray, img_width: int, img_height: int):
        """Create FaceModel instances for each detection."""

        def face_model(face):
            return FaceModel(
                xmin=int(max(0, (face[-4] * img_width))),
                ymin=int(max(0, (face[-3] * img_height))),
                xmax=int(min(img_width, (face[-2] * img_width))),
                ymax=int(min(img_height, (face[-1] * img_height))),
            )

        return batch_process([fc for fc in results], face_model)


class VinoEyeInferModel(EyeInferModel, BaseVinoModel):
    name: str = "facial-landmarks-35-adas-0002"

    def detect(self, img: np.ndarray, faces: list[FaceModel]) -> list[EyeModel]:
        # Initialize detection configuration
        eye_input_layer, eye_output_layer, compiled_eye_model, target_shape = (
            self._init_detection_config()
        )

        # Determine batch size from the input layer shape
        batch_size = eye_input_layer.shape[0]

        # Preprocess faces for detection
        preprocessed_faces = self._preprocess_faces_for_detection(
            img, faces, target_shape
        )

        # Process preprocessed faces in batches and return the detected eyes
        return self._process_faces_in_batches(
            preprocessed_faces, faces, batch_size, compiled_eye_model, eye_output_layer
        )

    def _preprocess_faces_for_detection(
        self, img: np.ndarray, faces: list[FaceModel], target_shape
    ):
        """Preprocess face images to prepare for eye detection."""
        face_regions = self._extract_face_region(img, faces)
        return self._resize_images_for_target(face_regions, target_shape)

    def _extract_face_region(
        self, img: np.ndarray, faces: list[FaceModel]
    ) -> list[np.ndarray]:
        """Extracts the face regions from the input image based on the provided
        list of FaceModel objects."""

        def get_face(img, face):
            xmin, ymin, xmax, ymax = face.coordinates
            return img[ymin:ymax, xmin:xmax]

        face_images = batch_process([(img, fc) for fc in faces], get_face)

        return face_images

    def _process_faces_in_batches(
        self,
        preprocessed_faces: np.ndarray,
        faces: list[FaceModel],
        batch_size: int,
        compiled_eye_model: CompiledModel,
        eye_output_layer: ConstOutput,
    ):
        """Process preprocessed faces in batches to detect eyes."""

        # Initialize variables for batch processing
        eye_models = []
        processed_faces_count = 0
        n_missing_faces = 0

        # Loop through batches of preprocessed faces
        for batch in batch_vector_generator(preprocessed_faces, batch_size):
            current_batch_size = len(batch)
            batch_end_idx = processed_faces_count + current_batch_size

            # Handle cases where the batch end index exceeds the number of faces
            if batch_end_idx > len(faces):
                n_missing_faces = batch_end_idx - len(faces)
                batch_faces = faces[processed_faces_count:]
            else:
                batch_faces = faces[processed_faces_count:batch_end_idx]

            # Detect eyes in the current batch
            eye_positions = self._predict_eye_positions(
                batch, eye_output_layer, compiled_eye_model
            )

            # Adjust the results for any missing faces in the batch
            eye_positions = self._adjust_eyes_batch_for_missing_faces(
                eye_positions, n_missing_faces
            )

            # Process the eye results to create EyeModel instances
            eyes_batch = self._process_eye_positions(eye_positions, batch_faces)

            # Collect the results from the current batch
            eye_models.extend(eyes_batch)
            processed_faces_count += current_batch_size

        # Return the aggregated results from all batches
        return eye_models

    def _adjust_eyes_batch_for_missing_faces(
        self, eyes_batch: np.ndarray, n_missing_faces: int
    ):
        """
        Adjust the eyes batch size if there were missing faces in the batch.
        """
        if n_missing_faces > 0:
            return eyes_batch[:-n_missing_faces]
        return eyes_batch

    def _predict_eye_positions(
        self,
        preprocessed_faces: np.ndarray,
        eye_output_layer: ConstOutput,
        compiled_eye_model: CompiledModel,
    ):
        """Predict eye positions using the compiled eye model."""
        results = compiled_eye_model(preprocessed_faces)[eye_output_layer]
        return results.reshape(results.shape[0], -1, 2)

    def _process_eye_positions(self, results: np.ndarray, faces: list[FaceModel]):
        """Process results to create eye models."""
        processed_batch = batch_process(
            [(result, face) for result, face in zip(results, faces)],
            self._create_eye_model,
        )

        return self._flatten_eye_batch(processed_batch)

    def _flatten_eye_batch(self, eye_batches: list[tuple[EyeModel, EyeModel]]):
        """Flatten the list of eye batches into a single list."""
        return [eye for batch in eye_batches for eye in batch]

    def _create_eye_model(self, eyes_result: np.ndarray, face: FaceModel):
        """Create models for left and right eyes based on detection results."""
        # Extract face dimensions for further calculations
        xmin, ymin, face_width, face_height = self._extract_face_dimensions(face)

        # Define indices for eyes and eyebrows to locate them in the face model
        eye_and_eyebrow_indices = self._define_eye_and_eyebrow_indices()

        # Create a model for the left eye using the extracted dimensions and indices
        left_eye_model = self._create_single_eye_model(
            eyes_result,
            eye_and_eyebrow_indices["left"],
            xmin,
            ymin,
            face_width,
            face_height,
        )

        # Create a model for the right eye using the extracted dimensions and indices
        right_eye_model = self._create_single_eye_model(
            eyes_result,
            eye_and_eyebrow_indices["right"],
            xmin,
            ymin,
            face_width,
            face_height,
        )

        # Return the models for both eyes
        return left_eye_model, right_eye_model

    def _extract_face_dimensions(self, face: FaceModel):
        """Extract face dimensions from a FaceModel object."""
        xmin, ymin = face.xmin, face.ymin
        face_width = face.width
        face_height = face.height
        return xmin, ymin, face_width, face_height

    def _define_eye_and_eyebrow_indices(self):
        """Define indices for left and right eyes and eyebrows."""
        return {
            "left": [0, 1, 12, 13, 14],  # Indices for left eye and eyebrow
            "right": [2, 3, 15, 16, 17],  # Indices for right eye and eyebrow
        }

    def _create_single_eye_model(
        self,
        eyes_result: np.ndarray,
        indices: list,
        xmin: int,
        ymin: int,
        face_width: int,
        face_height: int,
    ):
        """Create an EyeModel for either the left or right eye."""
        points = [
            PointModel(
                x=int(eyes_result[j][0] * face_width + xmin),
                y=int(eyes_result[j][1] * face_height + ymin),
            )
            for j in indices
        ]
        return EyeModel(points=points)
