import numpy as np

from src.app.models.features import EyeModel, FaceModel
from src.app.models.googly import GooglyModel
from src.app.utils.eye_utils import (
    calculate_eye_rotation,
    calculate_googly_center_and_size,
)
from src.app.utils.image_utils import load_image_from_bytes
from src.app.utils.overlay_utils import (
    overlay_transparent,
)


@staticmethod
def apply_googly_eyes(
    googly: GooglyModel, overlay_center_x=195, overlay_center_y=210
) -> np.ndarray:
    """
    Apply googly eyes to the input image based on the provided GooglyModel.

    Parameters:
    googly (GooglyModel): A data model containing information about the input image, googly eyes, faces, and overlay details.
    overlay_center_x (int): The x-coordinate of the center of the googly eye overlay. Default is 195.
    overlay_center_y (int): The y-coordinate of the center of the googly eye overlay. Default is 210.

    Returns:
    np.ndarray: The input image with googly eyes applied.

    Raises:
    ValueError: If the input image data or googly eye data is invalid.
    IndexError: If there are not enough eyes or faces provided in the GooglyModel.
    """

    input_img = load_image_from_bytes(googly.input_image.data)
    input_image_size = input_img.shape[:2]

    googly_eye_img = load_image_from_bytes(googly.googly.data, with_alpha=True)

    for ix in range(0, len(googly.eyes) - 1, 2):
        face_model = googly.faces[ix // 2]
        input_img = apply_on_face(
            face_model,
            googly.eyes[ix : ix + 2],
            input_img,
            input_image_size,
            googly_eye_img,
            overlay_center_x,
            overlay_center_y,
        )

    return input_img


@staticmethod
def apply_on_face(
    face_model: FaceModel,
    eyes: list[EyeModel],
    input_img: np.ndarray,
    input_image_size: tuple[int, int],
    googly_eye_img: np.ndarray,
    overlay_center_x: int,
    overlay_center_y: int,
) -> np.ndarray:
    """
    Apply googly eyes on a face image.

    Parameters:
    - face_model (FaceModel): The model representing the face in the image.
    - eyes (list[EyeModel]): List of EyeModel objects representing the eyes in the image.
    - input_img (np.ndarray): The input image on which to apply the googly eyes.
    - input_image_size (tuple[int, int]): The size of the input image (width, height).
    - googly_eye_img (np.ndarray): The image of the googly eyes to overlay on the face.
    - overlay_center_x (int): The x-coordinate of the center of the overlay.
    - overlay_center_y (int): The y-coordinate of the center of the overlay.

    Returns:
    - np.ndarray: The image with googly eyes applied.

    Raises:
    - ValueError: If the number of eyes provided is not equal to 2.
    - TypeError: If any of the input parameters are of incorrect type.

    This function applies googly eyes on the input image by calculating the properties for each eye, including center, size, and rotation.
    It then overlays the googly eyes on the input image for both eyes and returns the resulting image.
    """

    if len(eyes) != 2:
        raise ValueError("Exactly 2 eyes are required for applying googly eyes.")

    # Correct the assignment of the right eye model
    left_eye_model, right_eye_model = eyes

    # Calculate properties for the left eye
    left_eye_center, left_eye_size = calculate_googly_center_and_size(
        left_eye_model, face_model, input_image_size
    )
    left_eye_rotation = calculate_eye_rotation(
        left_corner=(left_eye_model.points[0].x, left_eye_model.points[0].y),
        right_corner=(left_eye_model.points[1].x, left_eye_model.points[1].y),
    )

    # Apply googly eye for the left eye
    input_img = overlay_transparent(
        input_img,
        googly_eye_img,
        left_eye_center,
        left_eye_size,
        rotation=left_eye_rotation,
        overlay_center_x=overlay_center_x,
        overlay_center_y=overlay_center_y,
    )

    # Calculate properties for the right eye
    right_eye_center, right_eye_size = calculate_googly_center_and_size(
        right_eye_model, face_model, input_image_size
    )
    right_eye_rotation = calculate_eye_rotation(
        left_corner=(right_eye_model.points[0].x, right_eye_model.points[0].y),
        right_corner=(right_eye_model.points[1].x, right_eye_model.points[1].y),
    )

    # Apply googly eye for the right eye
    input_img = overlay_transparent(
        input_img,
        googly_eye_img,
        right_eye_center,
        right_eye_size,
        rotation=right_eye_rotation,
        overlay_center_x=overlay_center_x,
        overlay_center_y=overlay_center_y,
    )

    return input_img
