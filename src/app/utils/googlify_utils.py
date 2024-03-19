import numpy as np
from pydantic import BaseModel, model_validator

from src.app.models.features import EyeModel, FaceModel
from src.app.models.googly import GooglyModel
from src.app.models.image import ImageModel
from src.app.utils.eye_utils import (
    calculate_eye_rotation,
    calculate_googly_center_and_size,
)
from src.app.utils.image_utils import load_image_from_bytes
from src.app.utils.overlay_utils import (
    overlay_transparent,
)


@staticmethod
def apply_googly_eyes(googly: GooglyModel):
    """
    Apply googly eyes to the input image.

    This function takes the input image and overlays googly eyes on the detected eye regions.
    It first loads the input image and the googly eye images from bytes.
    It then iterates over the detected eye regions, calculates the center and size of the googly eye,
    and the rotation angle based on the eye coordinates.
    The googly eye image is then overlaid on the input image at the calculated center with the specified size and rotation.
    The opacity, gamma, overlay bias, and minimum alpha values can also be adjusted for the overlay.

    Parameters:
    - self: The instance of the class containing the input image, googly eye images, and eye regions.

    Returns:
    - input_img: The modified image with googly eyes overlaid on the detected eye regions.

    Raises:
    - ValueError: If the input image or googly eye images cannot be loaded.
    - IndexError: If there are not enough googly eye images for all detected eye regions.
    """
    input_img = load_image_from_bytes(googly.input_image.data)  # Load the main image
    input_image_size = input_img.shape[:2]
    overlay_center_x, overlay_center_y = 195, 210

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
):
    """
    Apply googly eyes on the detected face.
    This function takes the detected face and eye regions and applies googly eyes on the eyes.
    It calculates the center and size of the googly eye based on the eye coordinates and face features.
    The rotation angle of the eye is also calculated to align the googly eye correctly.
    The googly eye image is then overlaid on the input image at the calculated center with the specified size and rotation.
    Parameters:
    - self: The instance of the class containing the input image, googly eye images, and eye regions.
    - face_model: The FaceModel object representing the detected face.
    - eye_model: The EyeModel object representing the detected eye.
    Returns:
    - input_img: The modified image with googly eyes overlaid on the detected eye regions.
    Raises:
    - ValueError: If the input image or googly eye images cannot be loaded.
    - IndexError: If there are not enough googly eye images for all detected eye regions.
    """
    # Correct the assignment of the right eye model
    left_eye_model = eyes[0]
    right_eye_model = eyes[1]
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
        opacity=1.0,
        gamma=2.2,
        overlay_bias=1.2,
        min_alpha=0.1,
        overlay_center_x=overlay_center_x,
        overlay_center_y=overlay_center_y,
    )

    right_eye_center, right_eye_size = calculate_googly_center_and_size(
        right_eye_model, face_model, input_image_size
    )
    # Calculate properties for the right eye
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
        opacity=1.0,
        gamma=2.2,
        overlay_bias=1.2,
        min_alpha=0.1,
        overlay_center_x=overlay_center_x,
        overlay_center_y=overlay_center_y,
    )

    return input_img
