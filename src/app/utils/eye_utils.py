import random

import numpy as np

from src.app.models.features import EyeModel, FaceModel


def find_centroid(p1: tuple, p2: tuple) -> tuple:
    """
    Calculate the centroid of two points in 2D space.

    Parameters:
    p1 (tuple): A tuple representing the coordinates of the first point.
    p2 (tuple): A tuple representing the coordinates of the second point.

    Returns:
    tuple: A tuple representing the coordinates of the centroid of the two points.

    Raises:
    ValueError: If the input points are not in the correct format (i.e., not tuples).
    """

    # Check if input points are tuples
    if not isinstance(p1, tuple) or not isinstance(p2, tuple):
        raise ValueError("Input points must be tuples")

    # Convert points to numpy array
    points = np.array([p1, p2])

    # Calculate centroid
    centroid = np.mean(points, axis=0).astype(int)

    return tuple(centroid)


def calculate_eye_rotation(left_corner, right_corner):
    """
    Calculate the rotation angle of the eye based on the positions of the left and right eye corners.

    Parameters:
    left_corner (tuple): The coordinates of the left eye corner.
    right_corner (tuple): The coordinates of the right eye corner.

    Returns:
    float: The rotation angle of the eye in degrees.

    Raises:
    ValueError: If the input parameters are not valid tuples of coordinates.

    This function calculates the rotation angle of the eye by first calculating the vector between the right and left eye corners.
    It then calculates the angle of this vector in radians using arctan2, and converts it to degrees using np.degrees.
    The resulting angle in degrees is returned as the output of the function.
    """
    vector = np.array(right_corner) - np.array(left_corner)
    angle_radians = np.arctan2(vector[1], vector[0])
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def calculate_center_shift(
    eye_index: int, face_feature: np.ndarray, image_size: tuple[int, int]
) -> tuple[int, int]:
    """
    Calculate the shift needed to center the face feature in the image.

    Parameters:
    eye_index (int): Index of the eye used to calculate the shift.
    face_feature (np.array): Array representing the face feature to be centered.
    image_size (Tuple[int, int]): Tuple containing the width and height of the image.

    Returns:
    Tuple[int, int]: A tuple containing the shift values in the x and y directions.

    Raises:
    ValueError: If the face_feature array is empty.
    """

    # Calculate the scale factor based on the image size
    scale_factor = face_feature.size / max(image_size)

    # Calculate the maximum shift values based on the scale factor
    max_shift_x = int(scale_factor * 20)
    max_shift_y = int(scale_factor * 10)

    # Generate random shift values within the maximum shift limits
    shift_x = random.randint(-max_shift_x, max_shift_x) * (eye_index // 2)
    shift_y = random.randint(-max_shift_y, max_shift_y) * (eye_index // 2)

    return (shift_x, shift_y)


def calculate_googly_center_and_size(
    eye_feature: EyeModel,
    face_feature: FaceModel,
    image_size: tuple[int, int],
    positive_shift: bool = True,
) -> tuple[tuple[int, int], int]:
    """
    Calculate the center point and size for a googly eye based on the provided eye and face features.

    Parameters:
    - eye_feature (EyeModel): The model representing the eye feature.
    - face_feature (FaceModel): The model representing the face feature.
    - image_size (tuple[int, int]): A tuple representing the size of the image in pixels.
    - positive_shift (bool, optional): A flag indicating whether to apply a positive shift for the centroid. Default is True.

    Returns:
    - tuple[tuple[int, int], int]: A tuple containing the corrected center point coordinates and the size of the googly eye.

    Raises:
    - ValueError: If the image_size is not a tuple of two integers.
    - TypeError: If eye_feature is not of type EyeModel or face_feature is not of type FaceModel.
    """

    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or not all(isinstance(i, int) for i in image_size)
    ):
        raise ValueError("Image size must be a tuple of two integers.")

    if not isinstance(eye_feature, EyeModel):
        raise TypeError("eye_feature must be of type EyeModel.")

    if not isinstance(face_feature, FaceModel):
        raise TypeError("face_feature must be of type FaceModel.")

    image_width, image_height = image_size
    proportion_of_image = 0.001

    # Calculate base size using the image size
    base_size_from_image = int(min(image_width, image_height) * proportion_of_image)

    # Adjust the base size based on the face size, if available
    proportion_of_face = 0.3
    size_from_face = int(face_feature.size * proportion_of_face)

    # Blend the size based on the image and face sizes
    size = int((base_size_from_image + size_from_face) / 2)

    # Apply a dynamic increase to the size based on some criteria, for example, randomness
    dynamic_increase_factor = random.uniform(0.9, 1.1)
    size = int(size * dynamic_increase_factor)

    # Ensure the size is even for simplicity in further calculations
    size += size % 2

    # Calculate the center point between the two eye coordinates
    center = find_centroid(
        (eye_feature.xmin, eye_feature.ymin), (eye_feature.xmax, eye_feature.ymax)
    )

    # Calculate corrective shifts
    eye_height = eye_feature.ymax - eye_feature.ymin
    eye_width = eye_feature.xmax - eye_feature.xmin

    # Apply shifts. Positive shifts move the centroid down and to the right; negative shifts do the opposite.
    y_shift = int(-eye_height * 0.4) if positive_shift else int(eye_height * 0.9)
    x_shift = int(-eye_width * 0.4) if positive_shift else int(eye_width * 0.8)

    corrected_center = (center[0] + x_shift, center[1] + y_shift)

    return corrected_center, size
