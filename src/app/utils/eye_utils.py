import random

import numpy as np

from src.app.models.features import FaceModel


def find_centroid(points: list) -> np.ndarray:
    """
    Calculate the centroid of a list of points.

    Parameters:
    points (list): A list of points in the form of [x, y, z] coordinates.

    Returns:
    np.ndarray: The centroid of the points as a numpy array in the form [x, y, z].

    Raises:
    ValueError: If the input points list is empty or if the points are not in the correct format.
    """

    if not points:
        raise ValueError("Input points list is empty")

    # Convert points to numpy array
    points_array = np.array(points)

    # Calculate centroid
    centroid = np.mean(points_array, axis=0).astype(int)

    return centroid


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


def calculate_eye_center(eye_model):
    """
    Calculate the center of the eye.

    Parameters:
    eye_model (EyeModel): An instance of the EyeModel class containing the points of the eye.

    Returns:
    tuple: A tuple containing the x and y coordinates of the center of the eye.

    Raises:
    ValueError: If the eye_model is empty or does not contain any points.
    """

    # Extract x and y coordinates from the points in the eye_model
    x_coords = [point.x for point in eye_model.points]
    y_coords = [point.y for point in eye_model.points]

    # Calculate the centroid of x and y coordinates to find the center of the eye
    center_x = find_centroid(x_coords)
    center_y = find_centroid(y_coords)

    return center_x, center_y


def estimate_eye_size(eye_model):
    """
    Estimate the size of an eye based on a model of its points.

    Parameters:
    eye_model (EyeModel): A model of the eye containing a list of points.

    Returns:
    float: The estimated size of the eye, calculated as the maximum distance between any two points in the model.

    Raises:
    ValueError: If the eye_model is empty or does not contain enough points to calculate the size.
    """

    if not eye_model.points:
        raise ValueError(
            "The eye_model must contain at least one point to calculate the size."
        )

    x_coords = [point.x for point in eye_model.points]
    y_coords = [point.y for point in eye_model.points]
    distances = [
        np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        for x1, y1 in zip(x_coords, y_coords)
        for x2, y2 in zip(x_coords, y_coords)
    ]
    return max(distances)


def adjust_size_based_on_proportions(
    size: int, face_model: FaceModel, image_size: tuple[int, int]
) -> int:
    """
    Adjusts the size based on proportions of the face model and the image size.

    Parameters:
    size (int): The initial size to be adjusted.
    face_model (FaceModel): The face model object containing information about the face.
    image_size (Tuple[int, int]): The size of the image as a tuple of width and height.

    Returns:
    int: The adjusted size based on the proportions of the face model and image size.

    Raises:
    None
    """
    image_width, image_height = image_size
    proportion_of_image = 0.001
    base_size_from_image = int(min(image_width, image_height) * proportion_of_image)
    proportion_of_face = 0.4
    size_from_face = int(face_model.size * proportion_of_face)
    adjusted_size = int((base_size_from_image + size_from_face + size) / 3)
    return adjusted_size


def apply_dynamic_increase(size):
    """
    Apply a dynamic increase factor to the given size.

    Parameters:
    size (int): The original size to be increased.

    Returns:
    int: The size after applying the dynamic increase factor.

    Raises:
    None

    Example:
    >>> apply_dynamic_increase(100)
    110
    """
    dynamic_increase_factor = random.uniform(0.9, 1.1)
    return int(size * dynamic_increase_factor)


def apply_corrective_shifts(center: tuple, size: int, positive_shift: bool) -> tuple:
    """
    Apply corrective shifts based on the positive_shift flag.

    Parameters:
    center (tuple): A tuple representing the center point as (x, y).
    size (int): The size of the shift factor.
    positive_shift (bool): A flag indicating whether the shift should be positive or negative.

    Returns:
    tuple: A tuple representing the corrected center point after applying the shifts.

    Raises:
    None

    Example:
    >>> apply_corrective_shifts((5, 5), 10, True)
    (15, 15)
    """
    shift_factor = 0.0  # Adjust this as needed
    x_shift = int(shift_factor * size) if positive_shift else -int(shift_factor * size)
    y_shift = int(shift_factor * size) if positive_shift else -int(shift_factor * size)
    corrected_center = (int(center[0]) + x_shift, int(center[1]) + y_shift)
    return corrected_center


def calculate_googly_center_and_size(
    eye_model, face_model, image_size, positive_shift=True
):
    """
    Calculate the center and size of a googly eye based on the eye model, face model, and image size.

    Parameters:
    - eye_model (EyeModel): The model representing the googly eye.
    - face_model (FaceModel): The model representing the face.
    - image_size (tuple): A tuple of two integers representing the size of the image.
    - positive_shift (bool): A flag indicating whether to apply a positive shift to the center (default is True).

    Returns:
    - tuple: A tuple containing the corrected center coordinates and the size of the googly eye.

    Raises:
    - ValueError: If the image_size is not a tuple of two integers.
    - ValueError: If the corrected center x coordinate is outside the face region.
    """

    # Validate image_size
    if not (isinstance(image_size, tuple) and len(image_size) == 2):
        raise ValueError("Image size must be a tuple of two integers.")

    center_x, center_y = calculate_eye_center(eye_model)
    max_distance = estimate_eye_size(eye_model)
    size = adjust_size_based_on_proportions(max_distance, face_model, image_size)
    size = apply_dynamic_increase(size)
    size = size + size % 2

    corrected_center = apply_corrective_shifts(
        (center_x, center_y), size, positive_shift
    )

    # Assert that centroid is within face region
    if not (face_model.xmin < corrected_center[0] < face_model.xmax):
        raise ValueError("Corrected center x coordinate is outside face region")

    return corrected_center, size
