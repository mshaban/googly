import random

import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate


def calculate_target_size(background_img, target_size):
    """
    Calculate the target size for the overlay image based on the background image dimensions and desired size.

    Parameters:
    - background_img: np.array, Background image array.
    - target_size: int, Desired target size of the overlay.

    Returns:
    - int: Calculated target size.
    """
    background_height, background_width = background_img.shape[:2]
    return min(target_size, background_width, background_height)


def resize_and_center_overlay(
    overlay_img, target_size, overlay_center_x=None, overlay_center_y=None
):
    """
    Resize the overlay image to the target size and calculate the new center if specific center coordinates are provided.

    Parameters:
    - overlay_img: np.array, Overlay image array.
    - target_size: int, Target size for the overlay.
    - overlay_center_x: int or None, Original X coordinate of the overlay's center.
    - overlay_center_y: int or None, Original Y coordinate of the overlay's center.

    Returns:
    - np.array: Resized overlay image.
    - int: New center X coordinate.
    - int: New center Y coordinate.
    """
    scale = target_size / max(overlay_img.shape[:2])
    overlay_resized = zoom(overlay_img, [scale, scale, 1], order=0)
    if overlay_center_x is not None and overlay_center_y is not None:
        new_center_x, new_center_y = int(overlay_center_x * scale), int(
            overlay_center_y * scale
        )
    else:
        new_center_x, new_center_y = (
            overlay_resized.shape[1] // 2,
            overlay_resized.shape[0] // 2,
        )
    return overlay_resized, new_center_x, new_center_y


def rotate_overlay(overlay_img, rotation, center_x, center_y):
    """
    Rotate the overlay image around a specified center point.

    Parameters:
    - overlay_img: np.array, Overlay image to rotate.
    - rotation: float, Rotation angle in degrees.
    - center_x: int, X coordinate of the center point for rotation.
    - center_y: int, Y coordinate of the center point for rotation.

    Returns:
    - np.array: Rotated overlay image.
    - int: Adjusted center X coordinate after rotation.
    - int: Adjusted center Y coordinate after rotation.
    """
    if rotation != 0.0:
        overlay_rotated = rotate(overlay_img, rotation, reshape=True, mode="nearest")
        # Recalculate the center as the image shape may change
        center_x, center_y = (
            overlay_rotated.shape[1] // 2,
            overlay_rotated.shape[0] // 2,
        )
    else:
        overlay_rotated = overlay_img
    return overlay_rotated, center_x, center_y


def adjust_alpha_channel(overlay_img, opacity, min_alpha):
    """
    Adjust the alpha channel of the overlay image based on the desired opacity and minimum alpha threshold.

    Parameters:
    - overlay_img: np.array, Overlay image array.
    - opacity: float, Desired opacity level [0.0, 1.0].
    - min_alpha: float, Minimum alpha value to ensure visibility.

    Returns:
    - np.array: Adjusted alpha channel.
    """
    alpha_channel = overlay_img[:, :, 3] / 255.0 * opacity
    alpha_channel = np.clip(alpha_channel, min_alpha, 1.0)
    return alpha_channel[..., np.newaxis]


def calculate_overlay_position(
    position: tuple,
    center_x: int,
    center_y: int,
    background_shape: tuple,
    overlay_shape: tuple,
) -> tuple:
    """
    Calculate the top-left corner position for overlay on a background image.

    Args:
    position (tuple): The desired center position on the background image.
    center_x (int): The x-coordinate of the center of the overlay.
    center_y (int): The y-coordinate of the center of the overlay.
    background_shape (tuple): A tuple containing the height and width of the background image.
    overlay_shape (tuple): A tuple containing the height and width of the overlay image.

    Returns:
    tuple: A tuple containing the x and y coordinates of the top-left corner position for the overlay.

    Raises:
    ValueError: If any of the input parameters are not valid (e.g., negative values).
    """

    # Desired center position on the background
    target_x, target_y = position

    # Calculate top-left corner position based on the desired center
    start_x = target_x - center_x
    start_y = target_y - center_y

    # Ensure the overlay does not start outside the background boundaries
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)

    # Ensure the overlay does not extend beyond the background's right or bottom edge
    background_height, background_width = background_shape[:2]
    overlay_height, overlay_width = overlay_shape[:2]

    if start_x + overlay_width > background_width:
        start_x = background_width - overlay_width
    if start_y + overlay_height > background_height:
        start_y = background_height - overlay_height

    # Prevent negative values which can occur if the overlay is larger than the background
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)

    return start_x, start_y


def blend_overlay(
    background_img, overlay_img, start_x, start_y, alpha_channel, gamma, overlay_bias
):
    """
    Blend the overlay image onto the background image at a specified position, considering the alpha channel,
    gamma correction, and overlay bias for blending.

    Parameters:
    - background_img: np.array, Background image array.
    - overlay_img: np.array, Overlay (transparent) image array.
    - start_x: int, X coordinate of the top-left corner to start overlaying.
    - start_y: int, Y coordinate of the top-left corner to start overlaying.
    - alpha_channel: np.array, Adjusted alpha channel for the overlay.
    - gamma: float, Gamma correction factor.
    - overlay_bias: float, Blending bias towards the overlay [0.0, 1.0].

    Returns:
    - np.array: Background image array with the overlay blended onto it.
    """
    # Extract the color channels of the overlay image
    overlay_color = overlay_img[:, :, :3]

    # Determine the region of the background to be blended with the overlay
    background_roi = background_img[
        start_y : start_y + overlay_color.shape[0],
        start_x : start_x + overlay_color.shape[1],
    ]

    # Apply gamma correction
    overlay_linear = np.power(overlay_color.astype(np.float32) / 255.0, gamma)
    background_linear = np.power(background_roi.astype(np.float32) / 255.0, gamma)

    # Compute the blended image
    alpha_foreground = alpha_channel * overlay_bias
    blended_linear = (
        alpha_foreground * overlay_linear + (1 - alpha_foreground) * background_linear
    )

    # Apply inverse gamma correction
    blended = np.power(blended_linear, 1 / gamma) * 255.0
    blended_clipped = np.clip(blended, 0, 255).astype(np.uint8)

    # Place the blended image back into the background image
    background_img[
        start_y : start_y + overlay_color.shape[0],
        start_x : start_x + overlay_color.shape[1],
    ] = blended_clipped

    return background_img


def overlay_transparent(
    background_img,
    overlay_img,
    position,
    size,
    rotation=0.0,
    opacity=1.0,
    gamma=1.0,
    overlay_bias=0.9,
    min_alpha=0.1,
    overlay_center_x=None,
    overlay_center_y=None,
):
    """
    Overlay a transparent image onto a background image with various transformations and adjustments.

    Parameters:
    - background_img: np.array, Background image array.
    - overlay_img: np.array, Overlay (transparent) image array.
    - position: tuple(int, int), Target position (x, y) for the overlay on the background.
    - size: int, Target size of the overlay image.
    - rotation: float, Rotation angle in degrees.
    - opacity: float, Opacity level of the overlay [0.0, 1.0].
    - gamma: float, Gamma correction factor.
    - overlay_bias: float, Blending bias towards the overlay [0.0, 1.0].
    - min_alpha: float, Minimum alpha value to ensure visibility of the overlay.
    - overlay_center_x: int, X coordinate of the overlay center point (optional).
    - overlay_center_y: int, Y coordinate of the overlay center point (optional).

    Returns:
    - np.array: Background image array with the overlay applied.
    """
    target_size = calculate_target_size(background_img, size)
    background_img_shape = background_img.shape
    overlay_resized, new_center_x, new_center_y = resize_and_center_overlay(
        overlay_img, target_size, overlay_center_x, overlay_center_y
    )
    rotation = random.uniform(-rotation, rotation)
    overlay_rotated, new_center_x, new_center_y = rotate_overlay(
        overlay_resized, rotation, new_center_x, new_center_y
    )
    alpha_channel = adjust_alpha_channel(overlay_rotated, opacity, min_alpha)
    start_x, start_y = calculate_overlay_position(
        position,
        new_center_x,
        new_center_y,
        background_img_shape,
        overlay_rotated.shape,
    )
    background_img = blend_overlay(
        background_img,
        overlay_rotated,
        start_x,
        start_y,
        alpha_channel,
        gamma,
        overlay_bias,
    )
    return background_img
