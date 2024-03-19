import random

import cv2
import numpy as np


def apply_rotation(img, angle, size=None):
    """
    Apply rotation to an image around its center.

    Parameters:
    - img: The image to be rotated.
    - angle: The rotation angle in degrees. Positive values mean counter-clockwise rotation.
    - size: The size of the image (used if the image needs to be resized before rotation).

    Returns:
    - The rotated image.
    """
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # Calculate the center of the image for rotation
    center = (img.shape[1] // 2, img.shape[0] // 2)

    # Get the rotation matrix for the calculated angle around the image center
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated_img = cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    return rotated_img


def interpolate_image(img, target_size):
    """
    Resize an image to a target size using appropriate interpolation.
    """
    # Determine the interpolation method based on the target size relative to the original size
    interpolation = (
        cv2.INTER_AREA
        if (target_size, target_size) < img.shape[:2]
        else cv2.INTER_LANCZOS4
    )
    resized_img = cv2.resize(
        img, (target_size, target_size), interpolation=interpolation
    )
    return resized_img


def blend_with_gamma_correction(overlay, background, alpha_channel, gamma):
    """
    Blend the overlay and the background using gamma correction.

    Parameters:
    - overlay: The overlay image after being resized and potentially rotated.
    - background: The background image's region of interest where the overlay will be placed.
    - alpha_channel: The alpha channel of the overlay, adjusted for opacity.
    - gamma: The gamma value for gamma correction to ensure proper blending, especially in lighting conditions.

    Returns:
    - The blended image ready to be placed back into the larger background image.
    """
    # Convert images to linear color space for accurate blending
    overlay_linear = np.power(overlay.astype(np.float32) / 255.0, gamma)
    background_linear = np.power(background.astype(np.float32) / 255.0, gamma)

    # Calculate blending
    alpha_foreground = alpha_channel[
        ..., np.newaxis
    ]  # Ensure alpha channel has correct shape for broadcasting
    blended_linear = (
        alpha_foreground * overlay_linear + (1 - alpha_foreground) * background_linear
    )

    # Convert back to standard color space
    blended = np.power(blended_linear, 1 / gamma) * 255.0
    blended_clipped = np.clip(blended, 0, 255).astype(
        np.uint8
    )  # Ensure values are within byte range

    return blended_clipped


def overlay_transparent(
    background_img,
    overlay_t,
    position,
    target_size,
    rotation=0.0,
    opacity=1.0,
    gamma=2.2,
):
    """
    Overlay a transparent image onto a background image at a specified position and size, with optional rotation, opacity, and gamma correction.
    """
    x, y = position
    background_height, background_width = background_img.shape[:2]

    # Ensure the target size is within background bounds and adjust for variety
    target_size = min(target_size, background_width, background_height)
    target_size = int(target_size * random.uniform(1.1, 1.4))

    # Resize the overlay image
    overlay_resized = interpolate_image(overlay_t, target_size)

    # Adjust the alpha channel for opacity
    alpha_channel = overlay_resized[:, :, 3] / 255.0 * opacity
    overlay_color = overlay_resized[:, :, :3]

    # Calculate the overlay placement coordinates
    start_x, start_y = max(x - target_size // 2, 0), max(y - target_size // 2, 0)
    end_x, end_y = min(start_x + target_size, background_width), min(
        start_y + target_size, background_height
    )

    # Ensure overlay fits within the background dimensions
    overlay_color = overlay_color[: end_y - start_y, : end_x - start_x]
    alpha_channel = alpha_channel[: end_y - start_y, : end_x - start_x]

    # Prepare the region of interest from the background image
    background_roi = background_img[start_y:end_y, start_x:end_x]

    # Apply rotation if needed
    if rotation != 0.0:
        overlay_color = apply_rotation(overlay_color, rotation, size=target_size)

    # Blend overlay with the background using linear gamma correction for better blending
    blended_roi = blend_with_gamma_correction(
        overlay_color, background_roi, alpha_channel, gamma
    )

    # Place the blended ROI back into the background image
    background_img[start_y:end_y, start_x:end_x] = blended_roi

    return background_img
