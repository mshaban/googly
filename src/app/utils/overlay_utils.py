import random

import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate


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
):
    """
    Overlay a transparent image onto a background image with options for position, size,
    rotation, opacity, gamma correction, blending bias, and minimum alpha threshold.

    Parameters:
    - background_img: Background image array.
    - overlay_img: Overlay (transparent) image array.
    - position: Tuple (x, y) for the overlay position on the background.
    - size: Target size of the overlay.
    - rotation: Rotation angle of the overlay in degrees.
    - opacity: Opacity level of the overlay [0.0, 1.0].
    - gamma: Gamma correction factor.
    - overlay_bias: Blending bias towards the overlay [0.0, 1.0].
    - min_alpha: Minimum alpha value to ensure visibility.
    """
    x, y = position
    background_height, background_width = background_img.shape[:2]
    target_size = min(size, background_width, background_height)
    target_size = int(target_size * random.uniform(1.1, 1.4))

    # Resize and rotate overlay
    scale_x = scale_y = target_size / max(overlay_img.shape[:2])
    overlay_resized = zoom(overlay_img, [scale_y, scale_x, 1], order=0)

    if rotation != 0.0:
        overlay_resized = rotate(
            overlay_resized, rotation, reshape=True, mode="nearest"
        )

    # Adjust the alpha channel for opacity and min alpha threshold
    alpha_channel = np.clip(overlay_resized[:, :, 3] / 255.0 * opacity, min_alpha, 1.0)
    overlay_color = overlay_resized[:, :, :3]

    # Determine the region of interest on the background
    start_x, start_y = max(x - target_size // 2, 0), max(y - target_size // 2, 0)
    end_x, end_y = min(start_x + overlay_color.shape[1], background_width), min(
        start_y + overlay_color.shape[0], background_height
    )

    background_roi = background_img[start_y:end_y, start_x:end_x]

    # Apply blending with gamma correction and bias
    overlay_linear = np.power(overlay_color.astype(np.float32) / 255.0, gamma)
    background_linear = np.power(background_roi.astype(np.float32) / 255.0, gamma)

    alpha_foreground = alpha_channel[..., np.newaxis] * overlay_bias
    alpha_background = 1 - alpha_foreground

    blended_linear = (
        alpha_foreground * overlay_linear + alpha_background * background_linear
    )
    blended = np.power(blended_linear, 1 / gamma) * 255.0
    blended_clipped = np.clip(blended, 0, 255).astype(np.uint8)

    # Place the blended ROI back into the background image
    background_img[start_y:end_y, start_x:end_x] = blended_clipped

    return background_img
