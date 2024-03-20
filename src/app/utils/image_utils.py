import io

import numpy as np
from PIL import Image

from src.app.core.enums import ImageFormatEnum


def save_image_from_bytes(image_bytes: bytes, file_path: str) -> None:
    """
    Saves image bytes to a file.

    Args:
        image_bytes: The bytes object containing the image data.
        file_path: The path where the image should be saved, including the filename and extension.
    """
    with open(file_path, "wb") as image_file:
        image_file.write(image_bytes)


def get_image_from_array(image_array: np.ndarray) -> Image.Image:
    """
    Saves an image array to a file, converting from BGR to RGB format if necessary.

    Args:
        image_array: The NumPy array containing the image data in BGR format.
        file_path: The path where the image should be saved, including the filename and extension.
    """

    # Check if the image has 3 channels (color image)
    if image_array.ndim == 3 and image_array.shape[2] == 3:
        # Convert from BGR to RGB
        image_array = image_array[:, :, [2, 1, 0]]

    # Ensure the array is of type uint8
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    image = Image.fromarray(image_array)
    return image


def get_image_format_from_bytes(image_bytes: bytes) -> ImageFormatEnum:
    """Determines the image format by checking the magic number in the bytes array.

    Args:
        image_bytes: A bytes array containing the image data.

    Returns:
        A string representing the image format (e.g., 'JPEG', 'PNG'), or 'Unknown' if the format is not recognized.
    """
    # Magic numbers for different image formats
    # Each tuple contains the byte signature and its associated format
    signatures = [
        (b"\xFF\xD8\xFF", ImageFormatEnum.JPEG),
        (b"\x89PNG\r\n\x1A\n", ImageFormatEnum.PNG),
        (b"GIF87a", ImageFormatEnum.GIF),
        (b"GIF89a", ImageFormatEnum.GIF),
        (b"BM", ImageFormatEnum.BMP),
        (b"II*\x00", ImageFormatEnum.TIFF),
        (b"MM\x00*", ImageFormatEnum.TIFF),
        (b"\x00\x00\x01\x00", ImageFormatEnum.ICO),
    ]

    # Only need to check the first 16 bytes
    file_header = image_bytes[:16]

    for signature, format_name in signatures:
        if file_header.startswith(signature):
            return format_name

    return ImageFormatEnum.UNKNOWN


def load_image_from_bytes(image_bytes: bytes, with_alpha: bool = False) -> np.ndarray:
    """
    Load an image from bytes and convert it to a NumPy array in OpenCV format.

    Parameters:
    image_bytes (bytes): The bytes representing the image data.
    with_alpha (bool): Flag indicating whether to include an alpha channel. Default is False.

    Returns:
    np.ndarray: The image data as a NumPy array in OpenCV format (BGR).

    Raises:
    ValueError: If the image_bytes parameter is not of type bytes.
    IOError: If there is an error loading the image from the bytes.

    Example:
    >>> image_data = load_image_from_bytes(image_bytes, with_alpha=True)
    """

    if not isinstance(image_bytes, bytes):
        raise ValueError("The image_bytes parameter must be of type bytes.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        if with_alpha:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
        open_cv_image = np.array(image)[:, :, ::-1].copy()
        return open_cv_image
    except IOError as e:
        raise IOError("Error loading image from bytes: {}".format(str(e))) from e


def image_path_to_bytes(image_path: str) -> bytes:
    """
    Reads an image file from the given path and returns its content as a bytes array.

    :param image_path: The file system path to the image file.
    :return: The image file's content as a bytes array.
    """
    return open(image_path, "rb").read()
