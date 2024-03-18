from fastapi import File, UploadFile
import io

from src.app.models.image import ImageModel
from src.app.core.logger import logger


def save_image_from_bytes(image_bytes: bytes, file_path: str) -> None:
    """
    Saves image bytes to a file.

    Args:
        image_bytes: The bytes object containing the image data.
        file_path: The path where the image should be saved, including the filename and extension.
    """
    with open(file_path, "wb") as image_file:
        image_file.write(image_bytes)


def get_image_format_from_bytes(image_bytes: bytes) -> str:
    """Determines the image format by checking the magic number in the bytes array.

    Args:
        image_bytes: A bytes array containing the image data.

    Returns:
        A string representing the image format (e.g., 'JPEG', 'PNG'), or 'Unknown' if the format is not recognized.
    """
    # Magic numbers for different image formats
    # Each tuple contains the byte signature and its associated format
    signatures = [
        (b"\xFF\xD8\xFF", "JPEG"),
        (b"\x89PNG\r\n\x1A\n", "PNG"),
        (b"GIF87a", "GIF"),
        (b"GIF89a", "GIF"),
        (b"BM", "BMP"),
        (b"II*\x00", "TIFF"),
        (b"MM\x00*", "TIFF"),
        (b"\x00\x00\x01\x00", "ICO"),
    ]

    # Only need to check the first 16 bytes
    file_header = image_bytes[:16]

    for signature, format_name in signatures:
        if file_header.startswith(signature):
            return format_name

    return "Unknown"


async def create_image_model(image: UploadFile = File(...)) -> ImageModel:
    """
    Asynchronously creates an ImageModel object from the provided UploadFile object.

    Parameters:
    - image (UploadFile): The UploadFile object containing the image data.

    Returns:
    - ImageModel: The ImageModel object created from the image data.

    Raises:
    - ValueError: If the image data is empty or cannot be read.
    - IOError: If there is an issue reading the image data.

    This function reads the contents of the provided UploadFile object, creates an ImageModel object with the image data,
    and returns it. It also logs the size of the image data at different stages of processing.
    """
    contents = await image.read()
    logger.debug(f"image size: {len(contents)}")
    image_bytes_io = io.BytesIO(contents).read()
    logger.debug(f"image size: {len(image_bytes_io)}")
    file_name = image.filename if image.filename else "processed_image.jpg"
    modified_filename = (
        f"{file_name.rsplit('.', 1)[0]}_googlyed.{file_name.rsplit('.', 1)[1]}"
        if "." in file_name
        else f"{file_name}_googlyed"
    )
    image_format = get_image_format_from_bytes(image_bytes_io)
    logger.debug(f"image size: {len(image_bytes_io)}")
    image_model = ImageModel(
        filename=file_name,
        data=image_bytes_io,
        format=image_format,
        modified_filename=modified_filename,
    )

    return image_model
