from fastapi import File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io

from src.app.models.image import ImageModel
from src.app.core.logger import logger


async def process_image(image: UploadFile = File(...)) -> ImageModel:
    """
    Process the uploaded image by resizing it to 100x100 pixels and returning the processed image as a BinaryDataModel.

    Parameters:
    image (UploadFile): The uploaded image file to be processed.

    Returns:
    ImageModel: The processed image data as a BinaryDataModel.

    Raises:
    None
    """
    contents = await image.read()

    logger.debug(f"Processing image: {image.filename}")
    image_bytes_io = io.BytesIO(contents)

    with Image.open(image_bytes_io) as img:
        original_format = img.format if img.format else "JPEG"
        img = img.resize((100, 100))

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=original_format)
        img_byte_arr.seek(0)

        binary_data_model = ImageModel(data=img_byte_arr.read(), format=original_format)

        logger.debug(f"Processed image saved: {image.filename}")

        return binary_data_model


async def googly(image: UploadFile = File(...)):
    """
    Apply a "googly eyes" effect to the uploaded image and return the modified image as a streaming response.

    Parameters:
    image (UploadFile): The uploaded image file to apply the effect on.

    Returns:
    StreamingResponse: The modified image with the "googly eyes" effect as a streaming response.

    Raises:
    None
    """
    logger.debug(f"Processing image: {image.filename}")
    file_name = image.filename if image.filename else "processed_image.jpg"
    modified_filename = (
        f"{file_name.rsplit('.', 1)[0]}_googlyed.{file_name.rsplit('.', 1)[1]}"
        if "." in file_name
        else f"{file_name}_googlyed"
    )

    image_model = await process_image(image)
    img_byte_arr = io.BytesIO(image_model.data)

    headers = {"Content-Disposition": f'attachment; filename="{modified_filename}"'}
    return StreamingResponse(
        img_byte_arr, media_type=f"image/{image_model.format.lower()}", headers=headers
    )
