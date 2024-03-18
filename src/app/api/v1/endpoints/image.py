from fastapi import File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io

from src.app.models.image import ImageModel
from src.app.utils.image_utils import create_image_model
from src.app.core.logger import logger


async def process_image(image: ImageModel) -> ImageModel:
    """
    Process the uploaded image by resizing it to 100x100 pixels and returning the processed image as a BinaryDataModel.

    Parameters:
    image (UploadFile): The uploaded image file to be processed.

    Returns:
    ImageModel: The processed image data as a BinaryDataModel.

    Raises:
    None
    """
    image_bytes, original_format = image.data, image.format
    image_bytes_io = io.BytesIO(image_bytes)

    with Image.open(image_bytes_io) as img:
        img = img.resize((100, 100))

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=original_format)
        img_byte_arr.seek(0)

        image_model = ImageModel(
            data=img_byte_arr.read(),
            format=original_format,
            filename=(
                image.modified_filename
                if image.modified_filename
                else "processed_image.jpg"
            ),
            modified_filename=None,
        )

        logger.debug(f"Processed image saved: {image.filename}")

        return image_model


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

    logger.debug(f"Creating image model: {image.filename}")

    image_model = await create_image_model(image)
    processed_image = await process_image(image_model)
    processed_bytes_io = io.BytesIO(processed_image.data)

    headers = {
        "Content-Disposition": f'attachment; filename="{processed_image.filename}"'
    }
    return StreamingResponse(
        processed_bytes_io,
        media_type=f"image/{image_model.format.lower()}",
        headers=headers,
    )
