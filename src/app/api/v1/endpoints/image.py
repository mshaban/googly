from fastapi import File, Response, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io


from src.app.core.logger import logger


async def process_image(image: UploadFile = File(...)) -> StreamingResponse:
    """
    Process the uploaded image file by resizing it to 100x100 pixels and returning a streaming response.

    Parameters:
    - image (UploadFile): The uploaded image file to be processed.

    Returns:
    - StreamingResponse: A streaming response containing the processed image.

    Raises:
    - None

    This function reads the contents of the uploaded image file, resizes it to 100x100 pixels, and saves it in the original format.
    The processed image is then returned as a streaming response with the appropriate media type based on the original format.
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

        logger.debug(f"Processed image saved: {image.filename}")

        return StreamingResponse(
            img_byte_arr,
            media_type=f"image/{original_format.lower()}",
        )


async def googly(image: UploadFile = File(...)) -> Response:
    """
    Process the uploaded image to add googly eyes and return the modified image.

    Parameters:
    image (UploadFile): The image file to be processed.

    Returns:
    Response: The response containing the modified image with googly eyes.

    Raises:
    HTTPException: If there is an issue processing the image.
    """

    file_name = image.filename if image.filename else "processed_image.jpg"
    original_filename_parts = file_name.rsplit(".", 1)
    modified_filename = (
        f"{original_filename_parts[0]}_googlyed.{original_filename_parts[1]}"
        if len(original_filename_parts) == 2
        else f"{file_name}_googlyed"
    )

    logger.debug(f"Sending image for processing: {image.filename}")
    response = await process_image(image)
    logger.debug(f"Process Image received: {image.filename}")

    response.headers["Content-Disposition"] = (
        f"attachment; filename={modified_filename}"
    )
    return response
