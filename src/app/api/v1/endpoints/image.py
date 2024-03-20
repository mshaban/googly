import io

import aiohttp
import numpy as np
from src.app.core.enums import ImageFormatEnum
from src.app.utils.image_utils import (
    get_image_format_from_bytes,
    get_image_from_array,
    image_path_to_bytes,
)
from fastapi import File, UploadFile
from fastapi.responses import StreamingResponse

from src.app.core.logger import logger
from src.app.core.settings import settings
from src.app.models.googly import GooglyModel
from src.app.models.image import ImageModel
from src.app.utils.googlify_utils import apply_googly_eyes


async def create_image_model(image: UploadFile = File(...)) -> ImageModel:
    """
    Creates an ImageModel object from the provided UploadFile object.

    Parameters:
    - image (UploadFile): The UploadFile object containing the image data.

    Returns:
    - ImageModel: The ImageModel object created from the image data.

    Raises:
    - ValueError: If the image data is empty or invalid.
    """

    logger.debug(f"Creating image model: {image.filename}")

    # Read the contents of the image file
    contents = await image.read()
    image_bytes_io = io.BytesIO(contents).read()

    # Determine the file name and format of the image
    file_name = image.filename if image.filename else "processed_image.jpg"
    modified_filename = (
        f"{file_name.rsplit('.', 1)[0]}_googlyed.{file_name.rsplit('.', 1)[1]}"
        if "." in file_name
        else f"{file_name}_googlyed"
    )
    image_format = get_image_format_from_bytes(image_bytes_io)

    # Create the ImageModel object
    image_model = ImageModel(
        filename=file_name,
        data=image_bytes_io,
        format=image_format,
        modified_filename=modified_filename,
    )

    logger.debug(f"Image model created: {image_model.filename}")

    return image_model


async def process_image(image: ImageModel) -> np.ndarray:
    """
    Process an image using a Ray Serve deployment.

    Parameters:
    image (ImageModel): An instance of the ImageModel class representing the image to be processed.

    Returns:
    np.ndarray: An array representing the processed image.

    Raises:
    Exception: If the processing of the image fails with Ray Serve, an
    exception is raised with the corresponding status code.
    """
    ray_serve_url = (
        f"{settings.SERVE_HOST}:{settings.SERVE_PORT}/{settings.SERVE_ENDPOINT}"
    )

    googly_bytes = image_path_to_bytes(settings.GOOGLY_PATH)
    googly_model = ImageModel(
        data=googly_bytes,
        filename=settings.GOOGLY_PATH.split("/")[-1],
        format=ImageFormatEnum.PNG.value,
        modified_filename=None,
    )
    # Use aiohttp to send the serialized ImageModel to the Ray Serve deployment
    async with aiohttp.ClientSession() as session:
        async with session.post(
            ray_serve_url, json=image.model_dump_json()
        ) as response:
            if response.status == 200:
                # The response is assumed to be a JSON representation of the processed ImageModel
                processed_image_data_json = await response.json()
                face_detection_results, eye_detection_results = (
                    processed_image_data_json["face_detection_results"],
                    processed_image_data_json["eye_detection_results"],
                )

                googly_overlay = GooglyModel(
                    faces=face_detection_results,
                    eyes=eye_detection_results,
                    input_image=image,
                    googly=googly_model,
                )
                processed_image_model = apply_googly_eyes(googly_overlay)

                return processed_image_model
            else:
                raise Exception(
                    "Failed to process image with Ray Serve. Status code: {}".format(
                        response.status
                    )
                )


async def googly(image: UploadFile = File(...)):
    """
    Process an uploaded image file, apply a googly effect, and return the
    processed image as a streaming response.

    Parameters:
    - image (UploadFile): The uploaded image file to be processed.

    Returns:
    - StreamingResponse: A streaming response containing the processed image
    with a googly effect applied.

    Raises:
    - HTTPException: If there is an issue with processing the image or creating
    the streaming response.

    This function takes an uploaded image file, creates an image model,
    processes the image to apply a googly effect, and returns the processed
    image as a streaming response. The processed image is saved in memory as a
    BytesIO object and then returned with appropriate headers for downloading.

    """
    logger.debug(f"Creating image model: {image.filename}")
    image_model = await create_image_model(image)

    logger.debug(f"Processing image: {image.filename}")
    image_array = await process_image(image_model)

    logger.debug(f"Saving processed image: {image_model.modified_filename}")
    image_obj = get_image_from_array(image_array)

    logger.debug(f"Returning processed image: {image_model.modified_filename}")
    processed_bytes_io = io.BytesIO()
    image_obj.save(processed_bytes_io, format=image_model.format.value)
    processed_bytes_io.seek(0)

    headers = {
        "Content-Disposition": f'attachment; filename="{image_model.modified_filename}"'
    }
    return StreamingResponse(
        processed_bytes_io,
        media_type=f"image/{image_model.format.value.lower()}",
        headers=headers,
    )
