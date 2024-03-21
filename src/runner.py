from src.app.core.settings import settings
import httpx

import argparse
import numpy as np
import asyncio
from pathlib import Path

from src.app.utils.file_utils import find_images


URL = f"http://{settings.FASTAPI.FAST_HOST}:{settings.FASTAPI.FAST_PORT}/{settings.FASTAPI.FAST_ENDPOINT}"


async def test_googly_effect(image_path: str):
    """
    Apply a googly effect to an image by sending it to a specified URL for processing.

    Parameters:
    image_path (str): The path to the image file to be processed.
    url (str): The URL of the service that will apply the googly effect to the image.

    Raises:
    FileNotFoundError: If the specified image file does not exist.
    IOError: If there is an issue reading or writing the image file.
    httpx.HTTPStatusError: If the HTTP request to the processing service returns a non-200 status code.

    Returns:
    None
    """
    # Ensure the file exists
    file = Path(image_path)
    if not file.is_file():
        raise FileNotFoundError(f"The file {image_path} does not exist.")

    # Open the file in binary read mode
    files = {"image": (file.name, open(file, "rb"), "image/jpeg")}
    print(URL)
    async with httpx.AsyncClient() as client:
        # Send the POST request with the file
        response = await client.post(URL, files=files)

        if response.status_code == 200:
            # Get the suggested filename from the content-disposition header
            filename = "out/processed_image.jpg"
            cd = response.headers.get("content-disposition")
            if cd:
                filename = cd.split("filename=")[1].strip('"')
                filename = f"out/{filename}"

            # Save the received image to disk
            with open(filename, "wb") as f_out:
                f_out.write(response.content)
            print(f"Processed image saved as {filename}.")
        else:
            raise httpx.HTTPStatusError(
                f"Failed to process image. Status code: {response.status_code}"
            )


def get_sample_images(images_path, sample_size=5):

    if not Path(images_path).is_dir():
        raise FileNotFoundError(f"The directory {images_path} does not exist.")
    images = find_images(images_path)
    images = np.random.choice(images, sample_size) if sample_size else images
    im_files = []
    for im in images:
        im = f"{images_path}/{im}"
        im_files.append(im)
    return im_files


def main():

    parser = argparse.ArgumentParser(description="Apply googly eyes effect to images.")
    parser.add_argument(
        "images_dir", type=str, help="Directory containing images to process"
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=float("inf"),
        help="Number of images to sample. If not specified, all images are used.",
    )

    args = parser.parse_args()

    images = get_sample_images(args.images_dir, args.sample_size)

    for im in images:
        asyncio.run(test_googly_effect(im))


if __name__ == "__main__":
    main()
