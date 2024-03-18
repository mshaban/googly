def save_image_from_bytes(image_bytes: bytes, file_path: str) -> None:
    """
    Saves image bytes to a file.

    Args:
        image_bytes: The bytes object containing the image data.
        file_path: The path where the image should be saved, including the filename and extension.
    """
    with open(file_path, "wb") as image_file:
        image_file.write(image_bytes)
