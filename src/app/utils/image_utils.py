import cv2
from fastapi import File, UploadFile
import io

from src.app.models.image import ImageModel
from src.app.core.enums import ImageFormatEnum
from src.app.core.logger import logger

from src.app.models.features import FaceModel, EyeModel


def save_image_from_bytes(image_bytes: bytes, file_path: str) -> None:
    """
    Saves image bytes to a file.

    Args:
        image_bytes: The bytes object containing the image data.
        file_path: The path where the image should be saved, including the filename and extension.
    """
    with open(file_path, "wb") as image_file:
        image_file.write(image_bytes)


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
    logger.debug(f"Creating image model: {image.filename}")
    contents = await image.read()
    image_bytes_io = io.BytesIO(contents).read()
    file_name = image.filename if image.filename else "processed_image.jpg"
    modified_filename = (
        f"{file_name.rsplit('.', 1)[0]}_googlyed.{file_name.rsplit('.', 1)[1]}"
        if "." in file_name
        else f"{file_name}_googlyed"
    )
    image_format = get_image_format_from_bytes(image_bytes_io)
    image_model = ImageModel(
        filename=file_name,
        data=image_bytes_io,
        format=image_format,
        modified_filename=modified_filename,
    )
    logger.debug(f"Image model created: {image_model.filename}")
    return image_model


def image_path_to_bytes(image_path: str) -> bytes:
    """
    Reads an image file from the given path and returns its content as a bytes array.

    :param image_path: The file system path to the image file.
    :return: The image file's content as a bytes array.
    """
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    return image_bytes


def draw_annotations(
    img, eye_coordinates: list[EyeModel], face_coordinates: list[FaceModel]
):
    """
    Draw rectangles around detected faces and circles around detected eyes in the given image.

    Parameters:
    image_path (str): The file path of the image to process.
    eye_coordinates (list[EyeModel]): A list of EyeModel objects containing the coordinates of detected eyes.
    face_coordinates (list[FaceModel]): A list of FaceModel objects containing the coordinates of detected faces.

    Returns:
    None

    Raises:
    None

    This function loads the image specified by image_path, draws rectangles
    around the detected faces using the coordinates provided in
    face_coordinates, and draws circles around the detected eyes using the
    coordinates provided in eye_coordinates. The faces are outlined in green
    and the eyes are circled in blue. The processed image is displayed in a
    window titled "Faces and Eyes Detected" until a key is pressed, at which
    point the window is closed.
    """
    # Draw rectangles around faces
    for fc in face_coordinates:
        xmin, ymin, xmax, ymax = fc.coordinates
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Draw in green

    # # Draw circles around eyes
    # for ey in eye_coordinates:
    #     xmin, ymin, xmax, ymax = ey.coordinates
    #
    #     center = (xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2)
    #     radius = max(xmax - xmin, ymin - ymax) // 5
    #     cv2.circle(img, center, radius, (255, 0, 0), 2)  # Draw in blue
    visualize_landmarks(img, eye_coordinates)
    # Display the image
    cv2.imshow("Faces and Eyes Detected", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_landmarks(image, landmarks):
    """
    Visualizes facial landmarks on the given image.

    Parameters:
    image (numpy.ndarray): The input image on which to visualize the landmarks.
    results (numpy.ndarray): The array of facial landmarks detected in the image.

    Returns:
    None

    Raises:
    None
    """
    for l in landmarks:
        x, y = l.xmin, l.ymin
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        x, y = l.xmax, l.ymax
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    # cv2.imshow("Facial Landmarks", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
