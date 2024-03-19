import cv2

from src.app.models.features import EyeModel, FaceModel


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
        for p in l.points:
            x, y = p.x, p.y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    # cv2.imshow("Facial Landmarks", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
