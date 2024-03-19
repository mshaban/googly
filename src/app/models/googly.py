from pydantic import BaseModel, model_validator

from src.app.models.features import EyeModel, FaceModel
from src.app.models.image import ImageModel


class GooglyModel(BaseModel):
    faces: list[FaceModel]
    eyes: list[EyeModel]
    input_image: ImageModel
    googly: ImageModel

    @model_validator(mode="before")
    def validate_eyes_faces(cls, values):
        """
        Validates the relationship between eyes and faces in a given set of values.

        Parameters:
        cls (class): The class to which this method belongs.
        values (dict): A dictionary containing the values to be validated, with keys "eyes" and "faces".

        Returns:
        dict: The validated values dictionary.

        Raises:
        ValueError: If the number of eyes is not twice the number of faces, or if any pair of eyes is not within their respective face.
        """

        eyes, faces = values["eyes"], values["faces"]

        # Check if the number of eyes is twice the number of faces
        if len(eyes) != 2 * len(faces):
            raise ValueError("Mismatch between number of eyes and faces")

        # Validate that each pair of eyes is within their respective face in the list
        for i in range(0, len(eyes) - 1, 2):
            face_model = faces[i // 2]
            left_eye, right_eye = eyes[i], eyes[i + 1]

            # Check if both eyes are within the boundaries of the face
            if not all(
                [
                    left_eye.points[0].x >= face_model.xmin,
                    left_eye.points[0].y >= face_model.ymin,
                    left_eye.points[1].x <= face_model.xmax,
                    left_eye.points[1].y <= face_model.ymax,
                    right_eye.points[0].x >= face_model.xmin,
                    right_eye.points[0].y >= face_model.ymin,
                    right_eye.points[1].x <= face_model.xmax,
                    right_eye.points[1].y <= face_model.ymax,
                ]
            ):
                raise ValueError("Eyes are not within the face")

        return values

    def model_post_init(self, *args, **kwargs):
        self.eyes = [
            EyeModel(points=self.normalize_eye_points(eye)) for eye in self.eyes
        ]

        super().model_post_init(*args, **kwargs)

    def normalize_eye_points(self, eye_model):
        """
        Sorts the eye points in the given eye model by their x-coordinate in ascending order.

        Args:
        eye_model (EyeModel): The EyeModel object containing the points to be normalized.

        Returns:
        list: A list of Point objects sorted by their x-coordinate in ascending order.

        Raises:
        TypeError: If the input eye_model is not of type EyeModel.
        """

        if not isinstance(eye_model, EyeModel):
            raise TypeError("Input must be of type EyeModel")

        sorted_points = sorted(eye_model.points, key=lambda point: point.x)
        return sorted_points
