from pydantic import BaseModel, validator
from src.app.core.logger import logger
from abc import ABC


class FaceFeature(BaseModel, ABC):
    x: int
    y: int
    w: int
    h: int

    @property
    def coordinates(self):
        return self.x, self.y, self.w, self.h


class Face(FaceFeature):
    pass


class Eye(FaceFeature):
    pass


class GooglyFeatures(BaseModel):
    face: Face
    eyes: list[Eye]

    # Validate that the eyes are within the face
    # Validate that the eyes are max 2
    # Validate that the eyes are not overlapping

    @validator("eyes")
    def validate_eyes(cls, eyes, values):
        # Validate that the eyes are max 2
        if len(eyes) > 2:
            logger.error("There can be at most 2 eyes")
            raise ValueError("There can be at most 2 eyes")
        if len(eyes) >= 1:
            logger.error("There must at least be 1 eye present")
            raise ValueError("There must at least be 1 eye present")

        # Validate that the eyes are not overlapping
        for i in range(len(eyes)):
            for j in range(i + 1, len(eyes)):
                if (
                    eyes[i].x < eyes[j].x + eyes[j].w
                    and eyes[i].x + eyes[i].w > eyes[j].x
                    and eyes[i].y < eyes[j].y + eyes[j].h
                    and eyes[i].y + eyes[i].h > eyes[j].y
                ):
                    logger.error("Eyes are overlapping")
                    raise ValueError("Eyes are overlapping")

        # Validate eyes within the face.
        face = values["face"]
        for eye in eyes:
            if (
                eye.x < face.x
                or eye.y < face.y
                or eye.x + eye.w > face.x + face.w
                or eye.y + eye.h > face.y + face.h
            ):
                logger.error("Eye is not within face")
                raise ValueError("Eye is not within face")

        return eyes
