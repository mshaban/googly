from pydantic import BaseModel, validator
import numpy as np
import cv2

from src.app.core.enums import ImageFormatEnum


class ImageModel(BaseModel):
    filename: str
    data: bytes
    modified_filename: str | None
    format: ImageFormatEnum

    @validator("format")
    def validate_known_format(cls, value):
        if value == ImageFormatEnum.UNKNOWN:
            raise ValueError("Unknown image format")
        return value

    @property
    def image(self):
        image_array = np.frombuffer(self.data, np.uint8)
        # Decode the array into an image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
