from pydantic import BaseModel, validator
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

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

    def model_post_init(self, *args, **kwargs):
        self.modified_filename = f"googly_{self.filename}"
        super().model_post_init(*args, **kwargs)
