from pydantic import BaseModel, validator
import io
from PIL import Image

from src.app.core.enums import ImageFormatEnum


class ImageModel(BaseModel):
    filename: str
    modified_filename: str | None
    data: bytes
    format: ImageFormatEnum

    @validator("format")
    def validate_known_format(cls, value):
        if value == ImageFormatEnum.UNKNOWN:
            raise ValueError("Unknown image format")
        return value
