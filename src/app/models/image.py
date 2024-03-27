from pydantic import BaseModel, validator
from pydantic import Base64Bytes

import base64

from src.app.core.enums import ImageFormatEnum


class ImageModel(BaseModel):
    filename: str
    data: Base64Bytes
    modified_filename: str | None
    format: ImageFormatEnum

    class Config:
        json_encoders = {
            bytes: lambda v: base64.b64encode(v).decode("utf-8"),
        }

    @validator("data", pre=True, always=True)
    def convert_bytes_to_base64(cls, v):
        if isinstance(v, bytes):
            return base64.b64encode(v)
        return v

    @validator("data", pre=False, always=True)
    def convert_base64_to_bytes(cls, v):
        if isinstance(v, str):
            return base64.b64decode(v)
        return v

    @validator("format")
    def validate_known_format(cls, value):
        if value == ImageFormatEnum.UNKNOWN:
            raise ValueError("Unknown image format")
        return value

    def model_post_init(self, *args, **kwargs):
        self.modified_filename = f"googly_{self.filename}"
        super().model_post_init(*args, **kwargs)
