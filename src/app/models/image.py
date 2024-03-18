from pydantic import BaseModel
import io
from PIL import Image


class ImageModel(BaseModel):
    filename: str
    modified_filename: str | None
    data: bytes
    format: str = "JPEG"
