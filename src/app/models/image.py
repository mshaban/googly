from pydantic import BaseModel
import io
from PIL import Image


class ImageModel(BaseModel):
    data: bytes
    format: str = "JPEG"
