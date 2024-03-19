from pydantic import BaseModel

from src.app.models.features import EyeModel, FaceModel
from src.app.models.image import ImageModel
from src.app.utils.image_utils import load_image_from_bytes
from src.app.utils.overlay_utils import (
    overlay_transparent,
)

from src.app.utils.eye_utils import (
    calculate_eye_rotation,
    calculate_googly_center_and_size,
)


class GooglyModel(BaseModel):
    face: FaceModel
    eyes: list[EyeModel]
    input_image: ImageModel
    googly0: ImageModel
    googly1: ImageModel

    def apply_googly_eyes(self):
        """
        Apply googly eyes to the input image.

        This function takes the input image and overlays googly eyes on the detected eye regions.
        It first loads the input image and the googly eye images from bytes.
        It then iterates over the detected eye regions, calculates the center and size of the googly eye,
        and the rotation angle based on the eye coordinates.
        The googly eye image is then overlaid on the input image at the calculated center with the specified size and rotation.
        The opacity, gamma, overlay bias, and minimum alpha values can also be adjusted for the overlay.

        Parameters:
        - self: The instance of the class containing the input image, googly eye images, and eye regions.

        Returns:
        - input_img: The modified image with googly eyes overlaid on the detected eye regions.

        Raises:
        - ValueError: If the input image or googly eye images cannot be loaded.
        - IndexError: If there are not enough googly eye images for all detected eye regions.
        """
        # Convert input image from bytes to an OpenCV image
        input_img = load_image_from_bytes(self.input_image.data)

        input_image_size = input_img.shape[:2]

        # Load googly eye images
        googly_eye_imgs = [
            load_image_from_bytes(self.googly0.data, with_alpha=True),
            load_image_from_bytes(self.googly1.data, with_alpha=True),
        ]

        for i, eye in enumerate(self.eyes):
            googly_eye_img = googly_eye_imgs[i % len(googly_eye_imgs)]
            if i % 2 == 0:
                eye.xmin, eye.ymin, eye.xmax, eye.ymax = (
                    eye.xmax,
                    eye.ymax,
                    eye.xmin,
                    eye.ymin,
                )
            center, size = calculate_googly_center_and_size(
                eye, self.face, input_image_size, i % 2 == 0
            )
            rotation = calculate_eye_rotation(eye.coordinates[:2], eye.coordinates[2:])
            input_img = overlay_transparent(
                input_img,
                googly_eye_img,
                center,
                size,
                rotation=rotation,
                opacity=1.1,
                gamma=2.2,
            )

        # Optionally, save or further process the modified image
        return input_img
