from pydantic import BaseModel, validator
import cv2
from src.app.core.logger import logger
from abc import ABC

from src.app.models.image import ImageModel


class FaceFeature(BaseModel, ABC):
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def coordinates(self):
        return self.xmin, self.ymin, self.xmax, self.ymax


class FaceModel(FaceFeature):
    pass


class EyeModel(FaceFeature):
    pass


class GooglyFeatures(BaseModel):
    face: FaceModel
    eyes: list[EyeModel]
    input_image: ImageModel
    googly_image: ImageModel

    @validator("eyes")
    def validate_eyes(cls, eyes, values):
        # Validate that the eyes are max 2
        if len(eyes) != 2:
            logger.error("Number of eyes is not 2")
            raise ValueError("Number of eyes is not 2")

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

    def overlay_transparent(self, overlay_size=None):
        """
        Overlay a transparent image `img_to_overlay_t` onto another image `background_img` at position (x, y)
        :param overlay_size: The size to scale the overlay (width, height).
        """
        bg_img = self.input_image.image.copy()
        img_to_overlay_t = self.googly_image.image.copy()
        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        # Extract the alpha mask of the RGBA image, convert to RGB
        b, g, r, a = cv2.split(img_to_overlay_t)
        overlay_color = cv2.merge((b, g, r))

        # Apply some simple filtering to remove edges from the alpha channel
        mask = cv2.medianBlur(a, 5)

        for ey in self.eyes:
            xmin, ymin, xmax, ymax = ey.coordinates
            roi = bg_img[ymin:ymax, xmin:xmax]

            # Black-out the area behind the overlay in the ROI
            img1_bg = cv2.bitwise_and(
                roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask)
            )

            # Mask out the overlay from the overlay image.
            img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

            # Update the original image with our new ROI
            bg_img[ymin:ymax, xmin:ymax] = cv2.add(img1_bg, img2_fg)

        return bg_img
