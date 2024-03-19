from abc import ABC

from pydantic import BaseModel


class FaceFeature(BaseModel, ABC):
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def coordinates(self):
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def width(self):
        return ((self.xmax - self.xmin) ** 2) ** 0.5

    @property
    def height(self):
        return ((self.ymax - self.ymin) ** 2) ** 0.5

    @property
    def size(self):
        return max(self.width, self.height)


class FaceModel(FaceFeature):
    pass


class PointModel(BaseModel):
    x: int
    y: int


class EyeModel(BaseModel):
    points: list[PointModel]
