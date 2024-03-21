from unittest.mock import MagicMock
from src.app.utils.eye_utils import (
    estimate_eye_size,
    adjust_size_based_on_proportions,
    calculate_googly_center_and_size,
)

from src.app.models.features import EyeModel, PointModel


import unittest


class TestEstimateEyeSize(unittest.TestCase):

    def test_estimate_eye_size_with_empty_model(self):
        empty_model = EyeModel(points=[])
        with self.assertRaises(ValueError):
            estimate_eye_size(empty_model)

    def test_estimate_eye_size_with_too_few_points(self):
        model = EyeModel(points=[PointModel(x=0, y=0)])
        with self.assertRaises(ValueError):
            estimate_eye_size(model)

    def test_estimate_eye_size_with_valid_model(self):
        model = EyeModel(
            points=[PointModel(x=0, y=0), PointModel(x=3, y=4), PointModel(x=6, y=8)]
        )
        self.assertEqual(estimate_eye_size(model), 10.0)


class TestAdjustSizeBasedOnProportions(unittest.TestCase):

    def test_adjust_size_based_on_proportions(self):
        face_model = MagicMock()
        face_model.size = 100
        image_size = (800, 600)
        adjusted_size = adjust_size_based_on_proportions(50, face_model, image_size)
        self.assertEqual(adjusted_size, 30)

    def test_adjust_size_based_on_proportions_large_image(self):
        face_model = MagicMock()
        face_model.size = 80
        image_size = (1600, 1200)
        adjusted_size = adjust_size_based_on_proportions(30, face_model, image_size)
        self.assertEqual(adjusted_size, 21)


class TestCalculateGooglyCenterAndSize(unittest.TestCase):

    def test_invalid_image_size(self):
        with self.assertRaises(ValueError):
            calculate_googly_center_and_size(MagicMock(), MagicMock(), (100, 200, 300))

    def test_corrected_center_outside_face_region(self):
        eye_model = MagicMock()
        face_model = MagicMock()
        face_model.xmin = 0
        face_model.xmax = 100
        eye_model.calculate_eye_center.return_value = (150, 50)

        with self.assertRaises(ValueError):
            calculate_googly_center_and_size(eye_model, face_model, (200, 200))
