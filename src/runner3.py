import cv2
from openvino.runtime import Core

from src.app.core.enums import ImageFormatEnum
from src.app.models.googly import GooglyModel
from src.app.models.image import ImageModel
from src.app.models.model_registery import ModelArtifacts, ModelRegistry
from src.app.models.vino_models import VinoEyeInferModel, VinoFaceInferModel
from src.app.utils.googlify_utils import apply_googly_eyes
from src.app.utils.image_utils import (
    image_path_to_bytes,
    load_image_from_bytes,
)

from src.app.utils.vis_utils import draw_annotations


def setup_registery():

    model_registery = ModelRegistry()

    # Initialize OpenVINO Core
    ie = Core()

    # Paths to the model's .xml and .bin files

    percision = "FP32"

    face_artifacts_path = "artifacts/intel/face-detection-adas-0001"
    face_xml = "face-detection-adas-0001.xml"
    face_bin = "face-detection-adas-0001.bin"

    face_xml_path = f"{face_artifacts_path}/{percision}/{face_xml}"
    face_bin_path = f"{face_artifacts_path}/{percision}/{face_bin}"

    # Load the face detection model
    face_model = ie.read_model(model=face_xml_path, weights=face_bin_path)
    compiled_face_model = ie.compile_model(model=face_model)

    # Get input and output layer information for both models
    face_input_layer = next(iter(compiled_face_model.inputs))
    face_output_layer = next(iter(compiled_face_model.outputs))

    model_registery.register_model(
        "face-detection-adas-0001",
        face_input_layer,
        face_output_layer,
        compiled_face_model,
    )

    landmarks_artifacts_path = "artifacts/intel/facial-landmarks-35-adas-0002"
    landmark_xml = "facial-landmarks-35-adas-0002.xml"
    landmark_bin = "facial-landmarks-35-adas-0002.bin"
    landmark_xml_path = f"{landmarks_artifacts_path}/{percision}/{landmark_xml}"
    landmark_bin_path = f"{landmarks_artifacts_path}/{percision}/{landmark_bin}"

    landmarks_model = ie.read_model(model=landmark_xml_path, weights=landmark_bin_path)
    compiled_landmarks_model = ie.compile_model(model=landmarks_model)

    landmarks_input_layer = next(iter(compiled_landmarks_model.inputs))
    landmarks_output_layer = next(iter(compiled_landmarks_model.outputs))

    model_registery.register_model(
        "facial-landmarks-35-adas-0002",
        landmarks_input_layer,
        landmarks_output_layer,
        compiled_landmarks_model,
    )


def setup_image_model(image_path):

    image_bytes = image_path_to_bytes(image_path)
    image_model = ImageModel(
        data=image_bytes,
        filename=image_path.split("/")[-1],
        format=ImageFormatEnum.JPEG,
        modified_filename=None,
    )

    googly0_path = "assets/googly0.png"
    googly0_bytes = image_path_to_bytes(googly0_path)
    googly0_model = ImageModel(
        data=googly0_bytes,
        filename=googly0_path.split("/")[-1],
        format=ImageFormatEnum.PNG,
        modified_filename=None,
    )
    return image_model, googly0_model


def test_image(image_model, googly0_model, show=False):

    # Example usage

    face_config_path = "config/face_model_config.json"
    artifacts = ModelArtifacts(config_path=face_config_path).setup_artifacts()
    face_infer_model = VinoFaceInferModel(artifacts=artifacts)
    eye_config_path = "config/eye_model_config.json"
    artifacts = ModelArtifacts(config_path=eye_config_path).setup_artifacts()
    eye_infer_model = VinoEyeInferModel(artifacts=artifacts)
    faces = face_infer_model.predict(image_model)
    # print(faces)
    if not faces:
        print("No faces detected")
        return

    eyes = eye_infer_model.predict(image_model, faces)
    print(len(eyes))
    # eyes = [eye for eyes in face_eyes_preds for eye in eyes[0:2]]
    # faces = [face[2] for face in face_eyes_preds]
    if not eyes:
        return
    # for fe in face_eyes_preds:
    #     left_eye, right_eye = fe[0:2]
    #     print(f"Face: {fe[2]}")
    #     print(f"Left Eye: {left_eye}")
    #     print(f"Right Eye: {right_eye}")
    googly_features = GooglyModel(
        faces=faces,
        eyes=eyes,
        input_image=image_model,
        googly=googly0_model,
    )
    # googly_image = apply_googly_eyes(googly_features)
    #
    # if show:
    #     cv2.imshow("Googly Eyes", googly_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    googly_image = load_image_from_bytes(image_model.data)
    draw_annotations(googly_image, eyes, faces)
    # save_image_from_array(googly_image, f"out/{image_model.modified_filename}")
    # image = load_image_from_bytes(image_model.data)
    # visualize_landmarks(image_model.image, eyes)


setup_registery()

i1_path = "assets/group2.jpg"
i2_path = "assets/group.jpg"
i3_path = "assets/group3.jpg"
i4_path = "assets/group4.jpg"
i5_path = "assets/person.jpg"

m1, g0 = setup_image_model(i1_path)
m2, g0 = setup_image_model(i2_path)
m3, g0 = setup_image_model(i3_path)
m4, g0 = setup_image_model(i4_path)
m5, g0 = setup_image_model(i5_path)

test_image(m1, g0, show=True)
# test_image(m2, g0, show=True)
# test_image(m3, g0, show=True)
# test_image(m4, g0, show=True)
# test_image(m5, g0, show=True)
