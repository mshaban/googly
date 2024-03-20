from app.core.deploy import (
    CompositeModelServe,
    VinoEyeInferModelServe,
    VinoFaceInferModelServe,
)
from ray import serve
import ray


ray.init()
serve.start()

face_config_path = "config/face_model_config.json"
eye_config_path = "config/eye_model_config.json"
app = CompositeModelServe.bind(
    VinoFaceInferModelServe.bind(config_path=face_config_path),
    VinoEyeInferModelServe.bind(config_path=eye_config_path),
)
