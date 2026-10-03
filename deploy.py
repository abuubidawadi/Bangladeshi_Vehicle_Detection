import os
from pathlib import Path

import roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError("Set the ROBOFLOW_API_KEY environment variable before deploying.")

rf = roboflow.Roboflow(api_key=api_key)
project = rf.workspace("hellos-workspace-nljlo").project("object_detection-br6ya")
version = project.version(1)

version.deploy(
    model_type="yolov8",
    model_path=str(Path(__file__).resolve().parent / "bnvd_deploy")
)