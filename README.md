# Bangladeshi Vehicle Detection

This repository contains a trained YOLO object-detection model, a script to run it on a video, a notebook documenting the dataset and training workflow, and a script for deploying the model to Roboflow.

The project focuses on road users and vehicles found in Bangladesh. Its 17 detection classes are Bicycle, Bus, Bhotbhoti, Car, CNG, Easybike, Leguna, Motorbike, MPV, Pedestrian, Pickup, PowerTiller, Rickshaw, ShoppingVan, Truck, Van, and Wheelbarrow.

## How Detection Works

This is computer vision object detection: for each video frame, the model predicts the locations of objects and assigns each detection a class and confidence score. Unlike image classification, which gives one label to an entire image, object detection can identify multiple objects and draw a bounding box around each one.

The model is a YOLOv8 network, loaded through the Ultralytics Python package. Ultralytics handles image and video input, prepares frames for the network, runs inference, and post-processes predictions into detections. YOLOv8 is built on PyTorch, which provides the tensor operations and neural-network execution on the CPU or a supported GPU. The included `detect.py` uses a confidence threshold of `0.4`; detections below that threshold are filtered out. The script saves an annotated copy of the input video.

## Training Workflow

The notebook `vehicle_detection/object-detection.ipynb` documents a Kaggle-based experiment using the Bangladeshi Native Vehicle Dataset. It prepares image/label pairs, splits them into training, validation, and test sets (70%, 15%, and 15%), and writes a YOLO dataset configuration for the 17 classes.

The notebook includes a short YOLOv8n run for pipeline testing and a longer YOLOv8m training run. The longer run uses 640-pixel images, SGD, and up to 100 epochs with early stopping. Training uses the Ultralytics API, which builds on PyTorch; PyTorch can use a compatible CUDA GPU when one is available. The notebook also includes code to send a trained model to Roboflow.

The full training dataset is not included in this repository. The notebook's training paths refer to Kaggle's `/kaggle/input` and `/kaggle/working` directories, so reproduce training by making the dataset available in Kaggle and adapting those paths if needed. The checked-in weights let you run inference without training again.

## Requirements and Setup

Use Python 3 and install the packages from the repository root. A virtual environment is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements include PyTorch, Ultralytics, and Roboflow. Install a PyTorch build compatible with your CUDA setup if you want GPU acceleration; CPU inference is also supported.

## Run Video Detection

From the repository root, change into the folder containing the video, script, and weights, then run:

```powershell
cd vehicle_detection
python detect.py
```

By default, `detect.py` reads `test_video.mp4`, loads `best.pt`, and writes the annotated video and related prediction files to `outputs/video_detection/`. To use another video or confidence threshold, edit the input path or `conf=0.4` in `detect.py`.

## Deploy to Roboflow

`deploy.py` uploads the model in `bnvd_deploy/` to the workspace, project, and version configured in the script. Provide a Roboflow API key through an environment variable; do not put the key in source code or commit it:

```powershell
$env:ROBOFLOW_API_KEY = "your-api-key"
python deploy.py
```

The deployment call contacts the Roboflow service and requires valid project access and credentials. It is separate from local video inference.

## Repository Layout

- `vehicle_detection/detect.py`: runs local video inference.
- `vehicle_detection/best.pt`: trained weights used by the detection script.
- `vehicle_detection/test_video.mp4`: sample inference input.
- `vehicle_detection/object-detection.ipynb`: Kaggle dataset preparation and training experiment.
- `bnvd_deploy/weights/best.pt`: same checkpoint in the folder layout expected by the Roboflow deployment script.
- `convert.py`: loads and saves the deployment checkpoint using PyTorch.
- `deploy.py`: configures and starts a Roboflow deployment.