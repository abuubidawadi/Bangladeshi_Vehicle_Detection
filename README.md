# Vehicle Detection

YOLO-based vehicle detection with an example video and a Roboflow deployment script.

## Setup

The commands below use PowerShell and Python 3. Install the dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run Detection

```powershell
cd vehicle_detection
python detect.py
```

The detector uses `best.pt` and `test_video.mp4` from `vehicle_detection/`. Results are written under `vehicle_detection/outputs/`.

## Deploy to Roboflow

Set your Roboflow API key in the environment, then run the deployment script from the repository root:

```powershell
$env:ROBOFLOW_API_KEY = "your-api-key"
python deploy.py
```

The target workspace, project, and version are configured in `deploy.py`. Never commit API keys or other credentials.

## Repository Contents

- `vehicle_detection/`: detection script, notebook, sample video, and model weights.
- `bnvd_deploy/`: model weights in the layout expected by Roboflow deployment.
- `convert.py`: loads and saves the deployment checkpoint.