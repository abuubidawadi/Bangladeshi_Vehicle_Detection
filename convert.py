from pathlib import Path

import torch

weights_path = Path(__file__).resolve().parent / "bnvd_deploy" / "weights" / "best.pt"

checkpoint = torch.load(
    weights_path,
    map_location=torch.device('cpu'),
    weights_only=False
)

torch.save(checkpoint, weights_path)

print("Converted successfully")