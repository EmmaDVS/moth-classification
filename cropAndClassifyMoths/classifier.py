import pickle
import torch
from pathlib import Path
from typing import Tuple
from torchvision import models
from moths.model import Model

def load_model(path: Path, zoo_name: str) -> Tuple[Model]:
    model_fn = getattr(models, zoo_name)
    model = model_fn()

    model.classifier[1] = torch.nn.Linear(1280, 2512)  # nr_in, nr_out
    model_path = str(path / "model_20231020_162530_19.ckpt")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.train(False)

    return model
