import json
from pathlib import Path
from typing import Tuple
import torch
import numpy as np
import typer
import pickle
from PIL import Image
from torch import nn
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

torch.hub.set_dir('/home/vlinderstichting/.cache/torch/hub')

from moths.classifier import load_model
from moths.label_hierarchy import LabelHierarchy

from cropAndClassifyMoths import classifier
from cropAndClassifyMoths import cropper

inference_app = typer.Typer()

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def infer_image(
    model: nn.Module, label_hierarchy: LabelHierarchy, path: Path
) -> Tuple[str, float]:

    image = Image.open(str(path)).convert("RGB")

    tfs = Compose(
        [
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[136.24, 133.32, 116.16], std=[46.18, 46.70, 48.89]),
        ]
    )

    image = tfs(image)
    image = torch.unsqueeze(image, 0)

    if torch.cuda.is_available():
        image = image.to("cuda")

    # get first item from batch, get first item from tuple (the class)
    predictions = model.forward(image)[0][0].detach() #.gpu()

    prediction_argmax = torch.argmax(predictions)
    prediction_class = label_hierarchy.classes[prediction_argmax]
    prediction_score = float(torch.softmax(predictions, 0)[prediction_argmax])

    return prediction_class, prediction_score


def inference_new(model_path: Path, image_path: Path, multiple_results: bool) -> Tuple[str, float]:
    # Crop the image
    boxes, images = cropper.crop(model_path, image_path, multiple_results)

    # Load the classification model
    model = classifier.load_model(model_path, "efficientnet_b1")

    with (model_path / "classMapping231106.txt").open("rb") as f:
        class_mapping = json.load(f)

    result = {}

    # Loop over the crops and classify them
    for i, image in enumerate(images):
        tfs = Compose([
            ToTensor(),
            Normalize(
                mean=[0.49426943, 0.4825833, 0.41643557],
                std=[0.26641452, 0.26151788, 0.27572504]
            )
        ])

        image = tfs(image)
        image = torch.unsqueeze(image, 0)

        output = model(image)
        output = output.data.cpu().numpy()
        index = output.argmax()
        score = softmax(output[0]).max()

        result[i] = {
            "path": str(image_path.absolute()),
            "class": str(class_mapping[index]),
            "score": float(score),
            "box": boxes[i]
        }

    # Return the crops with their score and class
    if multiple_results:  # Return nested JSON
        print(json.dumps(result))
    else:  # Return the first result only (there are no more anyway)
        print(json.dumps(result[0]))


@inference_app.command()
def inference(model_path: Path, image_path: Path, version: str = "old", multiple_results: bool = False) -> None:
    """
    Args:
        model_path: folder that contains the model weights and other artifacts needed to load the model
        image_path: which image to do inference on
        result_path: file to write the results to (must be non-existent)
    """
    if version == "new":
        inference_new(model_path, image_path, multiple_results)
    else:
        model, label_hierarchy = load_model(model_path, "efficientnet_b7")
        if torch.cuda.is_available():
           model = model.to("cuda")
        klass, score = infer_image(model, label_hierarchy, image_path)

        if multiple_results:  # New format with nested JSON
            print(json.dumps({0: {
                "path": str(image_path.absolute()),
                "class": klass,
                "score": score
            }}))
        else:  # Original format
            print(json.dumps({"path": str(image_path.absolute()), "class": klass, "score": score}))


if __name__ == "__main__":
    inference_app()
