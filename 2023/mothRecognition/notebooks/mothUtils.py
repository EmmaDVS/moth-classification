import torch
import torchvision.transforms.functional as F
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import find_classes

class FilteredImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: None,
        classes_to_trim: {},
    ):
        self.classes_to_trim = classes_to_trim
        super().__init__(
            root,
            transform=transform,
        )        
    
    # We do this by overriding the way ImageFolder finds the classes
    def find_classes(self, directory):
        classes, _ = find_classes(self.root)
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {}
        if len(self.classes_to_trim) > 0:
            i = 1
            for cls_name in classes:
                if cls_name in self.classes_to_trim:
                    class_to_idx[cls_name] = 0
                else:
                    class_to_idx[cls_name] = i
                    i += 1
        else:  # No classes are trimmed
            for i, cls_name in enumerate(classes):
                class_to_idx[cls_name] = i
        return classes, class_to_idx
