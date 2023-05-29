from __future__ import annotations
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import Compose, ToTensor, Resize, ConvertImageDtype, InterpolationMode


# _______________________________________________________________________________________________ #

def get_transforms(crop_size: tuple[int]) -> tuple[Compose]:
    """ ToTensor:
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor
    of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes
    (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """
    input_transforms  = [ToTensor(), Resize(crop_size), ConvertImageDtype(torch.float)]
    target_transforms = [ToTensor(), Resize(crop_size, InterpolationMode.NEAREST)]
    return Compose(input_transforms), Compose(target_transforms)


# _______________________________________________________________________________________________ #

class SegmentationDataset(torch.utils.data.Dataset):

    """ Base class for Cityscapes & GTAV datasets. """

    def __init__(
        self, rootdir: str, crop_size: tuple[int] = (321, 321), split: str = "train"
    ) -> None:
        super().__init__()
        self.rootdir = Path(rootdir)
        self.split = split
        files_list = self.rootdir / "files_list" / f"{split}.txt"
        self.images = [image.strip() for image in open(files_list)]
        self.crop_size = crop_size
        self.image_root, self.label_root = self.get_datadirs()
        self.image_transforms, self.label_transforms = get_transforms(crop_size)

    def get_datadirs(self) -> tuple[Path]:
        raise NotImplementedError

    def get_label_name(self, index: int) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[np.ndarray]:
        image_name = self.images[index]
        label_name = self.get_label_name(index)
        image = np.array(Image.open(self.image_root / image_name).convert("RGB"))
        label = np.array(Image.open(self.label_root / label_name).convert("RGB")).astype(np.int64)
        image = self.image_transforms(image)
        label = self.label_transforms(label)
        return image, label


# _______________________________________________________________________________________________ #

class Cityscapes(SegmentationDataset):

    """ Assumes the following tree:
        rootdir
        ├── files_list
        |   ├── train.txt
        |   ├── val.txt
        |   ├── test.txt
        |
        ├── leftImg8bit
            ├── img
            |   ├── train
            |   ├── val
            |   ├── test
            |
            ├── label
            |   ├── train
            |   ├── val
            |   ├── test

    Each folder train/, val/, and test/ contains subfolders, which themselfes contain PNGs.
    PNGs files in label/ are named alike their img/ counterpart with the suffix `labelTrainIds`.
    They can be generated from the files with the suffix `labelIds` with the script
    `convert_labels_cityscapes.py`.
    """

    def get_datadirs(self) -> tuple[Path]:
        image_root = self.rootdir / "leftImg8bit" / "img" / self.split
        label_root = self.rootdir / "leftImg8bit" / "label" / self.split
        return image_root, label_root

    def get_label_name(self, index: int) -> str:
        return self.images[index].replace("leftImg8bit", "gtFine_labelTrainIds")


# _______________________________________________________________________________________________ #

class GTAV(SegmentationDataset):

    """ Assumes the following tree:
        rootdir
        ├── files_list
        |    ├── train.txt
        |    ├── val.txt
        |    ├── test.txt
        |
        ├── img
        |
        ├── label

    - img/ should contain PNGs, named index.png, with index in [0, 24966].
    - label/ should contain PNGs, named index_labelTrainIds.png, with index in [0, 24966].
    The original labels can be converted to the good ones with the script `convert_labels_gtav.py`.
    """

    def get_datadirs(self) -> tuple[Path]:
        image_root = self.rootdir / "img"
        label_root = self.rootdir / "label"
        return image_root, label_root

    def get_label_name(self, index: int) -> str:
        image_name = Path(self.images[index])
        return image_name.stem + "_labelTrainIds" + image_name.suffix
