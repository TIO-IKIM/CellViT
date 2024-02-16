# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, Dataset
from natsort import natsorted
from PIL import Image
from pathlib import Path


class TissueDetectionDataset(Dataset):
    def __init__(self, patched_wsi_path, transforms):
        self.patched_wsi_path = Path(patched_wsi_path)
        self.image_folder = self.patched_wsi_path / "patches"
        self.transforms = transforms
        self.image_list = natsorted(
            [x for x in self.image_folder.iterdir() if x.is_file()]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_filepath = self.image_list[index].resolve()
        image_name = self.image_list[index].name

        image = Image.open(image_filepath)
        image = self.transforms(image)

        return image, image_name


def load_tissue_detection_dl(patched_wsi_path, transforms):
    inference_ds = TissueDetectionDataset(patched_wsi_path, transforms)
    inference_dl = DataLoader(
        dataset=inference_ds,
        batch_size=256,
        num_workers=8,
        prefetch_factor=4,
        shuffle=False,
    )

    return inference_dl
