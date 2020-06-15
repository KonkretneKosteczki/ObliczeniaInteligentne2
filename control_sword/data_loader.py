import csv
import os

import natsort as natsort
import torch
from torchvision import transforms
from PIL import Image
from torch import Tensor
from typing import Callable


def transform(pic: Image, offset: [int, int]) -> Tensor:
    def expand_data(*args):
        return transforms.Lambda(lambda x: x.expand(3, -1, -1))(*args)

    def crop(top: int, left: int, height: int, width: int) -> Callable[[Tensor], Tensor]:
        return lambda image: image[..., top:top + height, left:left + width]

    top = offset[1] - 112
    left = offset[0] - 112

    return transforms.Compose([
        transforms.ToTensor(),
        # crop centered at data form offset (center of a sword based on last visible position)
        # crop(offset[0] - 112, offset[1] - 112, 224, 224),
        crop(top, left, 224, 224),
        expand_data
    ])(pic)


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.total_images = natsort.natsorted(os.listdir(main_dir))
        self.labels = read_position("C:/MOO2/control_sword/data/train/out1.txt")

    def __len__(self):
        return len(self.total_images)

    def get_last_pos(self, idx):
        return self.labels[idx-1] if idx != 0 else self.labels[idx]

    @staticmethod
    def calc_offset(position: [int, int, int, int]) -> [int, int]:
        return [(position[0] + position[2]) // 2, (position[1] + position[3]) // 2]

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_images[idx])
        offset = self.calc_offset(self.get_last_pos(idx))
        sword_position = self.labels[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image, offset)
        return {"data": tensor_image, "offset": offset, "label": sword_position}
        # return tensor_image

def load_dataset(data_path):
    return CustomDataSet(data_path, transform)
    # return torchvision.datasets.ImageFolder(root=data_path, transform=transform)


def convert_string_position_to_int(position: str):
    try:
        return int(position)
    except ValueError:  # hash not string for unknown position
        return position


def read_position(file_path: str, delimiter=";"):
    rows = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            rows.append(list(map(convert_string_position_to_int, row)))
    return rows


def load_data(data_path: str, batch_size: int = 32):
    return torch.utils.data.DataLoader(load_dataset(data_path), batch_size, True)
