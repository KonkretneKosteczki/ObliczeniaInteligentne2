import csv
import os

import natsort as natsort
import torch
from torchvision import transforms
from PIL import Image
from torch import Tensor
from typing import Callable

crop_size = [224, 224]  # appropriate dimensions for AlexNet


def transform(pic: Image, offset: [int, int]) -> Tensor:
    def expand_data(*args):
        return transforms.Lambda(lambda x: x.expand(3, -1, -1))(*args)

    def centered_crop(center: [int, int], height: int, width: int):
        top = center[1] - height // 2
        left = center[0] - width // 2
        return crop(top, left, height, width)

    def crop(top: int, left: int, height: int, width: int) -> Callable[[Tensor], Tensor]:
        return lambda image: image[..., top:top + height, left:left + width]

    return transforms.Compose([
        transforms.ToTensor(),
        # crop centered at data form offset (center of a sword based on last visible position)
        centered_crop(offset, *crop_size),
        expand_data
    ])(pic)


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform

        self.total_images = natsort.natsorted(os.listdir(main_dir))
        self.labels = read_position("C:/MOO2/control_sword/data/train/out1.txt")
        self.offsets = list([self.calc_offset(self.get_last_pos(idx)) for idx in range(len(self.total_images))])
        # self.sword_positions = list([self.fill_labels(self.labels[idx], idx) for idx in range(len(self.total_images))])

        # remove partially obscured from training
        for idx, pos in reversed(list(enumerate(self.labels))):
            if "#" in pos:
                del self.total_images[idx], self.offsets[idx], self.labels[idx]

    def __len__(self):
        return len(self.total_images)

    def fill_labels(self, labs, idx):
        # replaces unknown positions with last known positions
        frame_labels = self.labels[idx - 1]
        changed = False
        for pos_id in range(len(labs)):
            if labs[pos_id] == "#":
                labs[pos_id] = frame_labels[pos_id]
                changed = True
        if changed:
            labs = self.fill_labels(labs, idx - 1)
        return labs

    def get_last_pos(self, idx):
        return self.fill_labels(self.labels[idx - 1] if idx != 0 else self.labels[idx], idx - 1)

    @staticmethod
    def calc_cropped_position(position, offset):
        offset_y = offset[1] - 112
        offset_x = offset[0] - 112
        return [position[0] - offset_x, position[1] - offset_y, position[2] - offset_x, position[3] - offset_y]

    @staticmethod
    def calc_offset(position: [int, int, int, int]) -> [int, int]:
        return [(position[0] + position[2]) // 2, (position[1] + position[3]) // 2]

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_images[idx])
        offset = self.offsets[idx]
        # offset = self.calc_offset(self.get_last_pos(idx))
        sword_position = self.fill_labels(self.labels[idx], idx)
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image, offset)
        return {"data": tensor_image, "offset": offset, "label": Tensor(self.calc_cropped_position(sword_position, offset))}


def load_dataset(data_path):
    return CustomDataSet(data_path, transform)


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
    dataset = load_dataset(data_path)
    return torch.utils.data.DataLoader(dataset, batch_size, True), len(dataset) // batch_size
