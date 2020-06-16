import csv
import os

import torch
from torchvision import transforms
from PIL import Image
from torch import Tensor
from typing import Callable, List

from control_sword.util import list_directories, get_sorted_images_from_dir

image_size = [640, 360]
crop_size = [224, 224]  # appropriate dimensions for AlexNet


# transforms.ToPILImage()(crop(img, 0, 0, 100, 100)).show()


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


class WorkingDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, device):
        self.main_dir = main_dir
        self.device = device
        self.total_images = get_sorted_images_from_dir(main_dir)

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_images[idx])
        image = Image.open(img_loc).convert("RGB")
        return image


class TrainingDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform, device):
        self.main_dir = main_dir
        self.transform = transform
        self.device = device

        self.total_images = []
        self.labels = []

        for directory in list_directories(main_dir):
            self.total_images.extend(get_sorted_images_from_dir(os.path.join(main_dir, directory)))
            self.labels.extend(read_position(os.path.join(main_dir, "out{}.txt".format(directory))))

        self.offsets = list([self.calc_offset(self.get_last_pos(idx)) for idx in range(len(self.total_images))])
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
        offset_x = offset[0] - 112
        offset_y = offset[1] - 112
        return [position[0] - offset_x, position[1] - offset_y, position[2] - offset_x, position[3] - offset_y]

    @staticmethod
    def calc_offset(position: [int, int, int, int]) -> [int, int]:
        offset_x = (position[0] + position[2]) // 2
        offset_y = (position[1] + position[3]) // 2

        # corrections for out of bounds of the image cropping
        if offset_x < 112:
            offset_x -= (offset_x - 112)
        elif offset_x + 112 > image_size[0]:
            offset_x -= (offset_x + 112 - image_size[0])
        if offset_y < 112:
            offset_y -= (offset_y - 112)
        elif offset_y + 112 > image_size[1]:
            offset_y -= (offset_y + 112 - image_size[1])

        return [offset_x, offset_y]

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_images[idx])
        offset = self.offsets[idx]
        sword_position = self.labels[idx]  # self.fill_labels(self.labels[idx], idx)
        image = Image.open(img_loc).convert("RGB")
        cropped_position = self.calc_cropped_position(sword_position, offset)
        tensor_image = self.transform(image, offset).to(self.device)
        return {"data": tensor_image,
                "offset": offset,
                "label": Tensor(cropped_position).to(self.device)}


def load_dataset(data_path, device, training):
    return TrainingDataSet(data_path, transform, device, training)


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


def write_position(file_path: str, rows: List[List[int, int, int, int]], delimiter=";"):
    with open(file_path) as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=delimiter)
        csv_writer.writerows(rows)


def load_data(data_path: str, batch_size: int = 32, device="cpu", training=True, shuffle=True):
    dataset = load_dataset(data_path, device, training)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle), len(dataset) // batch_size
