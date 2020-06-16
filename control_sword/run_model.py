import os
from typing import List

import torch

from control_sword.data_loader import WorkingDataSet, TrainingDataSet, transform, write_position
from control_sword.train_sword_detection_model import initialize_model
from control_sword.util import get_device

test_file_path = "C:/MOO2/control_sword/data/test/parsed"
output_file_path = "C:/MOO2/control_sword/data/test/out.csv"
device = get_device()
frame_parsed_data_path = "C:/MOO2/control_sword/data/train"
model_save_file_path = os.path.join(frame_parsed_data_path, "model.pth")

initial_coordinates = [232, 204, 399, 203]
model = initialize_model()

data_loader = torch.utils.data.DataLoader(WorkingDataSet(test_file_path, device), 1, shuffle=False)

coordinates = [initial_coordinates]
for data in data_loader:
    offset: List[int, int] = TrainingDataSet.calc_offset(coordinates[-1])
    img = data[0]
    tensor_image = transform(img, offset).to(device)
    cropped_coordinates: List[int, int, int, int] = model(tensor_image)
    coordinates.append([
        cropped_coordinates[0] + offset[0],
        cropped_coordinates[1] + offset[1],
        cropped_coordinates[2] + offset[0],
        cropped_coordinates[3] + offset[1],
    ])

write_position(output_file_path, coordinates)
