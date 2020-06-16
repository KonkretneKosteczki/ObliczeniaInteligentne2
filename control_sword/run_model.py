import os
from typing import List
import numpy as np
import torch
from torchvision import transforms

from control_sword.data_loader import WorkingDataSet, TrainingDataSet, transform, write_position, read_position
from control_sword.train_sword_detection_model import initialize_model
from control_sword.util import get_device
from control_sword.visual import mark_sword_on_image

cwd = os.getcwd()

test_file_path = cwd + "/data/test/parsed"
output_file_path = cwd + "/data/test/out.csv"
device = get_device()
frame_parsed_data_path = cwd + "/data/train"
model_save_file_path = os.path.join(frame_parsed_data_path, "model.pth")

initial_coordinates = [232, 204, 399, 203]
model = initialize_model()

data_loader = torch.utils.data.DataLoader(WorkingDataSet(test_file_path, device), 1, shuffle=False)

coordinates = [initial_coordinates]
real_coordinates = read_position(cwd+"/data/train/out1.txt")
for _ in range(3):
    for idx, co in enumerate(real_coordinates):
        real_coordinates[idx] = list(map(lambda x: x[1] if x[1] != "#" else real_coordinates[idx - 1][x[0]], enumerate(real_coordinates[idx])))

for idx, data in enumerate(data_loader):
    offset = TrainingDataSet.calc_offset(coordinates[-1])
    # offset = TrainingDataSet.calc_offset(real_coordinates[idx-1])
    img = data[0]
    tensor_image = transform(transforms.ToPILImage()(img), offset).to(device)
    cropped_coordinates = [int(i) for i in np.rint(model(tensor_image.view(1, 3, 224, 224))[0].cpu().detach().numpy())]

    coordinates.append([
        cropped_coordinates[0] + offset[0] - 112,
        cropped_coordinates[1] + offset[1] - 112,
        cropped_coordinates[2] + offset[0] - 112,
        cropped_coordinates[3] + offset[1] - 112,
    ])
    print("Offset: {} Coordinates {}".format(offset, coordinates[-1]))

write_position(output_file_path, coordinates)
