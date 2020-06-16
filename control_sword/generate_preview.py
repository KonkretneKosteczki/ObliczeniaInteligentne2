import os

import cv2
import torch
from torchvision import transforms

from control_sword.data_loader import WorkingDataSet, read_position
from control_sword.util import get_device
from control_sword.visual import mark_sword_on_image
import numpy as np

cwd = os.getcwd()
test_file_path = cwd + "/data/test/parsed"
label_file_path = cwd + "/data/test/out.csv"
output_file_path = cwd + "/data/test/out.csv"
device = get_device()

data_loader = torch.utils.data.DataLoader(WorkingDataSet(test_file_path, device), 1, shuffle=False)
labels = read_position(output_file_path)
outputs = read_position(output_file_path)

video = cv2.VideoWriter(cwd + "/test.mp4", cv2.VideoWriter_fourcc(*'avc1'), 30, (640, 360))
for i, data in enumerate(data_loader):
    img = transforms.ToPILImage()(data[0].cpu())
    # mark_sword_on_image(img, labels[0], (0, 255, 0))
    mark_sword_on_image(img, outputs[i], (255, 0, 255))
    video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

video.release()