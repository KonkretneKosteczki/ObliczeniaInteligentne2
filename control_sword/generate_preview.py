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
label_file_path = cwd + "/data/train/out1.txt"
output_file_path = cwd + "/data/test/out.csv"
device = get_device()

data_loader = torch.utils.data.DataLoader(WorkingDataSet(test_file_path, device), 1, shuffle=False)
labels = read_position(label_file_path)

for _ in range(3):
    for idx, co in enumerate(labels):
        labels[idx] = list(map(lambda x: x[1] if x[1] != "#" else labels[idx - 1][x[0]], enumerate(labels[idx])))

outputs = read_position(output_file_path)

video = cv2.VideoWriter(cwd + "/perfect_lastframe.mp4", cv2.VideoWriter_fourcc(*'avc1'), 30, (640, 360))
images = []
for i, data in enumerate(data_loader):
    # if i < len(data_loader) - 100:
    #     continue
    img = transforms.ToPILImage()(data[0].cpu())
    mark_sword_on_image(img, labels[i], (0, 255, 0))
    mark_sword_on_image(img, outputs[i], (255, 0, 255))
    video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    images.append(img)

img, *imgs = images

img.save(fp=cwd + "/perfect_lastframe.gif", format='GIF', duration=1000 / 30, append_images=imgs[-100:], save_all=True,
         loop=True)
video.release()
