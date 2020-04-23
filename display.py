from typing import List

import numpy as np
import torch
from matplotlib import pylab as plt
from matplotlib.backend_bases import KeyEvent
from torch import Tensor

fig, ax = plt.subplots()


def arrow_press(event: KeyEvent, img_batch: int) -> int:
    if event.key == 'right':
        return img_batch + 1
    if event.key == 'left':
        return img_batch - 1
    return img_batch


def draw_images(x_data: Tensor, y_data: Tensor, idx: Tensor, img_batch: int) -> None:
    for i in torch.arange(0, 16):
        index = img_batch * 16 + i
        predicted: int = idx[index].item()
        real: int = y_data.data[index].item()
        plt.subplot(4, 4, i + 1)
        plt.imshow(x_data[index][0].cpu())
        plt.title('P:{} R:{}'.format(predicted, real), color="k" if predicted == real else "r")
        plt.xticks([]), plt.yticks([])
        plt.plt.subplots_adjust()
    plt.draw()


def display_results(x_data: Tensor, y_data: Tensor, prediction: Tensor, cost: List[float], acc: List[float]):
    plt.subplot(121), plt.plot(np.arange(len(cost)), cost), plt.ylim([0, 10])
    plt.subplot(122), plt.plot(np.arange(len(acc)), 100 * torch.as_tensor(acc).cpu().numpy()), plt.ylim([0, 100])

    img_batch: int = 0

    def key_press_callback(event: KeyEvent) -> None:
        nonlocal img_batch
        img_batch = arrow_press(event, img_batch)
        draw_images(x_data, y_data, prediction, img_batch)

    fig.canvas.mpl_connect('key_press_event', key_press_callback)
    plt.show()
