from typing import List

import numpy as np
import torch
from matplotlib import pylab as plt
from matplotlib.backend_bases import KeyEvent
from torch import Tensor

fig, ax = plt.subplots()


def draw_images(data, model, device):
    loader = iter(torch.utils.data.DataLoader(data, batch_size=16))

    def draw(event: KeyEvent):
        images, labels = next(loader)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        prediction: Tensor = outputs.argmax(1)

        for i in torch.arange(0, 16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].cpu()[0])
            plt.title('P:{} R:{}'.format(prediction[i], labels[i]), color="k" if prediction[i] == labels[i] else "r")
            plt.xticks([]), plt.yticks([])
            plt.plt.subplots_adjust()
        plt.draw()

    return draw


def display_results(model, train_dataset, cost: List[float], acc: List[float], device="cuda"):
    plt.subplot(121), plt.plot(np.arange(len(cost)), cost), plt.ylim([0, 10])
    plt.subplot(122), plt.plot(np.arange(len(acc)), 100 * torch.as_tensor(acc).cpu().numpy()), plt.ylim([0, 100])
    draw_function = draw_images(train_dataset, model, device)
    fig.canvas.mpl_connect('key_press_event', draw_function)
    plt.show()
