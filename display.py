import itertools
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pylab as plt
from matplotlib.backend_bases import KeyEvent
from torch import Tensor


def draw_images(data, model, device):
    loader = iter(torch.utils.data.DataLoader(data, batch_size=16))

    def draw(_: Optional[KeyEvent] = None):
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


def display_results(model, train_dataset, device="cuda"):
    fig = plt.figure(figsize=(6, 7))
    draw_function = draw_images(train_dataset, model, device)
    fig.canvas.mpl_connect('key_press_event', draw_function)
    draw_function()


def display_cost(cost: List[List[float]]):
    plt.figure(figsize=(10, 2))

    data = list(map(sum, cost))
    ticks = np.arange(len(data)) + 1

    plt.ylim([0, data[0]])
    plt.plot(ticks, data)
    plt.xticks(ticks, ticks)
    plt.title("Total cost over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.show()



def display_confusion_matrix(matrix: Tensor, classes: Optional[list] = None, normalize: bool = False,
                             cmap=plt.cm.Blues, title: str = "Confusion Matrix") -> None:
    plt.figure()
    if classes is None:
        classes = range(len(matrix))

    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # ax.xaxis.tick_top()

