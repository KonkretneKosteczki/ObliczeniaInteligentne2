import torch
from matplotlib import pylab as plt
import numpy as np

fig, ax = plt.subplots()
img_batch = 0


def display_results(X_test, Y_test, prediction, train_cost, train_accu):
    plt.subplot(121), plt.plot(np.arange(len(train_cost)), train_cost), plt.ylim([0, 10])
    plt.subplot(122), plt.plot(np.arange(len(train_accu)), 100 * torch.as_tensor(train_accu).cpu().numpy()), plt.ylim(
        [0, 100])

    val, idx = torch.max(prediction, dim=1)

    def draw():
        for i in torch.arange(0, 12):
            index = img_batch * 12 + i
            plt.subplot(4, 4, i + 1)
            plt.imshow(X_test[index][0].cpu())
            plt.title('P:{0:>2} R:{0:>2}'.format(idx[index].item(), Y_test.data()[index].item()))
            plt.xticks([]), plt.yticks([])
            plt.plt.subplots_adjust()
        plt.draw()

    def press(event):
        global img_batch
        if event.key == 'right':
            img_batch += 1
        if event.key == 'left':
            img_batch -= 1
        draw()

    fig.canvas.mpl_connect('key_press_event', press)
    plt.show()
