import torch
from matplotlib import pylab as plt

fig, ax = plt.subplots()
img_batch = 0


def display_results(mnist_test, X_test, prediction):
    val, idx = torch.max(prediction, dim=1)

    def draw():
        for i in torch.arange(0, 12):
            plt.subplot(4, 4, i + 1)
            plt.imshow(X_test[img_batch * 12 + i][0])
            plt.title('{0:>2}'.format(idx[img_batch * 12 + i].item()))
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
