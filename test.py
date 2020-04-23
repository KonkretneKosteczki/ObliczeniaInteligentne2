import statistics
from typing import Tuple, Optional, Iterator

import torch
from torch import Tensor
from torch.autograd import Variable
from torchvision.datasets import MNIST

from network import CNN


def test_model(model: CNN, test_data: MNIST, in_batches: bool = False, test_batch_size: int = 100) \
        -> Tuple[Tensor, Tensor, Tensor]:
    x_data = Variable(test_data.data.view(len(test_data), 1, 28, 28).float())
    y_data = Variable(test_data.targets)

    accuracies: [float] = []
    predictions: Optional[Tensor] = None

    data: Iterator[Tuple[Tensor, Tensor]] = zip(x_data.split(test_batch_size), y_data.split(test_batch_size)) \
        if in_batches else zip([x_data], [y_data])

    for x, y in data:
        # need predictions for printing results later, but can't store it in gpu because it takes too much space
        # so i'm just stripping it down to max numbers (decisions) rather than keeping all of it
        prediction: Tensor = torch.max(model(x.cuda()).data, dim=1)[1]
        predictions = torch.cat((predictions, prediction), 0) if predictions is not None else prediction
        accuracy: Tensor = (prediction == y.cuda().data)
        accuracies.append(accuracy.float().mean().item())

    print('Set accuracy: {:2.2f} %'.format(statistics.mean(accuracies) * 100))
    return x_data, y_data, predictions
