import torch
from torch import Tensor
from torchvision.datasets import MNIST


def test_model(model: torch.nn.Module, test_data: MNIST, test_batch_size: int = 100, device="cuda") -> float:
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size)
    test_error_count = 0.0

    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        prediction: Tensor = outputs.argmax(1)
        test_error_count += torch.sum((labels != prediction))

    test_accuracy = (1.0 - float(test_error_count) / float(len(test_data))) * 100
    print('Test accuracy: {:2.2f} %'.format(test_accuracy))
    return test_accuracy
