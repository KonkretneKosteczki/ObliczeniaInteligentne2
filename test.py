import torch
from torch import Tensor
from torchvision.datasets import MNIST


def test_model(model: torch.nn.Module, test_data: MNIST, test_batch_size: int = 100, device: str = "cuda") -> Tensor:
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size)
    test_error_count: int = 0
    confusion_matrix: Tensor = torch.zeros(10, 10, dtype=torch.int64).cpu()

    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions: Tensor = outputs.argmax(1)
        test_error_count += torch.sum((labels != predictions))
        for prediction, label in zip(predictions, labels):
            confusion_matrix[prediction][label] += 1

    test_accuracy: float = (1.0 - float(test_error_count) / float(len(test_data))) * 100
    print('Test accuracy: {:2.2f} %'.format(test_accuracy))
    return confusion_matrix
