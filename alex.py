import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor

from display import display_results, display_confusion_matrix, display_cost
from test import test_model_matrix
from train import train

device = "cuda"
epochs = 20
batch_size = 128
learning_rate = 0.001
model_children_to_delete = [0, 1]

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def expand_data(*args):
    return transforms.Lambda(lambda x: x.expand(3, -1, -1))(*args)


def transform(pic: Image) -> Tensor:
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        expand_data
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(pic)


train_dataset = datasets.MNIST(root='MNIST_data/', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='MNIST_data/', train=False, transform=transform)

model = models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 10)
model.to(device)


def freeze_layers(model, layer_indexes):
    print(list(model.named_children()))
    for ct, child in enumerate(list(model.children())):
        if ct in layer_indexes:
            for param in child.parameters():
                param.requires_grad = False


# first and second fully connected layer
freeze_layers(model, model_children_to_delete)

total_batch: int = len(train_dataset) // batch_size
train_cost, train_accu, model = train(train_dataset, batch_size, epochs, learning_rate, total_batch, model,
                                      test_dataset)

print("\nTesting data")
test_confusion_matrix = test_model_matrix(model, test_dataset, 100)
train_confusion_matrix = test_model_matrix(model, train_dataset, 100)
display_results(model, test_dataset)
display_confusion_matrix(test_confusion_matrix, title="Confusion Matrix (Test Data)")
display_confusion_matrix(train_confusion_matrix, title="Confusion Matrix (Train Data)")
display_cost(train_cost)
