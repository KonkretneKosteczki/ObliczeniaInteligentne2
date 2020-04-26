import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor

from display import display_results
from test import test_model
from train import train

device = "cuda"
epochs = 10
partial_train = True
batch_size = 32
learning_rate = 0.001
best_accuracy = 0.0

torch.manual_seed(1)
torch.cuda.manual_seed(1)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


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

# print(list([x.size() for x in list(model.parameters())]))
for i, param in enumerate(model.parameters()):
    if i < 5:
        param.requires_grad = False
# list(model.parameters())[1].requires_grad = False

total_batch: int = len(train_dataset) // batch_size
train_cost, train_accu = train(train_dataset, batch_size, epochs, learning_rate, total_batch, model)

print("\nTesting data")
test_model(model, test_dataset, batch_size)
display_results(model, test_dataset, train_cost, train_accu)
torch.save(model.state_dict(), "alex.pth")
