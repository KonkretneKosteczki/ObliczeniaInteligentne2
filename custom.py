# results will be directly affected by batch_size, keep_prob, learning_rate, training_epochs,
# indirectly: amount and characteristics of layers (network architecture), size of dataset,
#  criterion adn optimizer functions

from typing import Tuple

import torch.nn.init
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torchvision.datasets import MNIST

from display import display_results, display_confusion_matrix, display_cost
from network import CNN
from test import test_model_matrix
from train import train

# arbitrarily chosen seed value
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# hyper-parameters
batch_size: int = 32
keep_prob: float = 1.0  # 0.7  # reduce overfitting

learning_rate: float = 0.0001
training_epochs: int = 10


def transform_to_gpu_tensor(pic: Image) -> Tensor:
    return transforms.ToTensor()(pic).cuda()


def load_data() -> Tuple[MNIST, MNIST]:
    # MNIST dataset
    return dsets.MNIST(root='MNIST_data/', train=True, transform=transform_to_gpu_tensor, download=True), \
           dsets.MNIST(root='MNIST_data/', train=False, transform=transform_to_gpu_tensor, download=True)


mnist_train, mnist_test = load_data()
total_batch: int = len(mnist_train) // batch_size


def describe_instance(network: CNN, train_data: MNIST, test_data: MNIST, total_batches: int):
    # Display data sets
    print('The training dataset:\t', train_data)
    print('\nThe testing dataset:\t', test_data)

    # Display information about parameters of the network
    for param in network.parameters():
        print(param.size())

    # Display information about the data sets, and training configuration
    print('Size of the testing dataset is {}'.format(test_data.data.size()))
    print('Size of the training dataset is {}'.format(train_data.data.size()))
    print('Batch size is : {}'.format(batch_size))
    print('Total number of batches is : {0:2.0f}'.format(total_batches))
    print('Total number of epochs is : {0:2.0f}'.format(training_epochs))


# instantiate CNN model
model = CNN(keep_prob)
model.cuda()
describe_instance(model, mnist_train, mnist_test, total_batch)
train_cost, train_accu, model = train(mnist_train, batch_size, training_epochs, learning_rate, total_batch, model, mnist_test)

# Test model and check accuracy
model.eval()  # set the model to evaluation mode (dropout=False)

print("\nTesting data")
test_confusion_matrix = test_model_matrix(model, mnist_test, 100)
train_confusion_matrix = test_model_matrix(model, mnist_train, 100)
display_results(model, mnist_test)
display_confusion_matrix(test_confusion_matrix, title="Confusion Matrix (Test Data)")
display_confusion_matrix(train_confusion_matrix, title="Confusion Matrix (Train Data)")
display_cost(train_cost)
