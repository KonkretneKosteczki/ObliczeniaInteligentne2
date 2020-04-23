# results will be directly affected by batch_size, keep_prob, learning_rate, training_epochs,
# indirectly: amount and characteristics of layers (network architecture), size of dataset,
#  criterion adn optimizer functions

from typing import Tuple, Any

import torch.nn.init
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import MNIST

from display import display_results
from network import CNN
from train import train
from test import test_model

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.manual_seed(2)  # arbitrarily chosen seed value
# Training data
# Set accuracy: 84.27 %
#
# Testing data
# Set accuracy: 83.65 %

# hyper-parameters
batch_size: int = 32
keep_prob: float = 1.0  # 0.7  # reduce overfitting

learning_rate: float = 0.001
training_epochs: int = 10


def transform_to_gpu_tensor(pic: Image):
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
train_cost, train_accu = train(mnist_train, batch_size, training_epochs, learning_rate, total_batch, model)

# Test model and check accuracy
model.eval()  # set the model to evaluation mode (dropout=False)

print("\nTraining data")
torch.cuda.empty_cache()
test_model(model, mnist_train, True, 100)

print("\nTesting data")
x_data, y_data, pre = test_model(model, mnist_test, True, 100)
display_results(x_data, y_data, pre, train_cost, train_accu)
