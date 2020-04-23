# results will be directly affected by batch_size, keep_prob, learning_rate, training_epochs,
# indirectly: amount and characteristics of layers (network architecture), size of dataset,
#  criterion adn optimizer functions

import torch
from display import display_results
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from timeit import default_timer as timer
from train import train
from network import CNN
import statistics

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.manual_seed(2)  # 1 93.37%

# hyper-parameters
batch_size = 32
keep_prob = 1  # 0.7  # reduce overfitting

learning_rate = 0.001
training_epochs = 1


def transform_to_gpu_tensor(pic):
    return transforms.ToTensor()(pic).cuda()


# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transform_to_gpu_tensor, download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transform_to_gpu_tensor, download=True)

# Display informations about the dataset
print('The training dataset:\t', mnist_train)
print('\nThe testing dataset:\t', mnist_test)


# Implementation of CNN/ConvNet Model using PyTorch (depicted in the picture above)


# instantiate CNN model
model = CNN(keep_prob)
model.cuda()

for param in model.parameters():
    print(param.size())


print('Training the Deep Learning network ...')
print('Size of the testing dataset'.format(mnist_test.data.size()))

training_start = timer()
train_cost, train_accu = train(mnist_train, batch_size, training_epochs, learning_rate, model)
print('Learning Finished! time spent = {}s'.format(timer() - training_start))

# Test model and check accuracy
model.eval()  # set the model to evaluation mode (dropout=False)


# train accuracy (done in batches, because my pc can't handle this much data written into memory at once)
X_train = Variable(mnist_train.data.view(len(mnist_train), 1, 28, 28).float())
Y_train = Variable(mnist_train.targets)
total_train_accuracy = []
for x, y in zip(X_train.split(100), Y_train.split(100)):
    train_prediction = model(x.cuda())
    correct_train_prediction = (torch.max(train_prediction.data, dim=1)[1] == y.cuda().data)
    train_accuracy = correct_train_prediction.float().mean().item()
    total_train_accuracy.append(train_accuracy)
print('\nTrain set accuracy: {:2.2f} %'.format(statistics.mean(total_train_accuracy) * 100))


# test accuracy
X_test = Variable(mnist_test.data.view(len(mnist_test), 1, 28, 28).float()).cuda()
Y_test = Variable(mnist_test.targets).cuda()
prediction = model(X_test)
correct_prediction = (torch.max(prediction.data, dim=1)[1] == Y_test.data)
accuracy = correct_prediction.float().mean().item()
print('\nTest set accuracy: {:2.2f} %'.format(accuracy * 100))

display_results(X_test, Y_test, prediction, train_cost, train_accu)
