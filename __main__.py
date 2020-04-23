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

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.manual_seed(2)  # 1 93.37%

# hyper-parameters
batch_size = 32
keep_prob = 1  # 0.7  # reduce overfitting


def transform_to_gpu_tensor(pic):
    return transforms.ToTensor()(pic).cuda()


# if __name__ == '__main__':
#     torch.multiprocessing.freeze_support()

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transform_to_gpu_tensor, download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transform_to_gpu_tensor, download=True)

# Display informations about the dataset
print('The training dataset:\t', mnist_train)
print('\nThe testing dataset:\t', mnist_test)


# Implementation of CNN/ConvNet Model using PyTorch (depicted in the picture above)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)  # initialize parameters

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# instantiate CNN model
model = CNN()
model.cuda()

for param in model.parameters():
    print(param.size())

learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()  # Softmax is internally computed. The cross-entropy cost function
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

print('Training the Deep Learning network ...')
train_cost = []
train_accu = []

training_epochs = 25
total_batch = len(mnist_train) // batch_size  # int division

print('Size of the training dataset is {}'.format(mnist_train.data.size()))
print('Size of the testing dataset'.format(mnist_test.data.size()))
print('Batch size is : {}'.format(batch_size))
print('Total number of batches is : {0:2.0f}'.format(total_batch))
print('\nTotal number of epochs is : {0:2.0f}'.format(training_epochs))

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
# data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)

training_start = timer()
for epoch in range(training_epochs):
    avg_cost = 0
    start = timer()
    for i, (batch_X, batch_Y) in enumerate(data_loader):
        X = Variable(batch_X)  # image is already size of (28x28), no reshape
        Y = Variable(batch_Y)  # label is not one-hot encoded

        optimizer.zero_grad()  # <= initialization of the gradients

        # forward propagation
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)  # <= compute the loss function

        # Backward propagation
        cost.backward()  # <= compute the gradient of the loss/cost function
        optimizer.step()  # <= Update the gradients

        # Print some performance to monitor the training
        prediction = hypothesis.data.max(dim=1)[1]
        train_accu.append(((prediction.data == Y.data).float().mean()).item())
        train_cost.append(cost.item())
        if i % 200 == 0:
            print("Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(epoch + 1, i, train_cost[-1],
                                                                                      train_accu[-1]))

        avg_cost += cost.data / total_batch

    end = timer()
    print(
        "[Epoch: {:>4}], averaged cost = {:>.9}, time spent = {}s".format(epoch + 1, avg_cost.item(), end - start))

print('Learning Finished! time spent = {}s'.format(timer() - training_start))

# Test model and check accuracy
model.eval()  # set the model to evaluation mode (dropout=False)

## train accuracy
import statistics

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
