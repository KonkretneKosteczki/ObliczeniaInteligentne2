from timeit import default_timer as timer
from typing import Tuple, List

import torch
from torch.autograd import Variable
from torchvision.datasets import MNIST

from model import save_model, load_model
from test import test_model_acc

criterion = torch.nn.CrossEntropyLoss()  # Softmax is internally computed. The cross-entropy cost function


def train(mnist_train: MNIST, batch_size: int, training_epochs: int, learning_rate: float, total_batch: int,
          model: torch.nn.Module, mnist_test: MNIST) -> Tuple[List[List[float]], List[float], torch.nn.Module]:
    print('\nTraining the Deep Learning network ...')

    best_test_accuracy: float = 0.0
    train_accu: [float] = []
    train_cost: [float] = []
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)

    # issues on windows with multiple workers, as project was developed on ubuntu
    # data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)

    training_start = timer()
    for epoch in range(training_epochs):
        avg_cost = 0
        train_cost.append([])
        start = timer()
        for i, (batch_X, batch_Y) in enumerate(data_loader):
            X: Variable = Variable(batch_X.cuda())  # image is already size of (28x28), no reshape
            Y: Variable = Variable(batch_Y.cuda())  # label is not one-hot encoded

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
            train_cost[epoch].append(cost.item())
            if i % 200 == 0:
                print("Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(epoch + 1, i, train_cost[epoch][-1],
                                                                                          train_accu[-1]))

            avg_cost += cost.data / total_batch
            del X, Y, prediction, hypothesis

        end = timer()
        print(
            "[Epoch: {:>4}], averaged cost = {:>.9}, time spent = {}s".format(epoch + 1, avg_cost.item(), end - start))

        test_accuracy = test_model_acc(model, mnist_test, 100)
        if best_test_accuracy < test_accuracy:
            best_test_accuracy = test_accuracy
            save_model(model, "model.pth")

    total_time = timer() - training_start
    print('Learning Finished! time spent = {}s, on average {}s per epoch'.format(total_time, total_time/training_epochs))
    return train_cost, train_accu, load_model(model, "model.pth")

