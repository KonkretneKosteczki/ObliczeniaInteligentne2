# dataset loader
import torch
from timeit import default_timer as timer
from torch.autograd import Variable


criterion = torch.nn.CrossEntropyLoss()  # Softmax is internally computed. The cross-entropy cost function


def train(mnist_train, batch_size, training_epochs, learning_rate, total_batch, model):
    train_accu = []
    train_cost = []
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=8)
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
