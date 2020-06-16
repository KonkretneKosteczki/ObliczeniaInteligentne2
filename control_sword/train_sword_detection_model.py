import os
from timeit import default_timer as timer

import cv2
import torchvision.models as models
from torchvision import transforms
import numpy as np

from control_sword.data_loader import load_data
import torch

from control_sword.util import save_model, load_model, get_device
from control_sword.visual import mark_sword_on_image

output_framerate = 30
batch_size = 128
epochs = 10
learning_rate = 0.0001
device = get_device()
cwd = os.getcwd()
frame_parsed_data_path = cwd + "/data/train"
model_save_file_path = os.path.join(frame_parsed_data_path, "model.pth")


def train(batch_s: int, training_epochs: int, l_rate: float, cnn_model: torch.nn.Module):
    print('\nTraining the Deep Learning network ...')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, cnn_model.parameters()), lr=l_rate)
    loader, batches_count = load_data(frame_parsed_data_path, batch_s, device)

    print("\nTotal batches: {}".format(batches_count))
    training_start = timer()
    for epoch in range(training_epochs):
        save_model(model, model_save_file_path)
        for i, batch in enumerate(loader):
            data_x = batch.get("data")
            data_y = batch.get("label")

            optimizer.zero_grad()  # <= initialization of the gradients
            # forward propagation
            hypothesis = cnn_model(data_x)
            cost = criterion(hypothesis, data_y)  # <= compute the loss function
            # Backward propagation
            cost.backward()  # <= compute the gradient of the loss/cost function
            optimizer.step()  # <= Update the gradients

            print("Epoch= {},\t batch = {}, MSELoss = {:2.4f}".format(epoch + 1, i, cost))

    total_time = timer() - training_start
    print('Learning Finished! time spent = {}s, on average {}s per epoch'.format(total_time,
                                                                                 total_time / training_epochs))


def initialize_model():
    m = models.alexnet(pretrained=False)
    m.classifier[6] = torch.nn.Linear(m.classifier[6].in_features, 4)
    m.to(device)
    return load_model(m, model_save_file_path)


if __name__ == "__main__":
    model = initialize_model()
    train(batch_size, epochs, learning_rate, model)

    save_model(model, model_save_file_path)

    test_data_loader, total_batches = load_data(frame_parsed_data_path, 1, device, shuffle=False)
    print("Output frames: {}".format(total_batches))
    video = cv2.VideoWriter(cwd + "/test.mp4", cv2.VideoWriter_fourcc(*'avc1'), 30, (224, 224))
    for batch_id, batch_data in enumerate(test_data_loader):
        if (batch_id + 1) % 100 == 0:
            print("Compiling output: Batch {}".format(batch_id + 1))

        data = batch_data.get("data")
        labels = batch_data.get("label")
        outputs = model(data)
        img = transforms.ToPILImage()(data[0].cpu())  # img.show()
        mark_sword_on_image(img, labels[0], (0, 255, 0))
        mark_sword_on_image(img, outputs[0], (255, 0, 255))
        video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        # input("Any key to continue: ")
    video.release()
