from timeit import default_timer as timer
import torchvision.models as models
from PIL import ImageDraw, Image
from torch.nn.modules.loss import _Loss
from torchvision import transforms

from control_sword.data_loader import load_data
import torch

batch_size = 16
epochs = 3
learning_rate = 0.001


class Jaccard(_Loss):
    def forward(self, true_bounding_boxes, own_bounding_boxes):
        def intersection(a, b):  # returns None if rectangles don't intersect
            dx = max(min(a[0], a[2]), min(b[0], b[2])) - min(max(a[0], a[2]), max(b[0], b[2]))
            dy = max(min(a[1], a[3]), min(b[1], b[3])) - min(max(a[1], a[3]), max(b[1], b[3]))
            return dx * dy if (dx <= 0) and (dy <= 0) else 0

        def area(a):
            return abs((a[0] - a[2]) * (a[1] - a[3]))

        def single_jaccard(true_bounding_box, own_bounding_box):
            inter = intersection(true_bounding_box, own_bounding_box)
            return inter / (area(true_bounding_box) + area(own_bounding_box) - inter)

        def not_so_jaccard(true_bounding_box, own_bounding_box):
            inter = intersection(true_bounding_box, own_bounding_box)
            return (area(true_bounding_box) + area(own_bounding_box) - 2 * inter)

        return torch.mean(torch.Tensor(list(map(not_so_jaccard, true_bounding_boxes, own_bounding_boxes))))


def train(batch_size: int, training_epochs: int, learning_rate: float, model: torch.nn.Module):
    # criterion = Jaccard()
    criterion = torch.nn.MSELoss()
    print('\nTraining the Deep Learning network ...')

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    loader, total_batches = load_data("C:/MOO2/control_sword/parsed", batch_size)

    training_start = timer()

    for epoch in range(training_epochs):
        for i, batch_data in enumerate(loader):
            X = batch_data.get("data")
            Y = batch_data.get("label")

            optimizer.zero_grad()  # <= initialization of the gradients
            # forward propagation
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)  # <= compute the loss function
            # cost.requires_grad = True
            # Backward propagation
            cost.backward()  # <= compute the gradient of the loss/cost function
            optimizer.step()  # <= Update the gradients

            print("Epoch= {},\t batch = {}, MSELoss = {:2.4f}".format(epoch + 1, i, cost))
            del X, Y, hypothesis

    total_time = timer() - training_start
    print('Learning Finished! time spent = {}s, on average {}s per epoch'.format(total_time,
                                                                                 total_time / training_epochs))
    # return train_cost, train_accu, load_model(model, "model.pth")


def mark_sword_on_image(img: Image, sword_coordinates: [int, int, int, int], fill=(255, 0, 255)):
    draw = ImageDraw.Draw(img)
    draw.line((sword_coordinates[0], sword_coordinates[1], sword_coordinates[2], sword_coordinates[3]),
              fill=fill, width=3)
    del draw


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 4)
    model.to(device)
    train(batch_size, epochs, learning_rate, model)

    test_data_loader, total_batches = load_data("C:/MOO2/control_sword/parsed", 1)
    # video = cv2.VideoWriter("C:/MOO2/control_sword/test.mp4", cv2.VideoWriter_fourcc(*'avc1'), 30, (224, 224))
    for batch_id, batch_data in enumerate(test_data_loader):
        # print(batch_id, batch_data)
        data = batch_data.get("data")
        labels = batch_data.get("label")
        outputs = model(data)
        # print(batch_data)
        img = transforms.ToPILImage()(data[0])
        mark_sword_on_image(img, labels[0], (0, 255, 0))
        mark_sword_on_image(img, outputs[0], (255, 0, 255))
        # video.write(np.array(img))
        # video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        img.show()
        input("Any key to continue: ")
    # video.release()
