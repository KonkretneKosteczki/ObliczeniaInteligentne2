import pandas as pd
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from numpy import int64, int32
from torch import Tensor

from SwordDataset import SwordDataset
from display import display_results, display_confusion_matrix, display_cost, get_label_dictionary
from test import test_model_matrix
from train import train

device = "cuda"
batch_size = 32
learning_rate = 0.0001
# model_children_to_delete = [0, 1]

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def expand_data(*args):
    return transforms.Lambda(lambda x: x.expand(3, -1, -1))(*args)


def transform(pic: Image) -> Tensor:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        expand_data
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(pic)


def load_dataset(data_path):
    return torchvision.datasets.ImageFolder(
        root=data_path,
        # transform=transform
        transform=transforms.ToTensor()
    )

video_filename = 'in1.mp4'

csv = pd.read_csv('train/out1.txt', header=None, sep=';', na_values='#')
#debug
# for i in range(0, len(csv)):
#     print(csv.iloc[i].values.astype(int))

train_dataset = SwordDataset(csv, "in/in1_", train=True, transform=transforms.ToTensor())
print(len(train_dataset))
# for id, each in enumerate(torch.utils.data.DataLoader(dataset=train_dataset)):
#     print(id,each)

print(train_dataset)

model = models.alexnet(pretrained=False)
# model = CNN()

num_classes = 4
epochs = 15

model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
print(model)
model.to(device)

train_length = int(0.98 * len(train_dataset))
val_length = len(train_dataset) - train_length
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (train_length, val_length))


def freeze_layers(model, layer_indexes):
    print(list(model.named_children()))
    for ct, child in enumerate(list(model.children())):
        if ct in layer_indexes:
            for param in child.parameters():
                param.requires_grad = False


# first and second fully connected layer
# freeze_layers(model, model_children_to_delete)

total_batch: int = len(train_dataset) // batch_size
train_cost, train_accu, model = train(train_dataset, batch_size, epochs, learning_rate, total_batch, model,
                                      val_dataset)

print("\nTesting data")
test_confusion_matrix = test_model_matrix(model, test_dataset, 100, matrix_shape=num_classes)
train_confusion_matrix = test_model_matrix(model, train_dataset, 100, matrix_shape=num_classes)
display_results(model, test_dataset)
labels = get_label_dictionary(test_dataset)
display_confusion_matrix(test_confusion_matrix, labels, title="Confusion Matrix (Test Data)")
display_confusion_matrix(train_confusion_matrix, labels, title="Confusion Matrix (Train Data)")
display_cost(train_cost)
