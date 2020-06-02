import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor

from display import display_results, display_confusion_matrix, display_cost
from test import test_model_matrix
from train import train

device = "cuda"
batch_size = 64
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


dataset_num = input("Dataset for mark 3, 4 or 5?")
if dataset_num == "3":
    train_dataset = load_dataset('Pneumonia1/train/')
    val_dataset = load_dataset('Pneumonia1/val')
    test_dataset = load_dataset('Pneumonia1/test/')
    num_classes = 2
    epochs = 15
elif dataset_num == "4":
    Pneumonia2_dataset = load_dataset('Pneumonia2/train/')
    train_length = int(0.98 * len(Pneumonia2_dataset))
    val_length = len(Pneumonia2_dataset) - train_length
    train_dataset, val_dataset = torch.utils.data.random_split(Pneumonia2_dataset, (train_length, val_length))
    test_dataset = load_dataset('Pneumonia2/test/')
    num_classes = 3
    epochs = 20
elif dataset_num =='5':
    Pneumonia3_dataset = load_dataset('Pneumonia3/train/')
    train_length = int(0.98 * len(Pneumonia3_dataset))
    val_length = len(Pneumonia3_dataset) - train_length
    train_dataset, val_dataset = torch.utils.data.random_split(Pneumonia3_dataset, (train_length, val_length))
    test_dataset = load_dataset('Pneumonia3/test/')
    num_classes = 4
    epochs = 20
else:
    print("display must be either 3, 4 or 5")
    exit(1)

print(train_dataset)

model = models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
model.to(device)


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
display_confusion_matrix(test_confusion_matrix, title="Confusion Matrix (Test Data)")
display_confusion_matrix(train_confusion_matrix, title="Confusion Matrix (Train Data)")
display_cost(train_cost)
