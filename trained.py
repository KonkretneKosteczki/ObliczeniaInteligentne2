from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as dsets
import torchvision.models as models
from torch.autograd import Variable


# torch.backends.cudnn.enabled = False

def expand_data(*args):
    return transforms.Lambda(lambda x: x.expand(3, -1, -1))(*args)

mnist_test = dsets.MNIST(
    root='MNIST_data/',
    train=False,
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        expand_data
    ]),
    download=True
)

loader = DataLoader(mnist_test, batch_size=10, shuffle=True)

# model = models.squeezenet1_1(pretrained=False)
model = models.squeezenet1_1(pretrained=True)
model.eval()
data, target = next(iter(loader))
output = model(data)

pred = output.argmax(dim=1, keepdim=True)
print(pred)
print(output.shape)


# batch_size = 32

# total_batch: int = len(mnist_train) // batch_size
