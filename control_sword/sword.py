from torchvision import transforms
from control_sword.data_loader import load_data


if __name__ == "__main__":
    data_loader = load_data("C:/MOO2/control_sword/parsed", 2)
    for batch_id, batch_data in enumerate(data_loader):
        print(batch_data)
        transforms.ToPILImage()(batch_data.get("data")[0]).show()
        break
