import os
import natsort
import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def list_directories(cwd):
    return [name for name in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, name))]


def get_sorted_images_from_dir(cwd):
    return [os.path.join(cwd, directory) for directory in natsort.natsorted(os.listdir(cwd))]


def save_model(model: torch.nn.Module, model_path: str) -> None:
    torch.save(model.state_dict(), model_path)


def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    return model
