import torch


def load_model(model_path: str) -> torch.nn.Module:
    return torch.load(model_path)


def save_model(model: torch.nn.Module, model_path: str) -> None:
    torch.save(model.state_dict(), model_path)
