import torch


def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    model.load_state_dict(torch.load(model_path))
    return model


def save_model(model: torch.nn.Module, model_path: str) -> None:
    torch.save(model.state_dict(), model_path)
