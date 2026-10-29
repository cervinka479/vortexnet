"""Smoke tests: the package imports and the compute stack is present."""

import torch

import vortexnet


def test_version() -> None:
    assert vortexnet.__version__ == "0.1.0"


def test_torch_runs_on_available_device() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.ones(3, device=device)
    assert (x * 2).sum().item() == 6.0
