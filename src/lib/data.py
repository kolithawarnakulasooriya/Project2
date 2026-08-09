"""Data loader helpers for the project.

This module exposes convenience functions to build PyTorch
``DataLoader`` instances for the FashionMNIST dataset. The dataset
is expected to live under the repository ``dataset/`` folder; if it is
missing it will be downloaded automatically by ``torchvision``.

Example
-------
from src.lib.data import getTrainData
loader = getTrainData(batch_size=64)
"""

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from .transform import getTransforms
from .utils import dataset_dir

# Root is two levels up from this module (repo root)
root_dir = Path.cwd().parent.parent


def getTrainData(batch_size: int = 32, num_workers: int = 1) -> DataLoader:

    dataset = FashionMNIST(root=dataset_dir, train=True, download=True, transform=getTransforms())
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def getTestData(batch_size: int = 32, num_workers: int = 1) -> DataLoader:
    dataset = FashionMNIST(root=dataset_dir, train=False, download=True, transform=getTransforms())
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
