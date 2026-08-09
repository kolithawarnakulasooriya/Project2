from torch import device
from torch.accelerator import current_accelerator, is_available
from pathlib import Path

root_dir = Path.cwd().parent.parent
models_dir = f"{root_dir}/models"
dataset_dir = f"{root_dir}/dataset"

model_path = f"{models_dir}/fmnist.pth"

def get_device() -> device:
    return device(current_accelerator() if is_available() else 'cpu') 
