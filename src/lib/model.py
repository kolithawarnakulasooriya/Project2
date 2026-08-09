import torch

class CNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = torch.nn.Sequential(*[
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 32 -> 16
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2) # 16 -> 8
        ])
        # 8 x 8 x 16
        self.feature = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=1024, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(self.filter(x))

    def save(self, path:str):
        torch.save(self.state_dict(), path)

    def load(self, path:str):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))