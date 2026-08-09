import torch, torchvision
from pathlib import Path
from lib.transform import getTransforms
from lib.model import CNNModel
from lib.data import getTrainData
from lib.utils import get_device
import argparse
from pathlib import Path

root_dir = Path.cwd()
device = get_device()

model_path = f"{root_dir}/models/fmnist.pth"

def train(epochs:int, batch_size:int=32, lr:float = 5e-3):

    """
    Train the model.
    """
    dataloader = getTrainData(batch_size=batch_size)
    model = CNNModel().to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fucntion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            optimizer.zero_grad()
            loss = loss_fucntion(output, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss

        running_loss = running_loss / len(dataloader.dataset)
        print(f"{epoch=} | {running_loss=}")

    model.save(model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training the Model")

    parser.add_argument("-e", "--epochs", default=10, help="Epochs")

    args = parser.parse_args()

    train(epochs=args.epochs)
