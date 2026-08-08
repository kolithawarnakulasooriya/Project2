import torch, torchvision
from pathlib import Path

from transform import getTransforms

from model import CNNModel

root_dir = Path.cwd()

dataset = torchvision.datasets.FashionMNIST(root=f"{root_dir}/dataset", train=True, download=True, transform=getTransforms())
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

model: torch.nn.Module = CNNModel()

device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu') 
model =  model.to(device=device)
# Training loop

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fucntion = torch.nn.CrossEntropyLoss()

epochs = 10

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

    running_loss = running_loss / len(dataset)
    print(f"{epoch=} | {running_loss=}")
    running_loss = 0.0

torch.save(model.state_dict(), f"{root_dir}/fashin.pth")
        
        
