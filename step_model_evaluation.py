import torch, torchvision
from pathlib import Path
from model import CNNModel
from transform import getTransforms

root_dir = Path.cwd()

model = CNNModel()
model.load_state_dict(torch.load(f"{root_dir}/fashin.pth"))

model.eval()

dataset = torchvision.datasets.FashionMNIST(root=f"{root_dir}/dataset", train=False, transform=getTransforms())
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=False)

torch.no_grad()
for images, labels in dataloader:
    output = model(images)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.profiler.record_function('model_inference'):
            predicts = output.argmax(dim = 1)

    trues = (predicts == labels).sum().item()

    print(f"Accuracy = {(trues/len(labels)*100):.2f}%")

print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))