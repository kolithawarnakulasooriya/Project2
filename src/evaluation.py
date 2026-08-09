from pathlib import Path
from lib.model import CNNModel
from lib.data import getTestData
from torch import no_grad
from torch.profiler import ProfilerActivity, profile, record_function
from pathlib import Path

root_dir = Path.cwd()
model_path = f"{root_dir}/models/fmnist.pth"

def evaluation():

    """
    Evaluate the trained model on the test dataset.
    
    """

    model = CNNModel()
    model.load(model_path)
    model.eval()

    dataloader = getTestData()

    no_grad()
    for images, labels in dataloader:
        output = model(images)

        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function('model_inference'):
                predicts = output.argmax(dim = 1)

        trues = (predicts == labels).sum().item()

        print(f"Accuracy = {(trues/len(labels)*100):.2f}%")

    print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))

if __name__ == "__main__":
    evaluation()