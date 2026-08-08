import torch
from model import CNNModel
from pathlib import Path
from transform import getTransforms
from fastapi import FastAPI, UploadFile, File, responses, exceptions
from PIL import Image
import io

model = CNNModel()
model.load_state_dict(torch.load(f"{Path.cwd()}/fashin.pth"))
model.eval()
print("Model Loaded!")

app = FastAPI(title="CNN FASHION")

@app.post('/predict')
async def predict(file: UploadFile):

    try:
        image_b = await file.read()
        image = Image.open(io.BytesIO(image_b)).convert("L")

        image = getTransforms()(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image).argmax(dim=1)
            print(output)

        return responses.JSONResponse(content={"prediction": output.tolist()})
    except Exception as e:
        raise exceptions.HTTPException(status_code=500, detail=str(e))


