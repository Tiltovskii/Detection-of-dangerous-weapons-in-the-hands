from typing import Annotated
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile
import mlflow
from config import *


mlflow.set_tracking_uri("http://localhost:5000")
classes = ['__background__'] + CLASSES
model_name = "Kek"
model_version = "1"
model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": True})


@app.post('/predict')
async def model_prediction(file: Annotated[bytes, File()]):
    img = Image.open(io.BytesIO(file))
    img_tensor = TRANSFORM_PROD(img)
    preds = model([img_tensor])
    preds = [{k: v.to(torch.device('cpu')).tolist() for k, v in p.items()} for p in preds][0]
    return preds
