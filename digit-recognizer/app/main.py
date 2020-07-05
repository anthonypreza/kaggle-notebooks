import os
from os.path import dirname, realpath
from typing import List

from fastapi import FastAPI
from numpy import argmax, array, squeeze
from pydantic import BaseModel
from tensorflow.keras.models import load_model


class Request(BaseModel):
    data: List[int]


DIR = dirname(realpath(__file__))
MODEL = load_model(f"{DIR}/model/model.h5")
app = FastAPI()


@app.post("/predict")
def predict(request: Request) -> int:
    data = array([request.data]) / 255.0
    x = data.reshape((1, 28, 28, 1))
    class_ = argmax(MODEL.predict(x), axis=-1)[0]
    return {"prediction": int(class_)}
