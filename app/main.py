import io
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File
import tensorflow as tf
# while running script outside a docker, remove 'app' from 'app.utils' in line 8.

from app.utils import load_model, to_tensor, perform_ocr, result_to_dict

app = FastAPI()
model = load_model()

# warmup the model
_ = model(tf.zeros([1,10,10,3],dtype=tf.uint8)) 

class detectionOutput(BaseModel):
    ocr_result:dict

@app.get("/")
def home():
    return {"health check": "ok"}

@app.post("/ocr",response_model=detectionOutput)
def ocr(file: bytes = File(...)):

    input_image =Image.open(io.BytesIO(file)).convert("RGB")

    # get image tensor and image numpy array
    (tensor, img_np) = to_tensor(input_image)

    # get ROI's on ID card
    detections = model(tensor)

    # extract text from ROI's
    ocr_result, classes = perform_ocr(detections,(img_np.shape[0], img_np.shape[1]), img_np)

    result = result_to_dict(ocr_result, classes)
    
    return {"ocr_result": result}