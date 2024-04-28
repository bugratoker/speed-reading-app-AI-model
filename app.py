from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from torch import autocast
import requests
from transformers import pipeline
import io
from PIL import Image
import time
import base64

app = FastAPI()
print(torch.cuda.is_available())
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
device = "cuda"
model_id="facebook/bart-large-cnn"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

@app.get("/")
def generate(prompt:str):
    start_time = time.time()
    with autocast(device):
        text = summarizer(prompt, max_length=80, min_length=10, do_sample=False)[0]['summary_text']
        image_bytes = query({
	        "inputs": text,
        })
    end_time = time.time()
    elapsed_time=  end_time - start_time
    #image = Image.open(io.BytesIO(image_bytes))
    #image.save("testimage.png")
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    result = {
        "image": image_base64,
        "summarizedText": text,
        "elapsedTime": elapsed_time
    }

    return JSONResponse(content=result)

#http://127.0.0.1:8000/?prompt=a%20man%20with%20red%20head