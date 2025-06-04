from transformers import pipeline
import os

model = None

def load_model():
    global model
    # Load your HuggingFace model or other ML model
    model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict(input_data, parameters=None):
    if model is None:
        raise Exception("Model not loaded")
    return model(input_data)