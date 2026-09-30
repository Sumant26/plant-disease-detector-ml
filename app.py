"""Tomato leaf disease detector — Gradio app.
Put plant_model.pt and class_names.json (from the notebook) in the same folder, then:
    pip install -r requirements.txt
    python app.py
Also works as-is on Hugging Face Spaces (SDK: Gradio).
"""
import gradio as gr

import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

with open("class_names.json") as f:
    CLASS_NAMES = json.load(f)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("plant_model.pt", map_location="cpu"))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def pretty(name):
    plant, _, disease = name.partition("___")
    return f"{plant}: {disease.replace('_', ' ').strip()}"


def diagnose(img):
    if img is None:
        return {}
    x = preprocess(Image.fromarray(img).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return {pretty(c): float(p) for c, p in zip(CLASS_NAMES, probs)}


demo = gr.Interface(
    fn=diagnose,
    inputs=gr.Image(type="numpy", sources=["upload", "webcam"], label="Tomato leaf photo"),
    outputs=gr.Label(num_top_classes=3, label="Diagnosis"),
    title="Tomato Leaf Disease Detector",
    description="Upload or take a photo of a single tomato leaf. Trained on PlantVillage lab photos, so results on messy field photos may be less reliable.",
)

if __name__ == "__main__":
    demo.launch()
