# 🌿 Tomato Leaf Disease Detector (ML)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15+-red.svg)](https://pytorch.org/vision/)
[![Gradio](https://img.shields.io/badge/Gradio-UI%20App-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Computer Vision and Deep Learning system to detect and diagnose **9 tomato leaf diseases** (plus healthy leaves) from photographs using transfer learning with **ResNet-18**, coupled with an interactive, mobile-friendly **Gradio web application** ready for Hugging Face Spaces deployment.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Supported Diseases & Classes](#-supported-diseases--classes)
- [Model Architecture & Training Pipeline](#-model-architecture--training-pipeline)
- [Performance & Results](#-performance--results)
- [Dataset Details](#-dataset-details)
- [Dataset Bias & Real-World Considerations](#-dataset-bias--real-world-considerations)
- [Repository Structure](#-repository-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Step-by-Step Guide](#-step-by-step-guide)
  - [1. Train in Google Colab (Recommended)](#1-train-in-google-colab-recommended)
  - [2. Run the Web App Locally](#2-run-the-web-app-locally)
  - [3. Deploy to Hugging Face Spaces](#3-deploy-to-hugging-face-spaces)
- [Inference Code Snippet](#-inference-code-snippet)
- [Ideas for Extension](#-ideas-for-extension)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🚀 Project Overview

Plant diseases cause substantial yield loss in agriculture worldwide. Early and accurate detection enables targeted interventions, reducing crop loss and pesticide overuse.

This repository provides:
1. **Interactive Jupyter / Colab Notebook** (`plant_disease_detector.ipynb`):
   - Automated data downloading via Git sparse-checkout from the PlantVillage dataset.
   - Robust data augmentation (random resized crops, rotations, flips, color jitter).
   - Two-stage Transfer Learning strategy: **Feature Extraction** (frozen backbone) followed by **Fine-Tuning** (full network, low learning rate).
   - Evaluation with Confusion Matrices, Classification Reports, and visual prediction samples.
2. **Gradio Web Application** (`app.py`):
   - Lightweight, mobile-friendly web interface.
   - Real-time image upload or live camera capture.
   - Top-3 disease confidence predictions with sanitized, human-readable labels.

---

## 🔬 Supported Diseases & Classes

The model is trained to classify images into **10 distinct categories**:

| # | Class Name in Dataset | Clean Display Name | Type |
|---|---|---|---|
| 1 | `Tomato___Bacterial_spot` | Bacterial Spot (*Xanthomonas*) | Bacterial |
| 2 | `Tomato___Early_blight` | Early Blight (*Alternaria solani*) | Fungal |
| 3 | `Tomato___Late_blight` | Late Blight (*Phytophthora infestans*) | Oomycete |
| 4 | `Tomato___Leaf_Mold` | Leaf Mold (*Passalora fulva*) | Fungal |
| 5 | `Tomato___Septoria_leaf_spot` | Septoria Leaf Spot (*Septoria lycopersici*) | Fungal |
| 6 | `Tomato___Spider_mites_Two-spotted_spider_mite` | Two-Spotted Spider Mites (*Tetranychus urticae*) | Pest / Mite |
| 7 | `Tomato___Target_Spot` | Target Spot (*Corynespora cassiicola*) | Fungal |
| 8 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Yellow Leaf Curl Virus (TYLCV) | Viral |
| 9 | `Tomato___Tomato_mosaic_virus` | Tomato Mosaic Virus (ToMV) | Viral |
| 10 | `Tomato___healthy` | Healthy Leaf | Normal |

---

## 🧠 Model Architecture & Training Pipeline

Training convolutional neural networks from scratch requires hundreds of thousands of images and long compute hours. This project uses **Transfer Learning** based on **ResNet-18** pretrained on ImageNet (1.2M images, 1,000 categories).

```
   [ Input Image (224x224 RGB) ]
                 │
                 ▼
   [ Pretrained ResNet-18 Backbone (Feature Extractor) ]
                 │
                 ▼
   [ Custom Fully Connected Layer (10 Output Classes) ]
                 │
                 ▼
     [ Softmax Probability Distribution ]
```

### Two-Stage Training Strategy
1. **Stage 1: Feature Extraction**
   - Freeze all convolutional backbone parameters (`requires_grad = False`).
   - Train only the newly replaced final linear classification head (`fc`).
   - Learning Rate: `1e-3` (Adam optimizer) for 3 epochs.
   - Rapid convergence to ~90–95% validation accuracy in minutes.
2. **Stage 2: Fine-Tuning**
   - Unfreeze all layers (`requires_grad = True`).
   - Train the entire network end-to-end with a lower learning rate (`1e-4`).
   - Adapts earlier convolutional filters to botanical texture nuances without catastrophic forgetting.
   - Pushes validation accuracy to **~98–99%**.

### Data Augmentation & Preprocessing
- **Training Pipeline**:
  - `RandomResizedCrop(224, scale=(0.7, 1.0))`
  - `RandomHorizontalFlip()`, `RandomVerticalFlip()`
  - `RandomRotation(20)`
  - `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)`
  - Normalization: ImageNet Mean `[0.485, 0.456, 0.406]` & Std `[0.229, 0.224, 0.225]`
- **Validation / Test / Inference Pipeline**:
  - `Resize(256)`
  - `CenterCrop(224)`
  - Normalization with ImageNet statistics

---

## 📊 Performance & Results

| Training Stage | Typical Validation Accuracy | Test Accuracy |
|---|---|---|
| **Stage 1 (Feature Extraction, 3 epochs)** | ~90–95% | — |
| **Stage 2 (Fine-Tuning, 3 epochs)** | **~98–99%** | **~98.5%+** |

*Checkpoints are saved automatically whenever validation accuracy reaches a new peak.*

---

## 📁 Dataset Details

- **Source:** [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) (~54,000 total images).
- **Subset:** ~18,160 images filtered exclusively for tomato crops across 10 classes.
- **Dataset Partitioning:**
  - **Train (70%):** Model parameter optimization.
  - **Validation (15%):** Model selection & checkpointing.
  - **Test (15%):** Unbiased final evaluation.

---

## ⚠️ Dataset Bias & Real-World Considerations

> **Important Note on Lab vs. Field Imagery:**  
> PlantVillage photos were captured in controlled laboratory settings (single leaf centered against a plain gray/black background under uniform lighting).  
> While the model achieves **~99% accuracy on lab data**, performance can drop on messy real-world field photos containing:
> - Multiple overlapping leaves and branches.
> - Direct sunlight, harsh shadows, and variable background soil/foliage.
> - Co-occurring infections and nutritional deficiencies.

### Suggestions for Robust Real-World Deployment:
1. Augment with in-the-wild datasets such as [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset).
2. Apply synthetic background replacement or segmentation during preprocessing.
3. Test against direct smartphone photos taken under natural conditions.

---

## 📂 Repository Structure

```text
plant-disease-detector-ml/
│
├── plant_disease_detector.ipynb   # Complete training, evaluation & export notebook
├── app.py                         # Standalone Gradio web application for inference
├── requirements.txt               # Production & inference dependencies
├── README.md                      # Project documentation and guide
│
└── [Generated after training]:
    ├── plant_model.pt             # Trained PyTorch model weights state dict
    └── class_names.json           # JSON list of sorted class names
```

---

## 💻 Installation & Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Sumant26/plant-disease-detector-ml.git
cd plant-disease-detector-ml
```

### 2. Set up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Or on Linux / macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Step-by-Step Guide

### 1. Train in Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Click **File → Upload notebook** and upload [`plant_disease_detector.ipynb`](plant_disease_detector.ipynb).
3. Set the accelerator: **Runtime → Change runtime type → T4 GPU → Save**.
4. Run all cells sequentially (**Runtime → Run all** or `Shift + Enter`).
5. After training completes, download `plant_model.pt` and `class_names.json` from the Colab file explorer.

---

### 2. Run the Web App Locally
1. Place the generated `plant_model.pt` and `class_names.json` in the project root alongside [`app.py`](app.py).
2. Launch the Gradio app:
   ```bash
   python app.py
   ```
3. Open your browser at `http://127.0.0.1:7860`.

---

### 3. Deploy to Hugging Face Spaces
To host a permanent, free, publicly accessible version of the app:

1. Create a free account at [Hugging Face](https://huggingface.co).
2. Go to **Spaces** → **Create new Space**.
3. Fill in:
   - **Space Name:** `tomato-disease-detector` (or your choice)
   - **Space SDK:** `Gradio`
   - **Hardware:** `Free CPU (basic)`
4. Upload the following 4 files to the Space repository:
   - [`app.py`](app.py)
   - [`requirements.txt`](requirements.txt)
   - `plant_model.pt`
   - `class_names.json`
5. Hugging Face will automatically build the container and provide a live URL (e.g. `https://huggingface.co/spaces/<your-username>/tomato-disease-detector`).

---

## ⚡ Inference Code Snippet

To run predictions programmatically within your own Python application:

```python
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Load class labels
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# 2. Recreate model architecture & load weights
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("plant_model.pt", map_location="cpu"))
model.eval()

# 3. Preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Predict function
def predict_disease(image_path: str):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    
    top_prob, top_idx = torch.topk(probs, k=3)
    return [
        {"class": class_names[idx], "confidence": float(prob)}
        for prob, idx in zip(top_prob, top_idx)
    ]

# Example run:
# print(predict_disease("leaf_test.jpg"))
```

---

## 💡 Ideas for Extension

- [ ] **Mobile-Optimized Backbones:** Experiment with `mobilenet_v3_small` or `efficientnet_b0` for ultra-fast edge deployment.
- [ ] **Explainability with Grad-CAM:** Integrate Grad-CAM heatmaps to visualize the exact leaf regions the network focuses on.
- [ ] **Multi-Crop Expansion:** Train across all 38 PlantVillage crop-disease categories (Apple, Corn, Grape, Potato, etc.).
- [ ] **Treatment Recommendations:** Add actionable organic and chemical treatment advice corresponding to each detected disease.

---

## 📜 License & Acknowledgments

- **Dataset:** [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) by David Hughes & Marcel Salathé.
- **Model Backbone:** ResNet architecture by He et al., pretrained on ImageNet via PyTorch `torchvision.models`.
- **License:** Open source under the [MIT License](LICENSE).
