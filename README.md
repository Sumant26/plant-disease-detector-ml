# 🌿 Plant Disease Detector

Identify 9 tomato leaf diseases (plus healthy) from a photo using transfer learning with ResNet-18, then deploy it as a phone-friendly web app.

**Skills:** transfer learning · feature extraction vs fine-tuning · image augmentation · train/val/test splits · checkpointing · dataset bias · Gradio · Hugging Face Spaces deployment

**Time:** two to three weeks · **Level:** advanced

## Run it
1. Open [Google Colab](https://colab.research.google.com) → **File → Upload notebook** → `plant_disease_detector.ipynb`.
2. **Runtime → Change runtime type → T4 GPU** (required).
3. Run top to bottom. The tomato images (~18k) download from the [PlantVillage GitHub repo](https://github.com/spMohanty/PlantVillage-Dataset).

## Deploy
- **Quick:** in section 9, un-comment `app.launch(share=True)` for a 72-hour public link.
- **Permanent:** upload `app.py`, `requirements.txt`, `plant_model.pt` and `class_names.json` to a free Gradio Space on [Hugging Face](https://huggingface.co/new-space).
- **Local:** put the two model files next to `app.py`, then `pip install -r requirements.txt` and `python app.py`.

## Expected results
| Stage | Validation accuracy |
|---|---|
| Frozen backbone (3 epochs) | ~90–95% |
| Fine-tuned (3 more epochs) | ~98–99% |

⚠️ These are lab photos. Expect lower accuracy on real field photos — see section 7 of the notebook.
# plant-disease-detector-ml
