# 🦈 Shark Species Image Classification

> Deep learning model using MobileNetV3 and transfer learning to classify 13 shark species from images.

---

## 📌 Problem Statement

Accurate identification of shark species is critical for marine conservation and research. Manual identification requires expert knowledge. This project automates species classification using a lightweight, high-accuracy deep learning model deployable on low-resource devices.

---

## 🗂️ Dataset

- **Classes:** 13 shark species
- **Preprocessing:** Resized to 224×224, normalized to ImageNet standards
- **Augmentation:** Random horizontal flip, rotation, zoom, brightness adjustment

The dataset consists of labeled shark images categorized by species. Each class is stored in a separate folder:
The dataset is available on kaggle Here's the link https://www.kaggle.com/datasets/larusso94/shark-species
---

## 🧠 Model Architecture

**Base Model:** MobileNetV3 (pre-trained on ImageNet)

**Transfer Learning Strategy:**
1. Froze base model weights initially
2. Added custom classification head (GlobalAveragePooling → Dense → Dropout → Softmax)
3. Fine-tuned top layers after initial training

**Training Configuration:**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Batch Size | 32 |
| Early Stopping | Patience = 5 |
| Input Size | 224 × 224 × 3 |

---

## 📁 Project Structure

```
shark-image-classification/
│
├── data/                    # Dataset directory
│   ├── train/               # Training images (per class folders)
│   └── test/                # Test images
├── notebooks/
│   └── shark_classifier.ipynb    # Full training pipeline
├── src/
│   ├── model.py             # Model definition
│   ├── train.py             # Training script
│   └── predict.py           # Inference script
├── reports/
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/siyasingh/shark-image-classification.git
cd shark-image-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset in data/ directory

# 4. Run the notebook
jupyter notebook notebooks/shark_classifier.ipynb

# OR train from script
python src/train.py
```

---

## 📦 Requirements

```
tensorflow>=2.10
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
Pillow
```

---

## 📈 Key Results

- Strong classification accuracy across all 13 species
- Transfer learning from MobileNetV3 dramatically reduced training time vs. training from scratch
- Data augmentation improved generalization and reduced overfitting
- Early stopping prevented overfitting with optimal checkpoint saving

**Evaluation:** Confusion matrix and per-class precision/recall/F1 documented in `reports/`

---

## 🔍 Design Decisions

- **Why MobileNetV3?** Lightweight architecture — ideal for deployment on edge devices, while still achieving competitive accuracy
- **Why transfer learning?** Limited domain-specific data; ImageNet features transfer well to natural image classification
- **Why early stopping?** Marine image data has high variance; early stopping + model checkpointing gave the best generalization

---

## 🙋 Author

**Siya Singh** — [LinkedIn](www.linkedin.com/in/siya-singh-947a2a289) | [GitHub](https://github.com/siyasingh2005)

---



