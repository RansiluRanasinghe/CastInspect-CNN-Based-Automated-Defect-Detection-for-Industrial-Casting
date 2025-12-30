# CastGuard — CNN-Based Industrial Casting Defect Detection

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

CastGuard is a deep learning–powered computer vision system designed to automatically detect manufacturing defects in industrial casting products using image classification.

This project simulates a **real-world quality inspection pipeline** commonly found in manufacturing environments, where manual inspection is slow, costly, and error-prone. The goal is to build a robust, production-aligned CNN model capable of distinguishing between **defective** and **non-defective (OK)** cast components.

This project focuses strictly on model development and evaluation, making it ideal as a high-impact CNN portfolio project completed within a short time frame.

---

## ⭐ Project Highlights

- ✅ Built a Convolutional Neural Network (CNN) for industrial defect detection
- ✅ Solved a **binary image classification problem**: Defective vs OK
- ✅ Trained on **real industrial manufacturing images**
- ✅ Worked with grayscale, high-resolution casting images
- ✅ Achieved strong generalization using validation-based evaluation
- ✅ Designed with industry inspection use cases in mind
- ✅ Clean, reproducible training pipeline suitable for extension to deployment

---

## 🧠 Problem Overview

In the casting manufacturing industry, defects such as cracks, blow holes, shrinkage, and surface irregularities can lead to:

- Product rejection
- Financial loss
- Customer dissatisfaction
- Safety risks in mechanical systems

### Traditional Manual Inspection Issues

❌ **Time-consuming**  
❌ **Subject to human error**  
❌ **Not scalable for large production volumes**

This project demonstrates how **deep learning–based visual inspection** can automate defect detection and improve consistency and efficiency in industrial quality control.

---

## 📊 Dataset Information

| Property | Details |
|----------|---------|
| **Dataset Name** | Casting Product Image Data for Quality Inspection |
| **Source** | [Kaggle](https://www.kaggle.com/) |
| **Data Provider** | PILOT TECHNOCAST, Shapar, Rajkot |
| **Image Type** | Grayscale industrial images |
| **Image Sizes** | 300 × 300 (augmented) / 512 × 512 (non-augmented) |

### Classes

- `def_front` — Defective casting
- `ok_front` — Non-defective casting

### Dataset Split (Pre-structured)

**Training Set:**
- Defective: 3,758 images
- OK: 2,875 images

**Test Set:**
- Defective: 453 images
- OK: 262 images

The dataset is pre-structured into `train` and `test` folders, closely resembling how data is organized in real industrial ML pipelines.

---

## 🏗️ Model Architecture

The CNN follows a production-aligned architecture optimized for industrial image classification:

```
Input Layer (Grayscale Images)
    ↓
Convolutional Layers (Feature Extraction)
    ↓
Max-Pooling Layers (Spatial Reduction)
    ↓
Fully Connected Dense Layers
    ↓
Dropout (Regularization)
    ↓
Sigmoid Output (Binary Classification)
```

**Design Principles:**
- Balances performance, interpretability, and training efficiency
- Suitable for real-world inspection systems
- Optimized for binary classification tasks

---

## 📈 Model Evaluation

The model was evaluated using **industry-relevant metrics**, including:

- ✓ Training & validation accuracy
- ✓ Training & validation loss
- ✓ Confusion matrix (to analyze false positives vs false negatives)

The evaluation emphasizes **generalization and reliability**, not just raw accuracy — a key requirement for industrial applications.

### Why This Matters

In manufacturing, the cost of:
- **False Negatives** (missing defects) → Shipping faulty products
- **False Positives** (rejecting good parts) → Wasted materials

Both must be carefully balanced through proper evaluation.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Programming Language** | Python |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Data Handling** | NumPy |
| **Visualization** | Matplotlib |
| **Model Type** | Convolutional Neural Network (CNN) |

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/RansiluRanasinghe/CastGuard-CNN-Industrial-Defect-Detection.git
cd CastGuard-CNN-Industrial-Defect-Detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
Run the training notebook or script to train the CNN using the provided dataset structure:
```bash
# If using Jupyter Notebook
jupyter notebook

# Open and run the training notebook
```

### 4️⃣ Evaluate
Use validation metrics and the confusion matrix to assess model performance.

---

## 🔮 Future Improvements

- [ ] Add **FastAPI or REST-based inference service**
- [ ] Integrate **real-time camera input** for inspection lines
- [ ] Apply **Grad-CAM** for defect localization and explainability
- [ ] Optimize the model for **edge deployment** (TensorFlow Lite)
- [ ] Extend to **multi-defect classification**

---

## 📌 Why This Project Matters

This project demonstrates more than just CNN training:

✓ **Real industrial problem solving**  
✓ **Manufacturing-focused computer vision**  
✓ **Practical dataset handling**  
✓ **Evaluation-driven model development**  
✓ **Production-aware ML thinking**

It reflects how deep learning systems are applied in **real manufacturing environments**, not just academic benchmarks.

---

## 🙏 Acknowledgements

Special thanks to **PILOT TECHNOCAST, Shapar, Rajkot**, for providing the industrial dataset and supporting research into automated quality inspection systems.

The dataset was sourced via [Kaggle](https://www.kaggle.com/) and is used strictly for educational and research purposes.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Connect

**Ransilu Ranasinghe**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ransilu-ranasinghe-a596792ba)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/RansiluRanasinghe)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:dinisthar@gmail.com)

**Field Interests:**  
Machine Learning • Computer Vision • AI Engineering

Always open to discussions around:
- Computer Vision
- Industrial AI
- CNN model design
- Manufacturing automation

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

</div>
