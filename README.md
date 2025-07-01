# MNIST Classification Project

This repository demonstrates multiple **classification techniques** applied to the MNIST dataset—one of the most well-known datasets in machine learning. It is based on **Chapter 3** of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron.

The project walks through binary, multiclass, multilabel, and multioutput classification, including performance metrics, error analysis, and decision threshold tuning.

---

## Project Highlights

- Binary, Multiclass, and Multilabel classification examples
- Confusion matrix, precision-recall, F1 score, and ROC curve
- Decision threshold tuning and precision/recall tradeoff
- Comparison of classifiers: SGD, Random Forest, KNN
- Cross-validation and `StratifiedKFold` evaluation
- Error analysis using prediction visuals and score plots

---

## Goal

To build a strong foundation in classification techniques by:
- Understanding how classifiers behave under different metrics
- Learning to evaluate, tune, and compare performance
- Analyzing misclassifications to guide model improvements

---

## Dataset

**MNIST Dataset**  
- Handwritten digits: 70,000 grayscale 28×28 images (0–9)  
- `train_set`: 60,000 images  
- `test_set`: 10,000 images  
- Classification task: predict the digit (0 to 9) from pixel data

---

## Classification Tasks Covered

| Task                        | Description                                      |
|----------------------------|--------------------------------------------------|
| Binary Classification      | Is digit == 5? Using `SGDClassifier`             |
| Multiclass Classification  | Classify digits 0–9 using OvA and OvO strategies |
| Multilabel Classification  | Predict multiple labels per image                |
| Multioutput Classification | Denoising digits using autoencoder-like logic    |

---

## Evaluation Techniques

- **Confusion Matrix**
- **Precision, Recall, F1 Score**
- **Cross-validation (`cross_val_score`, `StratifiedKFold`)**
- **Decision threshold adjustment**
- **ROC Curve and AUC Score**
- **Top errors and visualization of confusion**

---

## Models Used

| Model                   | Purpose                                  |
|-------------------------|------------------------------------------|
| `SGDClassifier`         | Fast linear classifier for baseline      |
| `KNeighborsClassifier`  | Lazy learning with proximity voting      |
| `RandomForestClassifier`| Ensemble-based robust classification     |
| Threshold tuning logic  | Manual control over decision boundaries  |

---

## Reference

Based on:

> *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*  
> by **Aurélien Géron**

- [📘 Book GitHub](https://github.com/ageron/handson-ml3)
- [📓 Google Colab Notebooks](https://colab.research.google.com/github/ageron/handson-ml3/blob/main)

---

## Notes

Key takeaways and insights from the project:

- **SGDClassifier** is very fast and memory-efficient for large datasets, but highly sensitive to feature scaling.
- In **binary classification**, using accuracy alone is misleading on imbalanced datasets—**precision and recall** matter more.
- **Decision threshold tuning** is powerful: by shifting the threshold, we can favor precision or recall depending on the use case.
- **Cross-validation** gives a more reliable view of model performance than a single train/test split.
- **Multiclass classification** in Scikit-Learn is handled automatically via **One-vs-All (OvA)** or **One-vs-One (OvO)** strategies.
- **Multilabel classification** is supported by passing a list of target labels per instance. `KNeighborsClassifier` works well here because it can vote across multiple labels.
- Visualizing the **confusion matrix** and focusing on the **most common errors** is more effective than blindly tuning hyperparameters.
- **ROC AUC** is useful, but **Precision-Recall curves** are often better for highly imbalanced datasets.
- Misclassifications between similar digits (like 4 vs 9, or 3 vs 5) often reveal the real limits of a model—not noise, but genuinely ambiguous data.