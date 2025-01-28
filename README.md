# Project Title: Breast Cancer Classification Using SVM Model

## Overview

This repository contains the implementation of a breast cancer classification model using Support Vector Machine (SVM). The dataset used in this project is based on data collected in 2023. The model classifies breast cancer as malignant or benign based on diagnostic features. The project emphasizes data preprocessing, feature selection, and hyperparameter tuning to optimize model performance.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Features](#features)
3. [Preprocessing](#preprocessing)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [How to Use](#how-to-use)
7. [Requirements](#requirements)
8. [Results](#results)
9. [Acknowledgments](#acknowledgments)

---

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which includes:

- **Features:** 6 numerical diagnostic features:
  - Mean radius
  - Mean texture
  - Mean perimeter
  - Mean area
  - Mean smoothness

- **Target Variable:**
  - Diagnosis:
    - Malignant (1)
    - Benign (0)

Dataset characteristics:
- 569 samples.
- No missing values.

---

## Features

Key diagnostic features used for classification include:

- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Mean smoothness

Feature selection was conducted to ensure optimal performance of the classification model.

---

## Preprocessing

Steps involved in preprocessing:

1. **Normalization:** Scaling features using Min-Max normalization.
2. **Data Splitting:** Dividing the dataset into training (80%) and testing (20%) sets.

---

## Model Training

The SVM model was trained using:

- **Kernel Types:** Linear, Polynomial, RBF.
- **Hyperparameter Tuning:** Grid search over parameters:
  - C (Regularization parameter)
  - Gamma (Kernel coefficient for RBF and Polynomial kernels)

---

## Evaluation

Evaluation metrics:

- **Accuracy:** Proportion of correctly classified samples.
- **Precision:** Ratio of true positives to total predicted positives.
- **Recall:** Ratio of true positives to total actual positives.
- **F1-Score:** Harmonic mean of precision and recall.

---

## How to Use

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/breast-cancer-svm.git
cd breast-cancer-svm
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the scripts

- Preprocess the data:
  ```bash
  python scripts/preprocess.py
  ```
- Train the model:
  ```bash
  python scripts/train_model.py
  ```
- Evaluate the model:
  ```bash
  python scripts/evaluate_model.py
  ```

---

## Requirements

The project was built using the following libraries:

- Scikit-learn 1.2.2
- NumPy 1.23.5
- Pandas 1.5.3
- Matplotlib 3.6.2

Install these dependencies using `requirements.txt`.

---

## Results

The trained model achieved the following performance:

- **Accuracy:** X% (to be updated after running the experiments)
- **F1-Score:** Y% (to be updated after running the experiments)

The confusion matrix and classification report are available in the `results/` folder.

---

## Acknowledgments

This project was developed as part of an undergraduate thesis at the Faculty of Science and Technology, Mathematics Department. Special thanks to my advisor and colleagues for their support.

---

Feel free to contribute or report issues through this repository. Thank you!

