# CardioAI: Deep Learning Cardiovascular Risk Prediction System

An end-to-end clinical decision support system utilizing a **PyTorch Artificial Neural Network (ANN)** and ensemble machine learning classifiers to predict 10-year risk of **Coronary Heart Disease (CHD)** on the gold-standard **Framingham Heart Study Cohort (4,240 patients)**.

---

## 🚀 Key Features

* **🧠 Deep Learning & Multi-Model Benchmarking**:
  * Custom **PyTorch Multi-Layer Perceptron (ANN)** with Batch Normalization, Dropout Regularization, and Weighted Binary Cross-Entropy Loss to handle epidemiological class imbalance.
  * Comparative benchmark against **7 Classical ML Baselines**: Random Forest, XGBoost, Support Vector Machine (SVM), Logistic Regression, Gradient Boosting, K-Nearest Neighbors, and Decision Trees.
* **🩺 Explainable AI & Clinical Risk Drivers**:
  * Heuristic and feature-attribution layer evaluating individual patient biomarkers against **AHA/ACC Clinical Guidelines** (Hypertension stages, Dyslipidemia, Hyperglycemia, Obesity).
* **⚡ Modern Web Application**:
  * Glassmorphic clinical dashboard built with Flask and Vanilla CSS.
  * Interactive 4-category patient entry with one-click **Quick-Fill Presets** (Healthy, Borderline, Clinical High-Risk).
  * Real-time risk probability calculation, risk tier categorization, and dynamic benchmark tabs.

---

## 📊 Dataset & Preprocessing

* **Dataset:** Framingham Heart Study (`framingham.csv` - 4,240 longitudinal patient records)
* **Target:** `TenYearCHD` (Binary outcome: 0 = Low Risk, 1 = Developed CHD within 10 years)
* **Features (15 Clinical Indicators):**
  1. **Demographics:** Gender (`male`), `age`, `education`
  2. **Lifestyle:** `currentSmoker`, `cigsPerDay`, `BMI`
  3. **Medical History:** `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes`
  4. **Clinical Biomarkers:** Total Cholesterol (`totChol`), Systolic BP (`sysBP`), Diastolic BP (`diaBP`), Heart Rate (`heartRate`), Fasting Glucose (`glucose`)
* **Pipeline:**
  * Leak-free Median Imputation (`SimpleImputer`)
  * Standard Feature Normalization (`StandardScaler`)
  * Stratified 80/20 Train-Test Partitioning

---

## 🧠 PyTorch Neural Network Architecture

```
Input (15 Features)
   │
   ▼
[Linear Layer: 15 -> 64] ──► [BatchNorm1d] ──► [ReLU] ──► [Dropout (0.3)]
   │
   ▼
[Linear Layer: 64 -> 32] ──► [BatchNorm1d] ──► [ReLU] ──► [Dropout (0.2)]
   │
   ▼
[Linear Layer: 32 -> 16] ──► [BatchNorm1d] ──► [ReLU] ──► [Dropout (0.1)]
   │
   ▼
[Linear Layer: 16 -> 1]  ──► [Sigmoid] ──► Output (Risk Probability)
```

* **Optimizer:** AdamW (`lr=0.003`, `weight_decay=1e-4`)
* **Learning Rate Scheduler:** `ReduceLROnPlateau(patience=5, factor=0.5)`
* **Loss Function:** `nn.BCEWithLogitsLoss(pos_weight=5.59)`

---

## 📈 Model Performance Summary (Test Cohort n=848)

| Architecture | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Artificial Neural Network (ANN)** | **66.39%** | **25.47%** | **62.79%** | **36.24%** | **68.23%** |
| **Logistic Regression** | 67.22% | 25.41% | 59.69% | 35.65% | **70.05%** |
| **Support Vector Machine (SVM)** | 67.69% | 24.38% | 53.49% | 33.50% | 67.03% |
| **Random Forest** | 70.17% | 25.00% | 48.06% | 32.89% | 66.78% |
| **XGBoost** | 68.04% | 23.11% | 47.29% | 31.04% | 64.28% |
| **Decision Tree** | 66.98% | 23.69% | 52.71% | 32.69% | 65.96% |
| **Gradient Boosting** | 84.20% | 40.00% | 7.75% | 12.99% | 66.23% |
| **K-Nearest Neighbors** | 84.32% | 40.00% | 6.20% | 10.74% | 61.73% |

> **Clinical Insight:** In medical screening, **Recall / Sensitivity** is paramount (minimizing false negatives so high-risk patients are not missed). The **PyTorch ANN achieved the highest recall (62.79%)** among all models.

---

## 💻 How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirement.txt
   ```
2. **Train Models and Generate Plots:**
   ```bash
   python train_and_evaluate.py
   ```
3. **Launch the Web Application:**
   ```bash
   python app.py
   ```
4. **Open in Browser:**
   Navigate to `http://127.0.0.1:5000`

---

## 📁 Project Structure

```
Heart Failure/
├── framingham.csv             # Framingham Heart Study Dataset (4,240 records)
├── train_and_evaluate.py      # PyTorch ANN & ML training, validation & plot export
├── app.py                     # Flask web server and prediction inference API
├── imputer.pkl                # Preprocessing SimpleImputer artifact
├── scaler.pkl                 # StandardScaler artifact
├── ann_model.pth              # PyTorch trained ANN weights
├── heart_model.pkl            # Random Forest ensemble artifact
├── templates/
│   └── index.html             # Interactive Glassmorphic Web Dashboard
├── static/
│   ├── model_comparison.png   # Benchmark comparison chart
│   ├── roc_curves.png         # Multi-model ROC comparison
│   ├── ann_loss_curve.png     # PyTorch training convergence curves
│   ├── confusion_matrices.png # Test set confusion matrices
│   ├── feature_importance.png # Top clinical feature weights
│   └── model_comparison_metrics.json
└── README.md                  # System Documentation
```

---

## 📜 License
Licensed under the [MIT License](licence).