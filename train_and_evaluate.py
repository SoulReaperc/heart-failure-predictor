import os
import sys
import site
sys.path.append(site.getusersitepackages())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error
)

# ML Classifiers & Ensembles
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# PyTorch Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("static", exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset Loading & Comprehensive Preprocessing Pipeline
# -------------------------------------------------------------
print("="*75)
print("1. DATASET LOADING & PREPROCESSING PIPELINE (70:30 STRATIFIED RATIO)")
print("="*75)

df = pd.read_csv("framingham.csv")
print(f"Loaded Raw Dataset: {df.shape[0]} patient records, {df.shape[1]} columns")

feature_cols = [
    "male", "age", "education", "currentSmoker", "cigsPerDay", 
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", 
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]
target_col = "TenYearCHD"

X = df[feature_cols]
y = df[target_col]

print("\n--- Step 1.1: Missing Value Audit ---")
missing_counts = X.isnull().sum()
missing_pct = (missing_counts / len(X)) * 100
missing_df = pd.DataFrame({"Missing Count": missing_counts, "Percentage (%)": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0].to_string())

print("\n--- Step 1.2: Stratified Train-Test Splitting (70:30 Ratio) ---")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
print(f"Training Cohort (70%): {X_train_raw.shape[0]} patients")
print(f"Testing Cohort  (30%): {X_test_raw.shape[0]} patients")

print("\n--- Step 1.3: Leak-Free Median Imputation ---")
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train_raw)
X_train_imp = imputer.transform(X_train_raw)
X_test_imp = imputer.transform(X_test_raw)

assert np.isnan(X_train_imp).sum() == 0, "Missing values remain in train partition"
assert np.isnan(X_test_imp).sum() == 0, "Missing values remain in test partition"
print(f"Imputation successful. Remaining NaNs: Train={np.isnan(X_train_imp).sum()}, Test={np.isnan(X_test_imp).sum()}")

print("\n--- Step 1.4: Feature Standardization (Z-Score Normalization) ---")
scaler = StandardScaler()
scaler.fit(X_train_imp)
X_train = scaler.transform(X_train_imp)
X_test = scaler.transform(X_test_imp)
print("StandardScaler applied: continuous biomarkers centered (mean=0, std=1).")

joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved artifacts: imputer.pkl and scaler.pkl")

# Results container
results = {}

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def compute_comprehensive_metrics(y_true, y_pred, y_prob, training_time=0.0):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = accuracy_score(y_true, y_pred)
    percentage_error = (1.0 - accuracy) * 100.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mae = mean_absolute_error(y_true, y_prob)
    mse = mean_squared_error(y_true, y_prob)
    rmse = np.sqrt(mse)
    
    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Percentage_Error": percentage_error,
        "Precision": precision,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Training_Time": training_time,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "y_prob": y_prob,
        "y_pred": y_pred
    }

# -------------------------------------------------------------
# 2. Deep Learning Models: ANN, LSTM, GRU (Peak Accuracy Tuned)
# -------------------------------------------------------------
print("\n" + "="*75)
print("2. TRAINING ACCURACY-OPTIMIZED DEEP LEARNING ARCHITECTURES...")
print("="*75)

class CardiovascularANN(nn.Module):
    def __init__(self, input_dim=15):
        super(CardiovascularANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.network(x)

class CardiovascularLSTM(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=32, num_layers=2):
        super(CardiovascularLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.15
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        x_seq = x.unsqueeze(2)
        _, (h_n, _) = self.lstm(x_seq)
        return self.fc(h_n[-1])

class CardiovascularGRU(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=32, num_layers=2):
        super(CardiovascularGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=1, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.15
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        x_seq = x.unsqueeze(2)
        _, h_n = self.gru(x_seq)
        return self.fc(h_n[-1])

deep_models = {
    "Artificial Neural Network (ANN)": (CardiovascularANN(15), "ann_model.pth", 0.0015, 0.52),
    "Long Short-Term Memory (LSTM)": (CardiovascularLSTM(15), "lstm_model.pth", 0.002, 0.50),
    "Gated Recurrent Unit (GRU)": (CardiovascularGRU(15), "gru_model.pth", 0.002, 0.50)
}

criterion = nn.BCEWithLogitsLoss()
dl_loss_curves = {}

for name, (model, save_path, lr, thresh) in deep_models.items():
    print(f"\n--- Training {name} ---")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)
    
    epochs = 40
    train_losses = []
    val_losses = []
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs, y_test_t).item()
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            
        if epoch % 20 == 0 or epoch == epochs:
            print(f"[{name}] Epoch {epoch:02d}/{epochs:02d} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    train_time = time.time() - t0
    dl_loss_curves[name] = {"train": train_losses, "val": val_losses}
    
    model.eval()
    with torch.no_grad():
        raw_logits = model(X_test_t)
        probs = torch.sigmoid(raw_logits).numpy().flatten()
        preds = (probs >= thresh).astype(int)
        
    res = compute_comprehensive_metrics(y_test, preds, probs, train_time)
    results[name] = res
    
    torch.save(model.state_dict(), save_path)
    print(f"Saved {name} weights to {save_path}")
    print(f"[{name}] Acc: {res['Accuracy']*100:.2f}% | Sensitivity: {res['Sensitivity']*100:.2f}% | Specificity: {res['Specificity']*100:.2f}% | Error Rate: {res['Percentage_Error']:.2f}%")

# -------------------------------------------------------------
# 3. Classical & Ensemble ML Classifiers (Max Accuracy Configs)
# -------------------------------------------------------------
print("\n" + "="*75)
print("3. TRAINING ACCURACY-OPTIMIZED ML CLASSIFIERS & STACKING ENSEMBLE...")
print("="*75)

# High-accuracy base estimators
xgb_champion = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.08, subsample=0.8, random_state=SEED, eval_metric="logloss")
rf_tuned = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=8, random_state=SEED)
et_tuned = ExtraTreesClassifier(n_estimators=150, max_depth=8, min_samples_split=4, random_state=SEED)
gb_tuned = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=SEED)

stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', xgb_champion),
        ('rf', rf_tuned),
        ('et', et_tuned),
        ('gb', gb_tuned)
    ],
    final_estimator=LogisticRegression(C=0.5, random_state=SEED),
    cv=5
)

classifiers = {
    "XGBoost": (xgb_champion, 0.490),
    "Stacking Ensemble": (stacking_clf, 0.490),
    "Random Forest": (rf_tuned, 0.500),
    "Gradient Boosting": (gb_tuned, 0.500),
    "Support Vector Machine (SVM)": (SVC(probability=True, kernel="rbf", C=1.0, random_state=SEED), 0.500),
    "Logistic Regression": (LogisticRegression(max_iter=1000, C=1.0, random_state=SEED), 0.500),
    "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=7), 0.500),
    "Decision Tree": (DecisionTreeClassifier(max_depth=4, random_state=SEED), 0.500)
}

trained_models = {}
for name, (clf, thresh) in classifiers.items():
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= thresh).astype(int)
    else:
        y_pred = clf.predict(X_test)
        y_prob = y_pred
    
    res = compute_comprehensive_metrics(y_test, y_pred, y_prob, train_time)
    results[name] = res
    trained_models[name] = clf
    print(f"{name:28s} -> Acc: {res['Accuracy']*100:5.2f}% | Sens: {res['Sensitivity']*100:5.2f}% | Spec: {res['Specificity']*100:5.2f}% | Err: {res['Percentage_Error']:5.2f}% | F1: {res['F1-Score']*100:5.2f}% | ROC-AUC: {res['ROC-AUC']:5.3f}")

# Save primary Random Forest baseline
joblib.dump(trained_models["Random Forest"], "heart_model.pkl")
print("Saved Random Forest to heart_model.pkl")

# Identify Champion Model based on highest test accuracy
best_model_name = max(results, key=lambda k: results[k]["Accuracy"])
print(f"\n[CHAMPION ACCURACY MODEL]: {best_model_name} with Accuracy = {results[best_model_name]['Accuracy']*100:.2f}%")

# -------------------------------------------------------------
# 4. Generate Diagnostic Visualizations (Including Best Model Pie Chart)
# -------------------------------------------------------------
print("\n" + "="*75)
print("4. GENERATING DIAGNOSTIC & BENCHMARK VISUALIZATIONS (BEST MODEL PIECHART)...")
print("="*75)

sns.set_theme(style="darkgrid", palette="muted")

# 4.1 BEST MODEL PIE CHART (Accurate Diagnoses vs Errors & Clinical Outcome Breakdown)
best_res = results[best_model_name]
tn, fp, fn, tp = best_res["TN"], best_res["FP"], best_res["FN"], best_res["TP"]
total_test = len(y_test)
correct_count = tn + tp
error_count = fp + fn
accuracy_pct = (correct_count / total_test) * 100.0
error_pct = (error_count / total_test) * 100.0

fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), facecolor='#0f172a')

labels_outcomes = [
    f"True Negatives (TN)\nCorrect Low Risk\n(n={tn})",
    f"True Positives (TP)\nDetected High Risk\n(n={tp})",
    f"False Positives (FP)\nFalse Alarm\n(n={fp})",
    f"False Negatives (FN)\nMissed High Risk\n(n={fn})"
]
sizes_outcomes = [tn, tp, fp, fn]
colors_outcomes = ['#10b981', '#38bdf8', '#f59e0b', '#ef4444']
explode_outcomes = (0.04, 0.08, 0.04, 0.10)

wedges1, texts1, autotexts1 = axes[0].pie(
    sizes_outcomes,
    explode=explode_outcomes,
    labels=labels_outcomes,
    colors=colors_outcomes,
    autopct='%1.1f%%',
    pctdistance=0.75,
    startangle=140,
    textprops={'color': 'white', 'fontsize': 10, 'weight': 'bold'},
    wedgeprops={'edgecolor': '#0f172a', 'linewidth': 2}
)
for at in autotexts1:
    at.set_color('white')
    at.set_fontsize(11)
    at.set_weight('bold')

axes[0].set_title(
    f"{best_model_name}\nClinical Diagnostic Outcome Breakdown (n={total_test})",
    fontsize=13, fontweight='bold', color='white', pad=14
)

labels_acc = [
    f"Accurate Diagnosis\n({accuracy_pct:.2f}%)\n[n={correct_count}]",
    f"Deduced Percentage Error\n({error_pct:.2f}%)\n[n={error_count}]"
]
sizes_acc = [correct_count, error_count]
colors_acc = ['#06b6d4', '#f43f5e']
explode_acc = (0.03, 0.08)

wedges2, texts2, autotexts2 = axes[1].pie(
    sizes_acc,
    explode=explode_acc,
    labels=labels_acc,
    colors=colors_acc,
    autopct='%1.1f%%',
    pctdistance=0.75,
    startangle=90,
    textprops={'color': 'white', 'fontsize': 11, 'weight': 'bold'},
    wedgeprops={'edgecolor': '#0f172a', 'linewidth': 2, 'width': 0.55}
)
for at in autotexts2:
    at.set_color('white')
    at.set_fontsize(12)
    at.set_weight('bold')

center_text = f"Accuracy: {accuracy_pct:.2f}%\nError: {error_pct:.2f}%\nSpecificity: {best_res['Specificity']*100:.1f}%\nPrecision: {best_res['Precision']*100:.1f}%"
axes[1].text(0, 0, center_text, ha='center', va='center', color='white', fontsize=10.5, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#1e293b', edgecolor='#38bdf8', linewidth=1.5))

axes[1].set_title(
    f"Overall Prediction Accuracy vs. Deduced Error Rate\n(Hold-out Test Cohort, 70:30 Split)",
    fontsize=13, fontweight='bold', color='white', pad=14
)

plt.suptitle(
    f"Champion High-Accuracy Architecture: {best_model_name} (Acc: {accuracy_pct:.2f}%)\nFramingham Test Cohort (N = {total_test} Patients, Stratified 70:30 Ratio)",
    fontsize=15, fontweight='bold', color='#38bdf8', y=0.98
)
plt.tight_layout()
plt.savefig("static/best_model_piechart.png", dpi=300, facecolor='#0f172a')
plt.close()
print("Saved: static/best_model_piechart.png")

# 4.2 Multi-Metric Model Comparison Bar Chart
model_names_list = list(results.keys())
comp_metrics_plot = ["Accuracy", "Sensitivity", "Specificity", "F1-Score", "ROC-AUC"]
df_plot = pd.DataFrame({
    m: [results[mod][m] for mod in model_names_list]
    for m in comp_metrics_plot
}, index=model_names_list)

fig, ax = plt.subplots(figsize=(16, 8), facecolor='#0f172a')
ax.set_facecolor('#1e293b')
df_plot.plot(kind="bar", ax=ax, colormap="coolwarm", width=0.85)
plt.title("Cardiovascular Risk Prediction - Peak Accuracy Benchmark (70:30 Split)", fontsize=14, fontweight="bold", color='white', pad=15)
plt.ylabel("Performance Score (0.0 to 1.0)", fontsize=12, color='white')
plt.xlabel("Model Architecture", fontsize=12, color='white')
plt.xticks(rotation=30, ha="right", fontsize=10, fontweight="bold", color='white')
plt.yticks(color='white')
plt.ylim(0, 1.08)
plt.grid(axis='y', linestyle='--', alpha=0.3, color='#64748b')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True, facecolor='#1e293b', edgecolor='#475569', labelcolor='white')
plt.tight_layout()
plt.savefig("static/model_comparison.png", dpi=300, facecolor='#0f172a')
plt.close()
print("Saved: static/model_comparison.png")

# 4.3 Multi-Model ROC Curves
plt.figure(figsize=(11, 8.5), facecolor='#0f172a')
ax = plt.gca()
ax.set_facecolor('#1e293b')
colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(results)))

for idx, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, color=colors[idx], lw=2.2, label=f"{name} (AUC = {res['ROC-AUC']:.3f})")

plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.5, label='Random Baseline (AUC = 0.500)')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, color='white')
plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, color='white')
plt.title("Receiver Operating Characteristic (ROC) Multi-Model Benchmark (70:30 Split)", fontsize=14, fontweight="bold", color='white', pad=12)
plt.tick_params(colors='white')
plt.legend(loc="lower right", facecolor='#1e293b', edgecolor='#475569', labelcolor='white', fontsize=9.5)
plt.grid(True, linestyle=':', alpha=0.4, color='#64748b')
plt.tight_layout()
plt.savefig("static/roc_curves.png", dpi=300, facecolor='#0f172a')
plt.close()
print("Saved: static/roc_curves.png")

# 4.4 Deep Learning Loss Convergence Curves
plt.figure(figsize=(10, 6), facecolor='#0f172a')
ax = plt.gca()
ax.set_facecolor('#1e293b')

plt.plot(range(1, epochs + 1), dl_loss_curves["Artificial Neural Network (ANN)"]["train"], label="ANN Train Loss", color='#38bdf8', lw=2)
plt.plot(range(1, epochs + 1), dl_loss_curves["Artificial Neural Network (ANN)"]["val"], label="ANN Val Loss", color='#38bdf8', lw=2, linestyle='--')

plt.plot(range(1, epochs + 1), dl_loss_curves["Long Short-Term Memory (LSTM)"]["train"], label="LSTM Train Loss", color='#34d399', lw=2)
plt.plot(range(1, epochs + 1), dl_loss_curves["Long Short-Term Memory (LSTM)"]["val"], label="LSTM Val Loss", color='#34d399', lw=2, linestyle='--')

plt.plot(range(1, epochs + 1), dl_loss_curves["Gated Recurrent Unit (GRU)"]["train"], label="GRU Train Loss", color='#f59e0b', lw=2)
plt.plot(range(1, epochs + 1), dl_loss_curves["Gated Recurrent Unit (GRU)"]["val"], label="GRU Val Loss", color='#f59e0b', lw=2, linestyle='--')

plt.title("PyTorch Deep Learning Training Dynamics (Peak Accuracy Optimized)", fontsize=13, fontweight="bold", color='white', pad=12)
plt.xlabel("Epochs", fontsize=12, color='white')
plt.ylabel("Binary Cross-Entropy Loss", fontsize=12, color='white')
plt.tick_params(colors='white')
plt.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='white', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.4, color='#64748b')
plt.tight_layout()
plt.savefig("static/ann_loss_curve.png", dpi=300, facecolor='#0f172a')
plt.close()
print("Saved: static/ann_loss_curve.png")

# 4.5 Confusion Matrices (3x4 Grid, N=1,272)
fig, axes = plt.subplots(3, 4, figsize=(18, 12), facecolor='#0f172a')
axes = axes.flatten()

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False,
                annot_kws={"size": 12, "weight": "bold"},
                xticklabels=["Low Risk", "High Risk"],
                yticklabels=["Low Risk", "High Risk"])
    axes[idx].set_title(f"{name} (Acc: {res['Accuracy']*100:.1f}%)", color='white', fontsize=10, fontweight="bold")
    axes[idx].set_xlabel("Predicted Label", color='#cbd5e1', fontsize=8.5)
    axes[idx].set_ylabel("True Label", color='#cbd5e1', fontsize=8.5)
    axes[idx].tick_params(colors='white')

for i in range(len(results), len(axes)):
    fig.delaxes(axes[i])

plt.suptitle(f"Model Confusion Matrices on Framingham Test Cohort (n={len(y_test)}, 70:30 Ratio)", fontsize=15, fontweight="bold", color='white', y=0.98)
plt.tight_layout()
plt.savefig("static/confusion_matrices.png", dpi=300, facecolor='#0f172a')
plt.close()
print("Saved: static/confusion_matrices.png")

# 4.6 Feature Importance
rf_importances = trained_models["Random Forest"].feature_importances_
feature_names_clean = [
    "Gender (Male)", "Age", "Education Level", "Current Smoker", "Cigarettes / Day",
    "BP Medication", "Prevalent Stroke", "Prevalent Hypertension", "Diabetes",
    "Total Cholesterol", "Systolic BP", "Diastolic BP", "BMI", "Heart Rate", "Glucose Level"
]
feat_df = pd.DataFrame({
    "Feature": feature_names_clean,
    "Importance": rf_importances
}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(10, 7), facecolor='#0f172a')
ax = plt.gca()
ax.set_facecolor('#1e293b')
bars = plt.barh(feat_df["Feature"], feat_df["Importance"], color='#38bdf8', edgecolor='#0284c7', height=0.65)
plt.title("Cardiovascular Risk Factor Importance (Clinical Drivers)", fontsize=14, fontweight="bold", color='white', pad=12)
plt.xlabel("Relative Predictive Importance Score", fontsize=11, color='white')
plt.tick_params(colors='white')
plt.grid(axis='x', linestyle=':', alpha=0.4, color='#64748b')

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.003, bar.get_y() + bar.get_height()/2, f"{width:.3f}",
             va='center', ha='left', color='white', fontsize=9, fontweight="bold")

plt.xlim(0, max(feat_df["Importance"]) * 1.15)
plt.tight_layout()
plt.savefig("static/feature_importance.png", dpi=300, facecolor='#0f172a')
plt.close()
print("Saved: static/feature_importance.png")

# -------------------------------------------------------------
# 5. Export JSON and CSV Metrics for Frontend Dashboard
# -------------------------------------------------------------
export_metrics = {}
export_rows = []

for name, res in results.items():
    export_metrics[name] = {
        "accuracy": round(float(res["Accuracy"]) * 100, 2),
        "sensitivity": round(float(res["Sensitivity"]) * 100, 2),
        "specificity": round(float(res["Specificity"]) * 100, 2),
        "error_rate": round(float(res["Percentage_Error"]), 2),
        "precision": round(float(res["Precision"]) * 100, 2),
        "recall": round(float(res["Sensitivity"]) * 100, 2),
        "f1_score": round(float(res["F1-Score"]) * 100, 2),
        "roc_auc": round(float(res["ROC-AUC"]) * 100, 2),
        "mae": round(float(res["MAE"]), 4),
        "mse": round(float(res["MSE"]), 4),
        "rmse": round(float(res["RMSE"]), 4),
        "training_time": f"{res['Training_Time']:.2f}s",
        "tp": res["TP"],
        "tn": res["TN"],
        "fp": res["FP"],
        "fn": res["FN"]
    }
    export_rows.append({
        "Model": name,
        "Accuracy (%)": round(float(res["Accuracy"]) * 100, 2),
        "Sensitivity (%)": round(float(res["Sensitivity"]) * 100, 2),
        "Specificity (%)": round(float(res["Specificity"]) * 100, 2),
        "Error Rate (%)": round(float(res["Percentage_Error"]), 2),
        "Precision (%)": round(float(res["Precision"]) * 100, 2),
        "F1-Score (%)": round(float(res["F1-Score"]) * 100, 2),
        "ROC-AUC (%)": round(float(res["ROC-AUC"]) * 100, 2),
        "MAE": round(float(res["MAE"]), 4),
        "MSE": round(float(res["MSE"]), 4),
        "RMSE": round(float(res["RMSE"]), 4),
        "Training Time": f"{res['Training_Time']:.2f}s"
    })

with open("static/model_comparison_metrics.json", "w") as f:
    json.dump(export_metrics, f, indent=4)
print("Saved: static/model_comparison_metrics.json")

df_export = pd.DataFrame(export_rows).sort_values(by="Accuracy (%)", ascending=False)
df_export.to_csv("static/model_comparison_metrics.csv", index=False)
print("Saved: static/model_comparison_metrics.csv")

print("\n" + "="*75)
print("PEAK HIGH-ACCURACY BENCHMARK TABLE (70:30 RATIO):")
print("="*75)
print(df_export.to_string(index=False))

print("\n" + "="*75)
print("ALL PEAK ACCURACY MODELS TRAINED, EVALUATED, AND SAVED SUCCESSFULLY!")
print("="*75)
