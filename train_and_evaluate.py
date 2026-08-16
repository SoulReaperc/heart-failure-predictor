import os
import sys
import site
sys.path.append(site.getusersitepackages())

import json
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

# ML Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

print("="*70)
print("1. Loading Framingham Heart Study Dataset...")
print("="*70)
df = pd.read_csv("framingham.csv")
print(f"Dataset shape: {df.shape}")

# Features and Target
feature_cols = [
    "male", "age", "education", "currentSmoker", "cigsPerDay", 
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", 
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]
target_col = "TenYearCHD"

X = df[feature_cols]
y = df[target_col]

# Train-Test Split (Stratified 80/20)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

# Imputation (Median strategy to avoid data leakage)
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_raw)
X_test_imp = imputer.transform(X_test_raw)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_imp)
X_test = scaler.transform(X_test_imp)

# Save preprocessors
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved imputer.pkl and scaler.pkl successfully.")

# Calculate class imbalance ratio for positive class weighting
pos_count = int((y_train == 1).sum())
neg_count = int((y_train == 0).sum())
pos_weight_val = float(neg_count / pos_count)
print(f"Class distribution in training: 0={neg_count}, 1={pos_count} (Pos Weight: {pos_weight_val:.2f})")

# Results container
results = {}

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# -------------------------------------------------------------
# 2. Deep Learning Models: ANN, LSTM, GRU
# -------------------------------------------------------------
print("\n" + "="*70)
print("2. Training Deep Learning Architectures (ANN, LSTM, GRU)...")
print("="*70)

# A) Artificial Neural Network (ANN / MLP)
class CardiovascularANN(nn.Module):
    def __init__(self, input_dim=15):
        super(CardiovascularANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.network(x)

# B) Long Short-Term Memory (LSTM)
class CardiovascularLSTM(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=32, num_layers=2):
        super(CardiovascularLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        x_seq = x.unsqueeze(2) # (batch_size, 15, 1)
        _, (h_n, _) = self.lstm(x_seq)
        out = self.fc(h_n[-1])
        return out

# C) Gated Recurrent Unit (GRU)
class CardiovascularGRU(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=32, num_layers=2):
        super(CardiovascularGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=1, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        x_seq = x.unsqueeze(2) # (batch_size, 15, 1)
        _, h_n = self.gru(x_seq)
        out = self.fc(h_n[-1])
        return out

deep_models = {
    "Artificial Neural Network (ANN)": (CardiovascularANN(15), "ann_model.pth", 0.003),
    "Long Short-Term Memory (LSTM)": (CardiovascularLSTM(15), "lstm_model.pth", 0.004),
    "Gated Recurrent Unit (GRU)": (CardiovascularGRU(15), "gru_model.pth", 0.004)
}

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
dl_loss_curves = {}

for name, (model, save_path, lr) in deep_models.items():
    print(f"\n--- Training {name} ---")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    epochs = 60
    train_losses = []
    val_losses = []
    
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
            
    dl_loss_curves[name] = {"train": train_losses, "val": val_losses}
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        raw_logits = model(X_test_t)
        probs = torch.sigmoid(raw_logits).numpy().flatten()
        preds = (probs >= 0.50).astype(int)
        
    mae = mean_absolute_error(y_test, probs)
    mse = mean_squared_error(y_test, probs)
    rmse = np.sqrt(mse)

    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1-Score": f1_score(y_test, preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probs),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "y_prob": probs,
        "y_pred": preds
    }
    
    torch.save(model.state_dict(), save_path)
    print(f"Saved {name} weights to {save_path}")

# -------------------------------------------------------------
# 3. Classical & Ensemble ML Baselines
# -------------------------------------------------------------
print("\n" + "="*70)
print("3. Training Baseline Classifiers (Random Forest, XGBoost, SVM, etc.)...")
print("="*70)

classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, scale_pos_weight=pos_weight_val, random_state=SEED, eval_metric="logloss"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    "Support Vector Machine (SVM)": SVC(probability=True, class_weight="balanced", kernel="rbf", C=1.0, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED)
}

trained_models = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred
    
    mae = mean_absolute_error(y_test, y_prob)
    mse = mean_squared_error(y_test, y_prob)
    rmse = np.sqrt(mse)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "y_prob": y_prob,
        "y_pred": y_pred
    }
    trained_models[name] = clf
    print(f"{name:30s} -> Accuracy: {results[name]['Accuracy']:.4f} | ROC-AUC: {results[name]['ROC-AUC']:.4f} | Recall: {results[name]['Recall']:.4f}")

# Save primary Random Forest baseline
joblib.dump(trained_models["Random Forest"], "heart_model.pkl")
print("Saved Random Forest to heart_model.pkl")

# -------------------------------------------------------------
# 4. Generate Publication-Quality Visualizations
# -------------------------------------------------------------
print("\n" + "="*70)
print("4. Generating Diagnostic & Comparative Visualizations...")
print("="*70)

sns.set_theme(style="darkgrid", palette="muted")

# 1. Model Comparison Bar Chart
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
model_names_list = list(results.keys())
df_metrics = pd.DataFrame({
    m: [results[model][m] for model in model_names_list]
    for m in metrics_names
}, index=model_names_list)

fig, ax = plt.subplots(figsize=(15, 7.5))
df_metrics.plot(kind="bar", ax=ax, colormap="viridis", width=0.84)
plt.title("Cardiovascular Risk Prediction - Comparative Model Performance", fontsize=15, fontweight="bold", pad=15)
plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
plt.xlabel("Model Architecture (ANN, LSTM, GRU & Baselines)", fontsize=12)
plt.xticks(rotation=30, ha="right", fontsize=10, fontweight="bold")
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, facecolor='#1e293b', labelcolor='white')
plt.tight_layout()
plt.savefig("static/model_comparison.png", dpi=300, facecolor='#0f172a', edgecolor='none')
plt.close()
print("Saved: static/model_comparison.png")

# 2. Multi-Model ROC Curves
plt.figure(figsize=(11, 8.5), facecolor='#0f172a')
ax = plt.gca()
ax.set_facecolor('#1e293b')
colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(results)))

for idx, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, color=colors[idx], lw=2.2, label=f"{name} (AUC = {res['ROC-AUC']:.3f})")

plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.500)')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, color='white')
plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, color='white')
plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold", color='white', pad=12)
plt.tick_params(colors='white')
plt.legend(loc="lower right", facecolor='#1e293b', edgecolor='#475569', labelcolor='white', fontsize=9.5)
plt.grid(True, linestyle=':', alpha=0.4, color='#64748b')
plt.tight_layout()
plt.savefig("static/roc_curves.png", dpi=300, facecolor='#0f172a')
plt.close()
print("Saved: static/roc_curves.png")

# 3. Deep Learning Loss Convergence Curves (ANN vs LSTM vs GRU)
plt.figure(figsize=(10, 6), facecolor='#0f172a')
ax = plt.gca()
ax.set_facecolor('#1e293b')

plt.plot(range(1, epochs + 1), dl_loss_curves["Artificial Neural Network (ANN)"]["train"], label="ANN Train Loss", color='#38bdf8', lw=2)
plt.plot(range(1, epochs + 1), dl_loss_curves["Artificial Neural Network (ANN)"]["val"], label="ANN Val Loss", color='#38bdf8', lw=2, linestyle='--')

plt.plot(range(1, epochs + 1), dl_loss_curves["Long Short-Term Memory (LSTM)"]["train"], label="LSTM Train Loss", color='#34d399', lw=2)
plt.plot(range(1, epochs + 1), dl_loss_curves["Long Short-Term Memory (LSTM)"]["val"], label="LSTM Val Loss", color='#34d399', lw=2, linestyle='--')

plt.plot(range(1, epochs + 1), dl_loss_curves["Gated Recurrent Unit (GRU)"]["train"], label="GRU Train Loss", color='#f59e0b', lw=2)
plt.plot(range(1, epochs + 1), dl_loss_curves["Gated Recurrent Unit (GRU)"]["val"], label="GRU Val Loss", color='#f59e0b', lw=2, linestyle='--')

plt.title("Deep Learning Architectures (ANN, LSTM, GRU) Training Dynamics", fontsize=14, fontweight="bold", color='white', pad=12)
plt.xlabel("Epochs", fontsize=12, color='white')
plt.ylabel("Weighted Binary Cross-Entropy Loss", fontsize=12, color='white')
plt.tick_params(colors='white')
plt.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='white', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.4, color='#64748b')
plt.tight_layout()
plt.savefig("static/ann_loss_curve.png", dpi=300, facecolor='#0f172a')
plt.close()
print("Saved: static/ann_loss_curve.png")

# 4. Confusion Matrices (3x4 Grid)
fig, axes = plt.subplots(3, 4, figsize=(18, 12), facecolor='#0f172a')
axes = axes.flatten()

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False,
                annot_kws={"size": 12, "weight": "bold"},
                xticklabels=["Low Risk", "High Risk"],
                yticklabels=["Low Risk", "High Risk"])
    axes[idx].set_title(name, color='white', fontsize=10.5, fontweight="bold")
    axes[idx].set_xlabel("Predicted Label", color='#cbd5e1', fontsize=8.5)
    axes[idx].set_ylabel("True Label", color='#cbd5e1', fontsize=8.5)
    axes[idx].tick_params(colors='white')

for i in range(len(results), len(axes)):
    fig.delaxes(axes[i])

plt.suptitle("Model Confusion Matrices on Framingham Test Cohort (n=848)", fontsize=15, fontweight="bold", color='white', y=0.98)
plt.tight_layout()
plt.savefig("static/confusion_matrices.png", dpi=300, facecolor='#0f172a')
plt.close()
print("Saved: static/confusion_matrices.png")

# 5. Feature Importance Analysis
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
# 5. Export JSON Metrics for Frontend Dashboard
# -------------------------------------------------------------
export_metrics = {}
for name, res in results.items():
    export_metrics[name] = {
        "accuracy": round(float(res["Accuracy"]) * 100, 2),
        "precision": round(float(res["Precision"]) * 100, 2),
        "recall": round(float(res["Recall"]) * 100, 2),
        "f1_score": round(float(res["F1-Score"]) * 100, 2),
        "roc_auc": round(float(res["ROC-AUC"]) * 100, 2),
        "mae": round(float(res["MAE"]), 4),
        "mse": round(float(res["MSE"]), 4),
        "rmse": round(float(res["RMSE"]), 4)
    }

with open("static/model_comparison_metrics.json", "w") as f:
    json.dump(export_metrics, f, indent=4)
print("Saved: static/model_comparison_metrics.json")

print("\n" + "="*70)
print("ALL DEEP LEARNING & ML MODELS TRAINED & EVALUATED SUCCESSFULLY!")
print("="*70)
