"""
╔══════════════════════════════════════════════════════════════════╗
║       INSURANCE FRAUD DETECTION - COMPLETE ML PROJECT           ║
║       Dataset: fraud_oracle.csv | Target: FraudFound_P          ║
╚══════════════════════════════════════════════════════════════════╝

PROBLEM STATEMENT:
─────────────────
This is a BINARY CLASSIFICATION problem.
Goal: Predict whether an insurance claim is FRAUDULENT (1) or LEGITIMATE (0).

DATASET OVERVIEW:
─────────────────
- 15,420 rows × 33 columns
- Target Column: FraudFound_P (0 = Legitimate, 1 = Fraud)
- Class Imbalance: ~94% Legitimate, ~6% Fraud
- No missing values (but many ordinal/categorical strings need encoding)

ALGORITHMS USED:
────────────────
1. Logistic Regression       → Baseline model
2. Random Forest Classifier  → Ensemble (handles imbalance well)
3. XGBoost Classifier        → Gradient Boosting (best for tabular fraud data)
4. SMOTE                     → Handles class imbalance

LIBRARIES:
──────────
- pandas, numpy              → Data manipulation
- matplotlib, seaborn        → Visualization
- scikit-learn               → Preprocessing, models, metrics
- xgboost                    → XGBoost model
- imbalanced-learn (imblearn)→ SMOTE oversampling
- joblib                     → Model saving/deployment
"""

# ═══════════════════════════════════════════════════════════════
# STEP 0: INSTALL DEPENDENCIES (run once)
# ═══════════════════════════════════════════════════════════════
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib

# ═══════════════════════════════════════════════════════════════
# STEP 1: IMPORT LIBRARIES
# ═══════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    BalancedAccuracyScore, f1_score, accuracy_score
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d2e',
    'axes.edgecolor':   '#3a3d52',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#b0b0b0',
    'ytick.color':      '#b0b0b0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d3e',
    'grid.linestyle':   '--',
    'grid.alpha':       0.4,
    'font.family':      'DejaVu Sans',
})

FRAUD_COLOR   = '#ff4757'
LEGIT_COLOR   = '#2ed573'
ACCENT_COLOR  = '#ffa502'
BLUE_COLOR    = '#1e90ff'
PURPLE_COLOR  = '#a29bfe'

OUTPUT_DIR = "fraud_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  INSURANCE FRAUD DETECTION — ML PIPELINE")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════
# STEP 2: LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("\n[1/8] Loading Dataset...")

df = pd.read_csv("fraud_oracle.csv")

print(f"  ✓ Shape        : {df.shape}")
print(f"  ✓ Columns      : {df.shape[1]}")
print(f"  ✓ Total Claims : {len(df):,}")
print(f"  ✓ Fraud Cases  : {df['FraudFound_P'].sum():,}  ({df['FraudFound_P'].mean()*100:.2f}%)")
print(f"  ✓ Legit Cases  : {(df['FraudFound_P']==0).sum():,}")
print(f"\n  Column Types:\n{df.dtypes.value_counts()}")

# ═══════════════════════════════════════════════════════════════
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA) + VISUALIZATION
# ═══════════════════════════════════════════════════════════════
print("\n[2/8] Exploratory Data Analysis & Visualization...")

# ── Figure 1: Overview Dashboard ──────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('INSURANCE FRAUD DETECTION — EDA DASHBOARD', 
             fontsize=18, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a. Class Distribution (Pie)
ax1 = fig.add_subplot(gs[0, 0])
sizes  = [df['FraudFound_P'].value_counts()[0], df['FraudFound_P'].value_counts()[1]]
labels = ['Legitimate\n(94%)', 'Fraud\n(6%)']
colors = [LEGIT_COLOR, FRAUD_COLOR]
wedges, texts, autotexts = ax1.pie(
    sizes, labels=labels, colors=colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': '#0f1117', 'linewidth': 2},
    textprops={'color': 'white', 'fontsize': 10}
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_color('#0f1117')
    at.set_fontweight('bold')
ax1.set_title('Class Distribution', fontsize=12, fontweight='bold', pad=10)

# 1b. Fraud by Accident Area
ax2 = fig.add_subplot(gs[0, 1])
area_fraud = df.groupby('AccidentArea')['FraudFound_P'].mean() * 100
bars = ax2.bar(area_fraud.index, area_fraud.values,
               color=[FRAUD_COLOR, BLUE_COLOR], edgecolor='white', linewidth=0.5)
ax2.set_title('Fraud Rate by Accident Area', fontsize=11, fontweight='bold')
ax2.set_ylabel('Fraud Rate (%)')
for bar, val in zip(bars, area_fraud.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', fontsize=10, color='white', fontweight='bold')
ax2.set_ylim(0, area_fraud.max() * 1.3)

# 1c. Fraud by Fault
ax3 = fig.add_subplot(gs[0, 2])
fault_fraud = df.groupby('Fault')['FraudFound_P'].mean() * 100
bars3 = ax3.bar(fault_fraud.index, fault_fraud.values,
                color=[PURPLE_COLOR, ACCENT_COLOR], edgecolor='white', linewidth=0.5)
ax3.set_title('Fraud Rate by Fault Party', fontsize=11, fontweight='bold')
ax3.set_ylabel('Fraud Rate (%)')
for bar, val in zip(bars3, fault_fraud.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', fontsize=10, color='white', fontweight='bold')
ax3.set_ylim(0, fault_fraud.max() * 1.3)

# 1d. Age Distribution
ax4 = fig.add_subplot(gs[1, 0])
fraud_ages = df[df['FraudFound_P'] == 1]['Age']
legit_ages  = df[df['FraudFound_P'] == 0]['Age']
ax4.hist(legit_ages, bins=30, alpha=0.7, color=LEGIT_COLOR, label='Legitimate', density=True)
ax4.hist(fraud_ages, bins=30, alpha=0.7, color=FRAUD_COLOR, label='Fraud',      density=True)
ax4.set_title('Age Distribution by Class', fontsize=11, fontweight='bold')
ax4.set_xlabel('Age')
ax4.set_ylabel('Density')
ax4.legend(fontsize=9)

# 1e. Fraud by Vehicle Category
ax5 = fig.add_subplot(gs[1, 1])
vc_fraud = df.groupby('VehicleCategory')['FraudFound_P'].mean() * 100
colors5  = [BLUE_COLOR, FRAUD_COLOR, LEGIT_COLOR, ACCENT_COLOR][:len(vc_fraud)]
bars5    = ax5.bar(vc_fraud.index, vc_fraud.values, color=colors5, edgecolor='white', linewidth=0.5)
ax5.set_title('Fraud Rate by Vehicle Category', fontsize=11, fontweight='bold')
ax5.set_ylabel('Fraud Rate (%)')
ax5.tick_params(axis='x', rotation=15)
for bar, val in zip(bars5, vc_fraud.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}%', ha='center', fontsize=9, color='white', fontweight='bold')

# 1f. Fraud by Policy Type
ax6 = fig.add_subplot(gs[1, 2])
pt_fraud = df.groupby('BasePolicy')['FraudFound_P'].mean() * 100
bars6    = ax6.barh(pt_fraud.index, pt_fraud.values,
                    color=[PURPLE_COLOR, ACCENT_COLOR, FRAUD_COLOR], edgecolor='white', linewidth=0.5)
ax6.set_title('Fraud Rate by Base Policy', fontsize=11, fontweight='bold')
ax6.set_xlabel('Fraud Rate (%)')
for bar, val in zip(bars6, pt_fraud.values):
    ax6.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=10, color='white', fontweight='bold')

# 1g. Police Report Filed vs Fraud
ax7 = fig.add_subplot(gs[2, 0])
pr_data = df.groupby(['PoliceReportFiled', 'FraudFound_P']).size().unstack()
pr_data.plot(kind='bar', ax=ax7, color=[LEGIT_COLOR, FRAUD_COLOR],
             edgecolor='white', linewidth=0.5)
ax7.set_title('Police Report vs Fraud', fontsize=11, fontweight='bold')
ax7.set_xlabel('Police Report Filed')
ax7.set_ylabel('Count')
ax7.legend(['Legitimate', 'Fraud'], fontsize=9)
ax7.tick_params(axis='x', rotation=0)

# 1h. Witness Present vs Fraud
ax8 = fig.add_subplot(gs[2, 1])
wp_data = df.groupby(['WitnessPresent', 'FraudFound_P']).size().unstack()
wp_data.plot(kind='bar', ax=ax8, color=[LEGIT_COLOR, FRAUD_COLOR],
             edgecolor='white', linewidth=0.5)
ax8.set_title('Witness Present vs Fraud', fontsize=11, fontweight='bold')
ax8.set_xlabel('Witness Present')
ax8.set_ylabel('Count')
ax8.legend(['Legitimate', 'Fraud'], fontsize=9)
ax8.tick_params(axis='x', rotation=0)

# 1i. Driver Rating Distribution
ax9 = fig.add_subplot(gs[2, 2])
for cls, color, label in [(0, LEGIT_COLOR, 'Legitimate'), (1, FRAUD_COLOR, 'Fraud')]:
    vals = df[df['FraudFound_P'] == cls]['DriverRating']
    ax9.hist(vals, bins=6, alpha=0.7, color=color, label=label, density=True)
ax9.set_title('Driver Rating Distribution', fontsize=11, fontweight='bold')
ax9.set_xlabel('Driver Rating')
ax9.set_ylabel('Density')
ax9.legend(fontsize=9)

plt.savefig(f"{OUTPUT_DIR}/01_eda_dashboard.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  ✓ EDA dashboard saved")

# ═══════════════════════════════════════════════════════════════
# STEP 4: DATA CLEANING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════
print("\n[3/8] Data Cleaning & Preprocessing...")

df_clean = df.copy()

# Drop irrelevant identifier columns
df_clean.drop(columns=['PolicyNumber', 'RepNumber'], inplace=True)
print("  ✓ Dropped identifier columns: PolicyNumber, RepNumber")

# Check missing values
missing = df_clean.isnull().sum()
print(f"  ✓ Missing values: {missing.sum()} (dataset is clean!)")

# ── FEATURE ENGINEERING ───────────────────────────────────────
# Map ordinal string columns to numeric ranks

# VehiclePrice ordinal mapping
price_map = {
    'less than 20000': 1, '20000 to 29000': 2, '30000 to 39000': 3,
    '40000 to 59000': 4, '60000 to 69000': 5, 'more than 69000': 6
}
df_clean['VehiclePrice'] = df_clean['VehiclePrice'].map(price_map)

# Days_Policy_Accident ordinal
dpa_map = {'none': 0, '1 to 7': 1, '8 to 15': 2, '15 to 30': 3, 'more than 30': 4}
df_clean['Days_Policy_Accident'] = df_clean['Days_Policy_Accident'].map(dpa_map)

# Days_Policy_Claim ordinal
dpc_map = {'none': 0, '8 to 15': 1, '15 to 30': 2, 'more than 30': 3}
df_clean['Days_Policy_Claim'] = df_clean['Days_Policy_Claim'].map(dpc_map)

# PastNumberOfClaims ordinal
pnc_map = {'none': 0, '1': 1, '2 to 4': 2, 'more than 4': 3}
df_clean['PastNumberOfClaims'] = df_clean['PastNumberOfClaims'].map(pnc_map)

# AgeOfVehicle ordinal
aov_map = {'new': 0, '2 years': 1, '3 years': 2, '4 years': 3,
           '5 years': 4, '6 years': 5, '7 years': 6, 'more than 7': 7}
df_clean['AgeOfVehicle'] = df_clean['AgeOfVehicle'].map(aov_map)

# AgeOfPolicyHolder ordinal
aph_map = {'16 to 17': 0, '18 to 20': 1, '21 to 25': 2, '26 to 30': 3,
           '31 to 35': 4, '36 to 40': 5, '41 to 50': 6, '51 to 65': 7, 'over 65': 8}
df_clean['AgeOfPolicyHolder'] = df_clean['AgeOfPolicyHolder'].map(aph_map)

# NumberOfSuppliments ordinal
nos_map = {'none': 0, '1 to 2': 1, '3 to 5': 2, 'more than 5': 3}
df_clean['NumberOfSuppliments'] = df_clean['NumberOfSuppliments'].map(nos_map)

# AddressChange_Claim ordinal
acc_map = {'no change': 0, 'under 6 months': 1, '1 year': 2, '2 to 3 years': 3, '4 to 8 years': 4}
df_clean['AddressChange_Claim'] = df_clean['AddressChange_Claim'].map(acc_map)

# NumberOfCars ordinal
noc_map = {'1 vehicle': 1, '2 vehicles': 2, '3 to 4': 3, '5 to 8': 4, 'more than 8': 5}
df_clean['NumberOfCars'] = df_clean['NumberOfCars'].map(noc_map)

# Binary mappings
binary_cols = ['Sex', 'AccidentArea', 'PoliceReportFiled', 'WitnessPresent', 'Fault', 'AgentType']
binary_maps = {
    'Sex':              {'Male': 0, 'Female': 1},
    'AccidentArea':     {'Urban': 0, 'Rural': 1},
    'PoliceReportFiled':{'No': 0, 'Yes': 1},
    'WitnessPresent':   {'No': 0, 'Yes': 1},
    'Fault':            {'Policy Holder': 0, 'Third Party': 1},
    'AgentType':        {'External': 0, 'Internal': 1},
}
for col, mapping in binary_maps.items():
    df_clean[col] = df_clean[col].map(mapping)

print("  ✓ Ordinal & binary encoding applied")

# Label encode remaining categorical columns
le = LabelEncoder()
cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
print(f"  ✓ Label encoding: {cat_cols}")
for col in cat_cols:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# Fill any NaN created by unmapped values (rare edge cases)
null_after = df_clean.isnull().sum().sum()
if null_after > 0:
    print(f"  ⚠ {null_after} NaN values from unmapped categories → filled with median")
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
else:
    print("  ✓ No NaN values after encoding")

print(f"  ✓ Final shape: {df_clean.shape}")

# ── Correlation Heatmap ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 14))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')
corr = df_clean.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='coolwarm', center=0,
            annot=False, linewidths=0.3, linecolor='#0f1117',
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color='white', pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  ✓ Correlation heatmap saved")

# ═══════════════════════════════════════════════════════════════
# STEP 5: FEATURE SELECTION & DATA SPLITTING
# ═══════════════════════════════════════════════════════════════
print("\n[4/8] Feature Selection & Train-Test Split...")

X = df_clean.drop(columns=['FraudFound_P'])
y = df_clean['FraudFound_P']

print(f"  ✓ Features (X): {X.shape}")
print(f"  ✓ Target  (y): {y.shape}")
print(f"  ✓ Fraud in dataset: {y.sum()} / {len(y)} ({y.mean()*100:.2f}%)")

# Split: 80% train, 20% test (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  ✓ Train set: {X_train.shape} | Fraud: {y_train.sum()}")
print(f"  ✓ Test  set: {X_test.shape}  | Fraud: {y_test.sum()}")

# ─── Feature Scaling ──────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("  ✓ StandardScaler applied")

# ─── SMOTE: Handle Class Imbalance ────────────────────────────
# SMOTE creates synthetic minority-class (Fraud) samples
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"  ✓ SMOTE applied → Balanced train: {X_train_res.shape}")
print(f"    Fraud after SMOTE: {y_train_res.sum()} | Legit: {(y_train_res==0).sum()}")

# ═══════════════════════════════════════════════════════════════
# STEP 6: MODEL BUILDING & TRAINING
# ═══════════════════════════════════════════════════════════════
print("\n[5/8] Model Building & Training...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1, class_weight='balanced'
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=15,        # handles imbalance (negatives/positives ratio)
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=-1
    )
}

trained_models = {}
results = {}

for name, model in models.items():
    print(f"\n  ▶ Training {name}...")
    model.fit(X_train_res, y_train_res)
    trained_models[name] = model

    # Predictions
    y_pred      = model.predict(X_test_scaled)
    y_prob      = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    ap      = average_precision_score(y_test, y_prob)

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': acc, 'f1': f1, 'roc_auc': roc_auc, 'avg_precision': ap
    }

    print(f"    Accuracy       : {acc:.4f}")
    print(f"    F1 Score       : {f1:.4f}")
    print(f"    ROC-AUC        : {roc_auc:.4f}")
    print(f"    Avg Precision  : {ap:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Fraud'])}")

# ═══════════════════════════════════════════════════════════════
# STEP 7: MODEL EVALUATION & PERFORMANCE VISUALIZATION
# ═══════════════════════════════════════════════════════════════
print("\n[6/8] Model Evaluation & Visualization...")

# ── Figure 3: Performance Comparison ──────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('MODEL PERFORMANCE EVALUATION', fontsize=16, fontweight='bold', color='white', y=1.01)

model_names  = list(results.keys())
colors_model = [BLUE_COLOR, LEGIT_COLOR, FRAUD_COLOR]

# Subplot 1: Metric Bar Chart
ax = axes[0, 0]
ax.set_facecolor('#1a1d2e')
metrics      = ['accuracy', 'f1', 'roc_auc', 'avg_precision']
metric_names = ['Accuracy', 'F1 Score', 'ROC-AUC', 'Avg Precision']
x_pos = np.arange(len(metric_names))
width = 0.25
for i, (mname, color) in enumerate(zip(model_names, colors_model)):
    vals = [results[mname][m] for m in metrics]
    bars = ax.bar(x_pos + i*width, vals, width=width, label=mname, color=color,
                  alpha=0.85, edgecolor='white', linewidth=0.5)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.15)
ax.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylabel('Score')
ax.yaxis.grid(True)

# Subplot 2-4: Confusion Matrices
for i, (name, color) in enumerate(zip(model_names, colors_model)):
    ax = axes[0, i+0] if i == 0 else axes[1, i-1]
    ax = axes[1, i]
    ax.set_facecolor('#1a1d2e')
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette(color, as_cmap=True),
                cbar=False, linewidths=1, linecolor='#0f1117',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    ax.set_title(f'Confusion Matrix — {name}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)

# Subplot: ROC Curves
ax_roc = axes[0, 1]
ax_roc.set_facecolor('#1a1d2e')
ax_roc.plot([0,1],[0,1], 'w--', lw=1, alpha=0.4, label='Random Baseline')
for name, color in zip(model_names, colors_model):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    auc_val = results[name]['roc_auc']
    ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")
ax_roc.set_xlim([0, 1])
ax_roc.set_ylim([0, 1.02])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curves', fontsize=12, fontweight='bold')
ax_roc.legend(fontsize=9)
ax_roc.yaxis.grid(True)

# Subplot: Precision-Recall Curves
ax_pr = axes[0, 2]
ax_pr.set_facecolor('#1a1d2e')
for name, color in zip(model_names, colors_model):
    prec, rec, _ = precision_recall_curve(y_test, results[name]['y_prob'])
    ap_val = results[name]['avg_precision']
    ax_pr.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap_val:.3f})")
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision')
ax_pr.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
ax_pr.legend(fontsize=9)
ax_pr.yaxis.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_model_performance.png", dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  ✓ Performance charts saved")

# ── Figure 4: Feature Importance (XGBoost + Random Forest) ───
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('FEATURE IMPORTANCE ANALYSIS', fontsize=14, fontweight='bold', color='white')

for ax, mname, color in zip(axes,
        ['XGBoost', 'Random Forest'],
        [FRAUD_COLOR, LEGIT_COLOR]):
    ax.set_facecolor('#1a1d2e')
    model   = trained_models[mname]
    imp     = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': imp})
    feat_df = feat_df.sort_values('Importance', ascending=True).tail(15)
    ax.barh(feat_df['Feature'], feat_df['Importance'], color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_title(f'{mname} — Top 15 Features', fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.xaxis.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_feature_importance.png", dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  ✓ Feature importance charts saved")

# ── Figure 5: Fraud Probability Distribution ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('FRAUD PROBABILITY SCORE DISTRIBUTION', fontsize=14, fontweight='bold', color='white')

for ax, name, color in zip(axes, model_names, colors_model):
    ax.set_facecolor('#1a1d2e')
    probs = results[name]['y_prob']
    ax.hist(probs[y_test == 0], bins=40, color=LEGIT_COLOR, alpha=0.7, label='Legitimate', density=True)
    ax.hist(probs[y_test == 1], bins=40, color=FRAUD_COLOR,  alpha=0.7, label='Fraud',      density=True)
    ax.axvline(0.5, color='white', linestyle='--', lw=1.5, label='Threshold=0.5')
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Fraud Probability')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_probability_distribution.png", dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  ✓ Probability distribution chart saved")

# ── Best Model ────────────────────────────────────────────────
best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
best_model      = trained_models[best_model_name]
print(f"\n  🏆 Best Model (by ROC-AUC): {best_model_name}")
print(f"     ROC-AUC = {results[best_model_name]['roc_auc']:.4f}")

# ═══════════════════════════════════════════════════════════════
# STEP 8: CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════
print("\n[7/8] Cross-Validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_res, y_train_res,
                             cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"  ✓ {best_model_name} 5-Fold CV ROC-AUC:")
print(f"    Scores : {cv_scores.round(4)}")
print(f"    Mean   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ═══════════════════════════════════════════════════════════════
# STEP 9: SAVE & DEPLOYMENT
# ═══════════════════════════════════════════════════════════════
print("\n[8/8] Saving Models & Deployment Artifacts...")

# Save best model
joblib.dump(best_model, f"{OUTPUT_DIR}/best_model_{best_model_name.replace(' ','_')}.pkl")
print(f"  ✓ Best model saved: best_model_{best_model_name.replace(' ','_')}.pkl")

# Save all models
for name, model in trained_models.items():
    fname = f"{OUTPUT_DIR}/model_{name.replace(' ','_')}.pkl"
    joblib.dump(model, fname)
print(f"  ✓ All models saved to {OUTPUT_DIR}/")

# Save scaler
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
print(f"  ✓ Scaler saved: scaler.pkl")

# Save feature names
feature_names = list(X.columns)
joblib.dump(feature_names, f"{OUTPUT_DIR}/feature_names.pkl")
print(f"  ✓ Feature names saved")

# ── Deployment Prediction Function ────────────────────────────
def predict_fraud(new_claim_df: pd.DataFrame,
                  model_path: str = f"{OUTPUT_DIR}/best_model_{best_model_name.replace(' ','_')}.pkl",
                  scaler_path: str = f"{OUTPUT_DIR}/scaler.pkl",
                  features_path: str = f"{OUTPUT_DIR}/feature_names.pkl") -> pd.DataFrame:
    """
    Predict fraud probability for new insurance claims.

    Parameters:
    ───────────
    new_claim_df : pd.DataFrame — New claims (preprocessed, same columns as training)
    model_path   : str          — Path to saved model .pkl
    scaler_path  : str          — Path to saved scaler .pkl
    features_path: str          — Path to saved feature names .pkl

    Returns:
    ────────
    pd.DataFrame with columns: prediction, fraud_probability, risk_level
    """
    loaded_model   = joblib.load(model_path)
    loaded_scaler  = joblib.load(scaler_path)
    loaded_features= joblib.load(features_path)

    # Align columns
    new_claim_df = new_claim_df[loaded_features]

    # Scale
    X_scaled = loaded_scaler.transform(new_claim_df)

    # Predict
    pred  = loaded_model.predict(X_scaled)
    proba = loaded_model.predict_proba(X_scaled)[:, 1]

    result = pd.DataFrame({
        'prediction':        pred,
        'fraud_probability': proba.round(4),
        'risk_level':        pd.cut(proba, bins=[0, 0.3, 0.6, 1.0],
                                     labels=['Low', 'Medium', 'High'])
    })
    return result


# ── Demo: Predict on 5 test samples ───────────────────────────
print("\n  ── DEMO PREDICTIONS (5 test samples) ──")
sample_X = X_test.iloc[:5]
demo_preds = predict_fraud(sample_X)
demo_preds['actual'] = y_test.values[:5]
print(demo_preds.to_string(index=False))

# ── Final Summary Table ────────────────────────────────────────
print("\n" + "═"*65)
print("  FINAL MODEL COMPARISON SUMMARY")
print("═"*65)
print(f"  {'Model':<25} {'Accuracy':>9} {'F1 Score':>10} {'ROC-AUC':>9} {'Avg Precision':>15}")
print("  " + "─"*63)
for name in model_names:
    r = results[name]
    print(f"  {name:<25} {r['accuracy']:>9.4f} {r['f1']:>10.4f} {r['roc_auc']:>9.4f} {r['avg_precision']:>15.4f}")

print("═"*65)
print(f"\n  🏆 BEST MODEL   : {best_model_name}")
print(f"  📁 OUTPUT DIR   : ./{OUTPUT_DIR}/")
print(f"  💾 SAVED FILES  : best_model, all_models, scaler, feature_names")
print(f"\n  All visualizations saved to ./{OUTPUT_DIR}/")
print("\n✅ PIPELINE COMPLETE!")
