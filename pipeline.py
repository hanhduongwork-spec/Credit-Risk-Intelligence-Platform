"""
Credit Risk Pipeline — Lending Club
=====================================
Converted from Final_BA_pipeline.ipynb

Run: python pipeline.py
"""

import os
import warnings
import joblib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
CLEANED_DATA_PATH = "lending_club_loan_cleaned.csv"
MODEL_DIR         = "models"
DATA_DIR          = "data"
RANDOM_STATE      = 100
N_TRIALS          = 20
LGD_BASE          = 0.70
ORIGINATION_FEE   = 0.03


# ============================================================
# Platt Calibrator — compatible with all sklearn versions
# ============================================================
class PlattCalibrator:
    """
    Manual Platt Scaling (sigmoid calibration).
    Fits a logistic regression on top of the base model's raw probabilities.
    Avoids sklearn version conflicts with CalibratedClassifierCV.
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self.sig        = LogisticRegression(max_iter=1000)

    def fit(self, X_val, y_val):
        raw = self.base_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
        self.sig.fit(raw, y_val)
        return self

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self.sig.predict_proba(raw)[:, 1]
        return np.column_stack([1 - cal, cal])


# ============================================================
# PROFIT FUNCTIONS (shared across all scenarios)
# ============================================================
def calculate_total_income(loan_amnt, int_rate, term_months, fee=ORIGINATION_FEE):
    """Total income = interest income + origination fee."""
    r = int_rate / 100 / 12
    n = int(term_months)
    pmt = loan_amnt / n if r == 0 else loan_amnt * (r*(1+r)**n) / ((1+r)**n - 1)
    return (pmt * n - loan_amnt) + loan_amnt * fee

def monte_carlo_profit(PD, loan_amnt, int_rate, term_months, lgd=LGD_BASE, N=10_000):
    """Simulate profit/loss for a single loan using Monte Carlo."""
    income  = calculate_total_income(loan_amnt, int_rate, term_months)
    loss    = loan_amnt * lgd
    profits = np.where(np.random.binomial(1, PD, N) == 1, -loss, income)
    return {
        'mean_profit': profits.mean(),
        'std_profit' : profits.std(),
        'var_95'     : np.percentile(profits, 5),
        'prob_profit': (profits > 0).mean()
    }


# ============================================================
# EVALUATION HELPER
# ============================================================
def evaluate_model(model, X_test, y_true, show_plot=False):
    """Compute and print standard classification metrics."""
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_true, pred)
    precision = precision_score(y_true, pred, zero_division=0)
    recall    = recall_score(y_true, pred, zero_division=0)
    f1        = f1_score(y_true, pred, zero_division=0)
    f2        = fbeta_score(y_true, pred, beta=2, zero_division=0)
    auc       = roc_auc_score(y_true, prob)

    print(f"  AUC: {auc:.4f} | Accuracy: {accuracy:.4f} | F2: {f2:.4f}")
    print(classification_report(y_true, pred, zero_division=0))

    if show_plot:
        cf = confusion_matrix(y_true, pred)
        import seaborn as sns
        cf_df = pd.DataFrame(cf, index=['Actual: Fully Paid', 'Actual: Charged Off'],
                             columns=['Pred: Fully Paid', 'Pred: Charged Off'])
        plt.figure(figsize=(4, 3))
        sns.heatmap(cf_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix'); plt.tight_layout(); plt.show()

    return {'auc': auc, 'f2': f2, 'precision': precision, 'recall': recall,
            'accuracy': accuracy, 'f1': f1}


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
def load_data(path=CLEANED_DATA_PATH):
    print("=" * 60)
    print("STEP 1 — LOAD DATA")
    print("=" * 60)
    df = pd.read_csv(path)
    print(f"✅ Loaded: {df.shape}")
    return df


# ============================================================
# STEP 2 — FEATURE ENGINEERING
# ============================================================
def feature_engineering(df):
    """
    Create new features:
    - term_months: numeric term from string
    - loan_to_revol_ratio: loan amount / revolving utilization
    - credit_age_years: age of credit history
    - total_bad_records: sum of public records
    - cbrt_*: cube root transforms for skewed distributions
    """
    print("\n" + "=" * 60)
    print("STEP 2 — FEATURE ENGINEERING")
    print("=" * 60)
    df = df.copy()

    # Convert term to numeric months
    if 'term' in df.columns:
        df['term_months'] = pd.to_numeric(
            df['term'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

    # Loan-to-revolving ratio (avoids multicollinearity vs loan_income_ratio)
    if {'loan_amnt', 'revol_util'}.issubset(df.columns):
        df['loan_to_revol_ratio'] = df['loan_amnt'] / (df['revol_util'] + 1)

    # Credit history age
    if 'earliest_cr_line' in df.columns:
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
        df['credit_age_years'] = 2026 - df['earliest_cr_line'].dt.year

    # Total derogatory records
    if {'pub_rec', 'pub_rec_bankruptcies'}.issubset(df.columns):
        df['total_bad_records'] = df['pub_rec'] + df['pub_rec_bankruptcies']

    # Cube root transforms (already done in preprocessing, skip if present)
    for col in ['annual_inc', 'open_acc', 'revol_bal', 'total_acc', 'mort_acc']:
        if col in df.columns and f'cbrt_{col}' not in df.columns:
            df[f'cbrt_{col}'] = np.cbrt(df[col])

    print(f"✅ Done: {df.shape[1]} columns")
    return df


# ============================================================
# STEP 3 — ENCODE & SPLIT
# ============================================================
def encode_and_split(df):
    """
    One-hot encode categorical features, create target, train/test split.
    Returns X_train, X_test, y_train, y_test, df_with_subgrade.
    """
    print("\n" + "=" * 60)
    print("STEP 3 — ENCODE & SPLIT")
    print("=" * 60)

    df_work = df.copy()

    # Create target variable
    df_work['default_flag'] = df_work['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

    # One-hot encoding
    cat_cols = ['sub_grade', 'home_ownership', 'verification_status',
                'purpose', 'initial_list_status', 'application_type']
    existing = [c for c in cat_cols if c in df_work.columns]
    dummies  = pd.get_dummies(df_work[existing], drop_first=True, dtype=int)

    # Drop columns not needed for modeling
    drop_cols = [c for c in existing + ['loan_status', 'term', 'earliest_cr_line',
                                         'grade', 'address', 'issue_d']
                 if c in df_work.columns]
    df_model = pd.concat([df_work.drop(columns=drop_cols), dummies], axis=1)
    df_model = df_model[df_model['default_flag'].notna()]

    X = df_model.drop('default_flag', axis=1)
    y = df_model['default_flag'].astype(int)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Convert bool columns to int
    for col in X_train.select_dtypes(include=['bool']).columns:
        X_train[col] = X_train[col].astype(int)
        X_test[col]  = X_test[col].astype(int)

    # Impute missing values with train median (prevent data leakage)
    for col in X_train.select_dtypes(include=['float64', 'int64']).columns:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col]  = X_test[col].fillna(median_val)

    print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"   Default rate — Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")
    print(f"   NaNs in X_train: {X_train.isnull().sum().sum()} | X_test: {X_test.isnull().sum().sum()}")
    return X_train, X_test, y_train, y_test, df_work


# ============================================================
# STEP 4 — BASELINE MODELS (3 models × 2 variants)
# ============================================================
def run_baseline(X_train, X_test, y_train, y_test):
    """
    Train 6 baseline models: LightGBM, XGBoost, Logistic Regression
    each with Full Features and Top 15 Features.
    Returns the best model name and variant.
    """
    print("\n" + "=" * 60)
    print("STEP 4 — BASELINE MODELS (3 × 2 variants)")
    print("=" * 60)

    scale_pos = y_train.value_counts()[0] / y_train.value_counts()[1]

    # ── LightGBM ──────────────────────────────────────────
    print("\n--- LightGBM ---")
    lgb_full = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE,
        is_unbalance=True, colsample_bytree=0.7, subsample=0.8, verbose=-1, n_jobs=-1)
    lgb_full.fit(X_train, y_train)
    lgbm_full_metrics = evaluate_model(lgb_full, X_test, y_test)
    print(f"  LightGBM Full → F2: {lgbm_full_metrics['f2']:.4f} | AUC: {lgbm_full_metrics['auc']:.4f}")

    # Top 15 features from LightGBM importance
    top_features = pd.Series(
        lgb_full.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False).head(15).index.tolist()

    lgb_top = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE,
        is_unbalance=True, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
    lgb_top.fit(X_train[top_features], y_train)
    lgbm_top_metrics = evaluate_model(lgb_top, X_test[top_features], y_test)
    print(f"  LightGBM Top15 → F2: {lgbm_top_metrics['f2']:.4f} | AUC: {lgbm_top_metrics['auc']:.4f}")

    best_lgb = 'LightGBM_Full' if lgbm_full_metrics['f2'] >= lgbm_top_metrics['f2'] else 'LightGBM_Top'
    best_lgb_features = 'Full Features' if best_lgb == 'LightGBM_Full' else 'Top 15 Features'

    # ── XGBoost ───────────────────────────────────────────
    print("\n--- XGBoost ---")
    xgb_full = XGBClassifier(
        n_estimators=300, random_state=RANDOM_STATE, eval_metric='auc',
        scale_pos_weight=scale_pos, tree_method='hist', verbosity=0, n_jobs=-1)
    xgb_full.fit(X_train, y_train)
    xgb_full_metrics = evaluate_model(xgb_full, X_test, y_test)
    print(f"  XGBoost Full → F2: {xgb_full_metrics['f2']:.4f} | AUC: {xgb_full_metrics['auc']:.4f}")

    # Top 15 features from XGBoost (sampled for speed)
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train, train_size=0.2, stratify=y_train, random_state=RANDOM_STATE)
    xgb_sample = XGBClassifier(
        n_estimators=100, random_state=RANDOM_STATE, eval_metric='auc',
        scale_pos_weight=scale_pos, tree_method='hist', verbosity=0, n_jobs=-1)
    xgb_sample.fit(X_sample, y_sample)
    top_features_xgb = pd.Series(
        xgb_sample.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False).head(15).index.tolist()

    xgb_top = XGBClassifier(
        n_estimators=300, random_state=RANDOM_STATE, eval_metric='auc',
        scale_pos_weight=scale_pos, tree_method='hist', verbosity=0, n_jobs=-1)
    xgb_top.fit(X_train[top_features_xgb], y_train)
    xgb_top_metrics = evaluate_model(xgb_top, X_test[top_features_xgb], y_test)
    print(f"  XGBoost Top15 → F2: {xgb_top_metrics['f2']:.4f} | AUC: {xgb_top_metrics['auc']:.4f}")

    best_xgb = 'XGBoost_Full' if xgb_full_metrics['f2'] >= xgb_top_metrics['f2'] else 'XGBoost_Top'
    best_xgb_features = 'Full Features' if best_xgb == 'XGBoost_Full' else 'Top 15 Features'

    # ── Logistic Regression ───────────────────────────────
    print("\n--- Logistic Regression ---")
    lr_full = make_pipeline(
        SimpleImputer(strategy='median'), StandardScaler(),
        LogisticRegression(class_weight='balanced', max_iter=1000,
                            random_state=RANDOM_STATE, n_jobs=-1, solver='lbfgs'))
    lr_full.fit(X_train, y_train)
    lr_full_metrics = evaluate_model(lr_full, X_test, y_test)
    print(f"  LR Full → F2: {lr_full_metrics['f2']:.4f} | AUC: {lr_full_metrics['auc']:.4f}")

    lr_top = make_pipeline(
        SimpleImputer(strategy='median'), StandardScaler(),
        LogisticRegression(class_weight='balanced', max_iter=1000,
                            random_state=RANDOM_STATE, n_jobs=-1, solver='lbfgs'))
    lr_top.fit(X_train[top_features_xgb], y_train)
    lr_top_metrics = evaluate_model(lr_top, X_test[top_features_xgb], y_test)
    print(f"  LR Top15 → F2: {lr_top_metrics['f2']:.4f} | AUC: {lr_top_metrics['auc']:.4f}")

    best_lr = 'LR_Full' if lr_full_metrics['f2'] >= lr_top_metrics['f2'] else 'LR_Top'
    best_lr_features = 'Full Features' if best_lr == 'LR_Full' else 'Top 15 Features'

    # ── Comparison Table ──────────────────────────────────
    print("\n" + "=" * 70)
    print("=== MODEL COMPARISON ===")
    print("=" * 70)
    comparison = pd.DataFrame([
        {'Model': 'LightGBM', 'Version': best_lgb_features,
         'F2': lgbm_full_metrics['f2'] if best_lgb == 'LightGBM_Full' else lgbm_top_metrics['f2'],
         'AUC': lgbm_full_metrics['auc'] if best_lgb == 'LightGBM_Full' else lgbm_top_metrics['auc'],
         'Recall': lgbm_full_metrics['recall'] if best_lgb == 'LightGBM_Full' else lgbm_top_metrics['recall']},
        {'Model': 'XGBoost', 'Version': best_xgb_features,
         'F2': xgb_full_metrics['f2'] if best_xgb == 'XGBoost_Full' else xgb_top_metrics['f2'],
         'AUC': xgb_full_metrics['auc'] if best_xgb == 'XGBoost_Full' else xgb_top_metrics['auc'],
         'Recall': xgb_full_metrics['recall'] if best_xgb == 'XGBoost_Full' else xgb_top_metrics['recall']},
        {'Model': 'Logistic Regression', 'Version': best_lr_features,
         'F2': lr_full_metrics['f2'] if best_lr == 'LR_Full' else lr_top_metrics['f2'],
         'AUC': lr_full_metrics['auc'] if best_lr == 'LR_Full' else lr_top_metrics['auc'],
         'Recall': lr_full_metrics['recall'] if best_lr == 'LR_Full' else lr_top_metrics['recall']},
    ]).sort_values('F2', ascending=False).reset_index(drop=True)
    print(comparison.to_string(index=False))

    best_row = comparison.iloc[0]
    print(f"\n🏆 Best Model: {best_row['Model']} ({best_row['Version']}) — F2={best_row['F2']:.4f}")
    return best_row['Model'], best_row['Version']


# ============================================================
# STEP 5 — FINE-TUNE LightGBM (Optuna)
# ============================================================
def fine_tune(X_train, y_train, n_trials=N_TRIALS):
    """
    Fine-tune LightGBM hyperparameters using Optuna.
    Uses 30% of training data for speed, with early stopping.
    """
    print("\n" + "=" * 60)
    print(f"STEP 5 — FINE-TUNE LightGBM ({n_trials} trials)")
    print("=" * 60)

    # Sample 30% of train data for tuning (faster)
    X_tune, _, y_tune, _ = train_test_split(
        X_train, y_train, train_size=0.3, stratify=y_train, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tune, y_tune, test_size=0.2, stratify=y_tune, random_state=RANDOM_STATE)

    def objective(trial):
        param = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'boosting_type': 'gbdt', 'random_state': RANDOM_STATE, 'n_jobs': -1,
            'n_estimators'      : 300,
            'learning_rate'     : trial.suggest_float('learning_rate',    0.005, 0.05),
            'scale_pos_weight'  : trial.suggest_float('scale_pos_weight', 1.0,   5.0),
            'num_leaves'        : trial.suggest_int('num_leaves',          15,    63),
            'max_depth'         : trial.suggest_int('max_depth',           3,     12),
            'min_child_samples' : trial.suggest_int('min_child_samples',   200,   2000),
            'feature_fraction'  : trial.suggest_float('feature_fraction', 0.4,   0.8),
            'lambda_l1'         : trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
            'lambda_l2'         : trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        }
        model = LGBMClassifier(**param)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[early_stopping(50), log_evaluation(0)])
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=5, seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    print(f"✅ Best AUC (validation): {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")

    # Re-train on full X_train with best params
    final_model = LGBMClassifier(**study.best_params, n_estimators=400, random_state=RANDOM_STATE)
    final_model.fit(X_train, y_train)
    print("✅ Final model trained on full X_train")
    return final_model, X_val, y_val


# ============================================================
# STEP 6 — EVALUATE FINAL MODEL
# ============================================================
def evaluate_final(final_model, X_train, X_test, y_train, y_test):
    """
    Evaluate final model: AUC, KS, Gini, F2, overfit check.
    """
    print("\n" + "=" * 60)
    print("STEP 6 — EVALUATE FINAL MODEL")
    print("=" * 60)

    y_prob = final_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    ks_stat = max(tpr - fpr)
    opt_thr = thresholds[np.argmax(tpr - fpr)]
    y_pred  = (y_prob >= opt_thr).astype(int)

    auc  = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1

    print(f"AUC-ROC      : {auc:.4f}")
    print(f"KS Statistic : {ks_stat:.4f}")
    print(f"Gini         : {gini:.4f}")
    print(f"Threshold(KS): {opt_thr:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Overfit check
    auc_train = roc_auc_score(y_train, final_model.predict_proba(X_train)[:, 1])
    gap = auc_train - auc
    print(f"Train AUC={auc_train:.4f} | Test AUC={auc:.4f} | Gap={gap:.4f} "
          f"{'⚠️  Possible overfit' if gap > 0.05 else '✅ OK'}")

    return y_prob, opt_thr


# ============================================================
# STEP 7 — CALIBRATION (Platt Scaling)
# ============================================================
def calibrate(final_model, X_val_inner, y_val_inner, X_test, y_test):
    """
    Calibrate model probabilities using Platt Scaling.
    Compares before/after calibration curves and Brier Score.
    """
    print("\n" + "=" * 60)
    print("STEP 7 — CALIBRATION (Platt Scaling)")
    print("=" * 60)

    # Raw PD scores (before calibration)
    pd_raw = final_model.predict_proba(X_test)[:, 1]
    print(f"Before: PD mean={pd_raw.mean():.4f} | min={pd_raw.min():.4f} | max={pd_raw.max():.4f}")

    # Fit calibrator on validation set
    cal_model = PlattCalibrator(final_model)
    cal_model.fit(X_val_inner, y_val_inner)
    pd_calibrated = cal_model.predict_proba(X_test)[:, 1]

    # Brier Score comparison
    brier_raw = brier_score_loss(y_test, pd_raw)
    brier_cal = brier_score_loss(y_test, pd_calibrated)
    print(f"After : PD mean={pd_calibrated.mean():.4f} | min={pd_calibrated.min():.4f} | max={pd_calibrated.max():.4f}")
    print(f"Brier Score — Before: {brier_raw:.4f} | After: {brier_cal:.4f} "
          f"{'✅ Improved' if brier_cal < brier_raw else '⚠️ Check calibration'}")

    # Calibration curve comparison plot
    frac_old, pred_old = calibration_curve(y_test, pd_raw,       n_bins=10)
    frac_new, pred_new = calibration_curve(y_test, pd_calibrated, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0,1],[0,1],'k--', label='Perfectly Calibrated')
    ax.plot(pred_old, frac_old, 's-', color='gray',  alpha=0.6, label='Before Calibration')
    ax.plot(pred_new, frac_new, 'o-', color='green', lw=2,      label='After Calibration (Platt)')
    ax.set_xlabel('Mean Predicted Probability (PD)')
    ax.set_ylabel('Actual Default Rate')
    ax.set_title('Calibration Curve — Before vs After Platt Scaling')
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('calibration_curve.png', dpi=150)
    plt.show()

    print("✅ Using calibrated PD scores going forward")
    return cal_model, pd_calibrated


# ============================================================
# STEP 8A — KB1: Optimal Interest Rate (single customer)
# ============================================================
def run_kb1(cal_model, X_test, customer_idx=0):
    """
    Scenario 1: Find the optimal interest rate for a single customer.
    Sweeps a rate grid, computes E[Profit] via Monte Carlo at each rate.
    The endogeneity effect (higher rate → higher PD) is captured by re-predicting PD.
    """
    print("\n" + "=" * 60)
    print("STEP 8A — KB1: OPTIMAL INTEREST RATE (single customer)")
    print("=" * 60)

    customer = X_test.iloc[[customer_idx]].copy()
    loan_amnt   = customer['loan_amnt'].values[0]
    term_months = int(customer['term_months'].values[0])
    orig_rate   = customer['int_rate'].values[0]
    pd_orig     = cal_model.predict_proba(customer)[0, 1]

    print(f"Customer #{customer_idx}: Loan=${loan_amnt:,.0f} | Term={term_months}mo | "
          f"Original Rate={orig_rate:.1f}% | PD={pd_orig:.1%}")

    rows = []
    for rate in np.arange(6.0, 25.5, 0.5):
        cust_mod = customer.copy()
        cust_mod['int_rate'] = rate
        PD = np.clip(cal_model.predict_proba(cust_mod)[0, 1], 0.001, 0.999)
        mc = monte_carlo_profit(PD, loan_amnt, rate, term_months, N=10_000)
        rows.append({'int_rate': rate, 'PD': PD, **mc})

    df_kb1   = pd.DataFrame(rows)
    opt_idx  = df_kb1['mean_profit'].idxmax()
    opt_rate = df_kb1.loc[opt_idx, 'int_rate']
    opt_prof = df_kb1.loc[opt_idx, 'mean_profit']
    opt_pd   = df_kb1.loc[opt_idx, 'PD']

    print("-" * 40)
    print(f"🎯 Optimal Rate   : {opt_rate:.1f}%")
    print(f"   PD at optimal  : {opt_pd:.1%}")
    print(f"   E[Profit]      : ${opt_prof:,.0f}")
    print(f"   VaR 95%        : ${df_kb1.loc[opt_idx,'var_95']:,.0f}")
    print(f"   P(profit > 0)  : {df_kb1.loc[opt_idx,'prob_profit']:.1%}")
    return df_kb1


# ============================================================
# STEP 8B — KB2: Pricing Policy by Sub_Grade
# ============================================================
def run_kb2(cal_model, X_test, df_original):
    """
    Scenario 2: Determine optimal interest rate for each sub_grade group.
    Uses median customer profile per sub_grade and analytical profit formula.
    """
    print("\n" + "=" * 60)
    print("STEP 8B — KB2: PRICING POLICY BY SUB_GRADE")
    print("=" * 60)

    if 'sub_grade' not in df_original.columns:
        print("⚠️  No sub_grade column found, skipping KB2")
        return pd.DataFrame()

    # Get sub_grade labels for test set (index-aligned)
    sg_series = df_original.loc[X_test.index, 'sub_grade']
    valid_sg  = sorted(sg_series.value_counts()[sg_series.value_counts() >= 20].index)

    # Rate bounds per sub_grade from observed data
    rate_bounds = df_original.groupby('sub_grade')['int_rate'].agg(
        min_rate='min', max_rate='max').to_dict('index')

    print(f"{'Sub':>5} {'Rate Grid':>22} {'Opt Rate':>10} {'PD':>8} {'E[Profit]':>12}")
    print("-" * 65)
    rows = []

    for sg in valid_sg:
        rep  = X_test.loc[sg_series == sg].median().to_frame().T
        la   = rep['loan_amnt'].values[0]
        tm   = int(rep['term_months'].values[0])
        lb   = max(5.0,  rate_bounds.get(sg, {}).get('min_rate', 5.0) - 1.0)
        ub   = min(35.0, rate_bounds.get(sg, {}).get('max_rate', 30.0) + 5.0)
        best = {'rate': None, 'profit': -np.inf, 'PD': None}

        for rate in np.arange(lb, ub + 0.5, 0.5):
            rep_mod = rep.copy(); rep_mod['int_rate'] = rate
            PD  = np.clip(cal_model.predict_proba(rep_mod)[0, 1], 0.001, 0.999)
            # Analytical profit (faster than Monte Carlo for grid search)
            prf = (1 - PD) * calculate_total_income(la, rate, tm) - PD * la * LGD_BASE
            if prf > best['profit']:
                best = {'rate': rate, 'profit': prf, 'PD': PD}

        rows.append({'sub_grade': sg, 'grade': sg[0],
                     'optimal_rate': best['rate'],
                     'PD_at_optimal': best['PD'],
                     'E_profit': best['profit']})
        print(f"  {sg:>5}  [{lb:.0f}%–{ub:.0f}%]"
              f"  {best['rate']:>8.1f}%  {best['PD']:>6.1%}  ${best['profit']:>10,.0f}")

    df_kb2 = pd.DataFrame(rows)

    # Summary by grade
    summary = df_kb2.groupby('grade').agg(
        Rate_Min=('optimal_rate','min'), Rate_Max=('optimal_rate','max'),
        Avg_PD=('PD_at_optimal','mean'), Avg_Profit=('E_profit','mean')
    ).reset_index()
    print("\n" + "=" * 70)
    print(f"{'GRADE':^7} | {'RATE RANGE':^20} | {'AVG PD':^10} | {'AVG PROFIT':^12} | ACTION")
    print("-" * 70)
    for _, row in summary.iterrows():
        rng    = f"{row['Rate_Min']:.1f}%" if row['Rate_Min']==row['Rate_Max'] else f"{row['Rate_Min']:.1f}%–{row['Rate_Max']:.1f}%"
        action = "✅ APPROVE" if row['Avg_Profit'] > 0 else "❌ REJECT"
        print(f"  {row['grade']:^5} | {rng:^20} | {row['Avg_PD']:>8.1%} | ${row['Avg_Profit']:>10,.0f} | {action}")

    print(f"\n✅ KB2 complete: {len(df_kb2)} sub_grades processed")
    return df_kb2


# ============================================================
# STEP 8C — KB3: Portfolio Approval Policy
# ============================================================
def run_kb3(X_test, pd_scores):
    """
    Scenario 3: Find optimal PD threshold to maximize Profit/Loan ratio.
    Uses profit_ratio (not total profit) to avoid scale bias.
    """
    print("\n" + "=" * 60)
    print("STEP 8C — KB3: PORTFOLIO APPROVAL POLICY")
    print("=" * 60)

    result_df = X_test.copy()
    result_df['PD'] = pd_scores

    # Compute E[Profit] analytically for each loan
    PD_vals = result_df['PD'].values
    incomes = np.array([
        calculate_total_income(r['loan_amnt'], r['int_rate'], int(r['term_months']))
        for _, r in result_df.iterrows()
    ])
    result_df['E_profit'] = (1 - PD_vals) * incomes - PD_vals * result_df['loan_amnt'].values * LGD_BASE

    # Sweep thresholds
    rows = []
    for thr in np.arange(0.05, 0.95, 0.01):
        approved = result_df[result_df['PD'] <= thr]
        if len(approved) == 0: continue
        rows.append({
            'threshold'    : thr,
            'n_approved'   : len(approved),
            'approval_rate': len(approved) / len(result_df) * 100,
            'total_profit' : approved['E_profit'].sum(),
            'total_loan'   : approved['loan_amnt'].sum(),
            'avg_PD'       : approved['PD'].mean(),
        })

    df_kb3 = pd.DataFrame(rows)
    # Maximize Profit/Loan ratio (economically sound — avoids scale bias)
    df_kb3['profit_ratio'] = df_kb3['total_profit'] / df_kb3['total_loan']
    opt = df_kb3.loc[df_kb3['profit_ratio'].idxmax()]

    print(f"✅ Optimal Threshold  : PD ≤ {opt['threshold']:.2f}")
    print(f"   Approval Rate      : {opt['approval_rate']:.1f}%")
    print(f"   Profit/Loan Ratio  : {opt['profit_ratio']*100:.2f}%")
    print(f"   Total E[Profit]    : ${opt['total_profit']:,.0f}")
    print(f"   Loans Approved     : {int(opt['n_approved']):,}")
    return df_kb3, result_df


# ============================================================
# STEP 8D — KB4: Capital Allocation by Risk/Return
# ============================================================
def run_kb4(result_df, df_original, X_test):
    """
    Scenario 4: Allocate capital across sub_grades to maximize E[Profit]
    subject to avg PD ≤ 35% constraint.
    Uses greedy allocation weighted by profit_ratio.
    """
    print("\n" + "=" * 60)
    print("STEP 8D — KB4: CAPITAL ALLOCATION BY RISK/RETURN")
    print("=" * 60)

    MAX_PD = 0.35
    BUDGET = result_df['loan_amnt'].sum()

    result_df = result_df.copy()
    if 'sub_grade' in df_original.columns:
        result_df['sub_grade'] = df_original.loc[X_test.index, 'sub_grade'].values
        result_df['grade']     = result_df['sub_grade'].str[0]

    # Risk/Return profile per sub_grade
    profile = result_df.groupby('sub_grade').agg(
        n_loans     =('loan_amnt','count'),
        avg_loan    =('loan_amnt','mean'),
        avg_PD      =('PD','mean'),
        avg_profit  =('E_profit','mean'),
        total_profit=('E_profit','sum'),
        total_loan  =('loan_amnt','sum'),
    ).reset_index()
    profile['grade']        = profile['sub_grade'].str[0]
    profile['profit_ratio'] = profile['avg_profit'] / profile['avg_loan']

    # Filter eligible sub_grades
    eligible = profile[
        (profile['avg_PD'] <= MAX_PD) &
        (profile['avg_profit'] > 0) &
        (profile['n_loans'] >= 20)
    ].copy().sort_values('profit_ratio', ascending=False)

    # Greedy allocation proportional to profit_ratio
    eligible['weight']               = eligible['profit_ratio'] / eligible['profit_ratio'].sum()
    eligible['allocated_budget']     = eligible['weight'] * BUDGET
    eligible['n_loans_target']       = (eligible['allocated_budget'] / eligible['avg_loan']).astype(int)
    eligible['expected_total_profit']= eligible['n_loans_target'] * eligible['avg_profit']

    print(f"Total budget           : ${BUDGET:,.0f}")
    print(f"Max PD allowed         : {MAX_PD:.0%}")
    print(f"Eligible sub_grades    : {len(eligible)}")
    print(f"\nOptimal Allocation:")
    print(eligible[['sub_grade','avg_PD','profit_ratio','weight',
                     'allocated_budget','n_loans_target','expected_total_profit']].to_string(index=False))
    print(f"\n✅ Total Expected Profit: ${eligible['expected_total_profit'].sum():,.0f}")
    return eligible


# ============================================================
# STEP 9 — SAVE ARTIFACTS
# ============================================================
def save_artifacts(final_model, cal_model, X_train,
                   df_kb2, df_kb3, eligible, result_df, subgrade_profile=None):
    """Save all models and data artifacts for Streamlit app."""
    print("\n" + "=" * 60)
    print("STEP 9 — SAVE ARTIFACTS")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Models
    joblib.dump(final_model,              f'{MODEL_DIR}/final_model.pkl')
    joblib.dump(cal_model,                f'{MODEL_DIR}/cal_model.pkl')
    joblib.dump(X_train.columns.tolist(), f'{MODEL_DIR}/selected_features.pkl')

    # Data artifacts
    result_df.to_csv(f'{DATA_DIR}/result_with_pd.csv',     index=False)
    df_kb2.to_csv(   f'{DATA_DIR}/kb2_pricing.csv',         index=False)
    df_kb3.to_csv(   f'{DATA_DIR}/kb3_optimization.csv',    index=False)
    eligible.to_csv( f'{DATA_DIR}/kb4_allocation.csv',      index=False)
    if subgrade_profile is not None:
        subgrade_profile.to_csv(f'{DATA_DIR}/subgrade_profile.csv', index=False)

    print("✅ Saved:")
    for f in sorted(os.listdir(MODEL_DIR)): print(f"   models/{f}")
    for f in sorted(os.listdir(DATA_DIR)):  print(f"   data/{f}")


# ============================================================
# MAIN — Run full pipeline
# ============================================================
def run_pipeline(data_path=CLEANED_DATA_PATH, n_trials=N_TRIALS):
    print("=" * 60)
    print("  CREDIT RISK PIPELINE — Lending Club")
    print("=" * 60)

    # Steps 1–3: Data preparation
    df      = load_data(data_path)
    df_fe   = feature_engineering(df)
    X_train, X_test, y_train, y_test, df_orig = encode_and_split(df_fe)

    # Step 4: Baseline comparison
    run_baseline(X_train, X_test, y_train, y_test)

    # Step 5: Fine-tune LightGBM
    final_model, X_val, y_val = fine_tune(X_train, y_train, n_trials)

    # Step 6: Evaluate
    evaluate_final(final_model, X_train, X_test, y_train, y_test)

    # Step 7: Calibrate
    cal_model, pd_scores = calibrate(final_model, X_val, y_val, X_test, y_test)

    # Step 8: Scenarios
    df_kb1           = run_kb1(cal_model, X_test)
    df_kb2           = run_kb2(cal_model, X_test, df_orig)
    df_kb3, result_df= run_kb3(X_test, pd_scores)
    eligible         = run_kb4(result_df, df_orig, X_test)

    # Step 9: Save
    save_artifacts(final_model, cal_model, X_train,
                   df_kb2, df_kb3, eligible, result_df)

    print("\n🎉 Pipeline complete!")
    return final_model, cal_model, result_df, df_kb2, df_kb3, eligible


if __name__ == "__main__":
    run_pipeline()
