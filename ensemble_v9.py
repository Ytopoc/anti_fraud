"""
ensemble_v9.py - LightGBM + CatBoost + XGBoost Ensemble v9
SKELAR x mono AI Competition

v9 changes over v8:
- features_v9 with cross-user card/holder sharing, card/holder toxicity (LOO),
  tx-level target encoding, specialized low-card features, velocity acceleration
- CatBoost replaces LGB v2 for real model diversity
- early_stopping increased to 300
- Honest v8 vs v9 comparison, ablation, threshold sweep, two-stage decision
"""
import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, brier_score_loss,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import rankdata
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from features_v9 import build_all_features as build_v9_features

try:
    from features_v8 import build_all_features as build_v8_features
    HAS_V8 = True
except ImportError:
    HAS_V8 = False
    build_v8_features = None

DATA = os.path.join(BASE_DIR, "data")
OUT = BASE_DIR
N_FOLDS = 5
SEED = 42


def evaluate_full(y_true, y_prob, threshold, prefix=""):
    """Comprehensive evaluation: F1, PR-AUC, Precision, Recall, Top-k, Brier."""
    y_pred = (y_prob > threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    print(f"\n{prefix}Metrics @ threshold={threshold:.3f}:")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  Brier:     {brier:.4f}")

    n = len(y_true)
    sorted_idx = np.argsort(-y_prob)
    for pct in [1, 3, 5, 10]:
        k = int(n * pct / 100)
        top_k_labels = y_true[sorted_idx[:k]]
        fraud_in_top_k = top_k_labels.sum()
        total_fraud = y_true.sum()
        precision_at_k = fraud_in_top_k / k
        recall_at_k = fraud_in_top_k / total_fraud
        print(f"  Top {pct:2d}%: precision={precision_at_k:.4f}, recall={recall_at_k:.4f} ({int(fraud_in_top_k)}/{int(total_fraud)} frauds caught)")

    print(f"\n{classification_report(y_true, y_pred, target_names=['Legit','Fraud'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

    return {'f1': f1, 'precision': prec, 'recall': rec, 'pr_auc': pr_auc, 'brier': brier}


def find_best_threshold(y_true, y_prob, lo=0.05, hi=0.95, step=0.005):
    """Find threshold that maximizes F1."""
    best_f1, best_t = 0, 0.5
    for t in np.arange(lo, hi, step):
        f = f1_score(y_true, (y_prob > t).astype(int))
        if f > best_f1:
            best_f1 = f
            best_t = t
    return best_t, best_f1


# ================================================================
# v9 ENSEMBLE: LightGBM + CatBoost + XGBoost
# ================================================================
def train_ensemble_v9(X, y, test_X, fnames, skf, sp, label=""):
    """Train 3-model ensemble: LGB v1 + CatBoost + XGB."""

    # ------------------------------------------------------------------
    # Model 1: LightGBM (aggressive)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[1/3] LightGBM (aggressive) {label}")
    print(f"{'=' * 60}")

    lgb_p1 = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 255,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.75,
        'colsample_bytree': 0.6,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'scale_pos_weight': sp,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': SEED,
    }

    oof1 = np.zeros(len(X))
    test1 = np.zeros(len(test_X))
    lgb_models = []

    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        dt = lgb.Dataset(X[ti], y[ti], feature_name=fnames)
        dv = lgb.Dataset(X[vi], y[vi], reference=dt, feature_name=fnames)
        m = lgb.train(lgb_p1, dt, 5000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
        oof1[vi] = m.predict(X[vi])
        test1 += m.predict(test_X.values) / N_FOLDS
        lgb_models.append(m)
        _, bf = find_best_threshold(y[vi], oof1[vi])
        print(f"  Fold {fold}: F1={bf:.4f} iter={m.best_iteration}")

    t1, f1_1 = find_best_threshold(y, oof1)
    print(f"  OOF F1: {f1_1:.4f} @ {t1:.3f}")

    # ------------------------------------------------------------------
    # Model 2: CatBoost (real model diversity)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[2/3] CatBoost {label}")
    print(f"{'=' * 60}")

    cat_params = {
        'iterations': 5000,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 5.0,
        'auto_class_weights': 'Balanced',
        'eval_metric': 'AUC',
        'random_seed': 123,
        'verbose': 0,
        'early_stopping_rounds': 300,
    }

    oof2 = np.zeros(len(X))
    test2 = np.zeros(len(test_X))

    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        pool_train = cb.Pool(X[ti], y[ti])
        pool_val = cb.Pool(X[vi], y[vi])
        model = cb.CatBoostClassifier(**cat_params)
        model.fit(pool_train, eval_set=pool_val, verbose=0)
        oof2[vi] = model.predict_proba(X[vi])[:, 1]
        test2 += model.predict_proba(test_X.values)[:, 1] / N_FOLDS
        _, bf = find_best_threshold(y[vi], oof2[vi])
        print(f"  Fold {fold}: F1={bf:.4f} iter={model.best_iteration_}")

    t2, f1_2 = find_best_threshold(y, oof2)
    print(f"  OOF F1: {f1_2:.4f} @ {t2:.3f}")

    # ------------------------------------------------------------------
    # Model 3: XGBoost (lossguide)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[3/3] XGBoost (lossguide) {label}")
    print(f"{'=' * 60}")

    xgb_p = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.02,
        'max_depth': 0,
        'max_leaves': 255,
        'grow_policy': 'lossguide',
        'min_child_weight': 20,
        'subsample': 0.75,
        'colsample_bytree': 0.6,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'scale_pos_weight': sp,
        'tree_method': 'hist',
        'verbosity': 0,
        'random_state': SEED,
    }

    oof3 = np.zeros(len(X))
    test3 = np.zeros(len(test_X))

    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        dt = xgb.DMatrix(X[ti], y[ti], feature_names=fnames)
        dv = xgb.DMatrix(X[vi], y[vi], feature_names=fnames)
        m = xgb.train(xgb_p, dt, 5000, evals=[(dv, 'v')],
                      early_stopping_rounds=300, verbose_eval=0)
        oof3[vi] = m.predict(dv)
        test3 += m.predict(xgb.DMatrix(test_X.values, feature_names=fnames)) / N_FOLDS
        _, bf = find_best_threshold(y[vi], oof3[vi])
        print(f"  Fold {fold}: F1={bf:.4f} iter={m.best_iteration}")

    t3, f1_3 = find_best_threshold(y, oof3)
    print(f"  OOF F1: {f1_3:.4f} @ {t3:.3f}")

    return oof1, oof2, oof3, test1, test2, test3, lgb_models


# ================================================================
# v8 ENSEMBLE: LightGBM v1 + LightGBM v2 + XGBoost (for comparison)
# ================================================================
def train_ensemble_v8(X, y, test_X, fnames, skf, sp, label=""):
    """Train v8-style 3-model ensemble: LGB v1 + LGB v2 + XGB."""

    # LightGBM v1
    print(f"\n{'=' * 60}")
    print(f"[1/3] LightGBM v1 (aggressive) {label}")
    print(f"{'=' * 60}")

    lgb_p1 = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'learning_rate': 0.02, 'num_leaves': 255, 'max_depth': -1,
        'min_child_samples': 20, 'subsample': 0.75, 'colsample_bytree': 0.6,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'scale_pos_weight': sp,
        'verbose': -1, 'n_jobs': -1, 'random_state': SEED,
    }

    oof1 = np.zeros(len(X))
    test1 = np.zeros(len(test_X))
    lgb_models = []

    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        dt = lgb.Dataset(X[ti], y[ti], feature_name=fnames)
        dv = lgb.Dataset(X[vi], y[vi], reference=dt, feature_name=fnames)
        m = lgb.train(lgb_p1, dt, 5000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        oof1[vi] = m.predict(X[vi])
        test1 += m.predict(test_X.values) / N_FOLDS
        lgb_models.append(m)
        _, bf = find_best_threshold(y[vi], oof1[vi])
        print(f"  Fold {fold}: F1={bf:.4f} iter={m.best_iteration}")

    t1, f1_1 = find_best_threshold(y, oof1)
    print(f"  OOF F1: {f1_1:.4f} @ {t1:.3f}")

    # LightGBM v2
    print(f"\n{'=' * 60}")
    print(f"[2/3] LightGBM v2 (conservative) {label}")
    print(f"{'=' * 60}")

    lgb_p2 = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'learning_rate': 0.03, 'num_leaves': 127, 'max_depth': 8,
        'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.5,
        'reg_alpha': 0.5, 'reg_lambda': 3.0, 'scale_pos_weight': sp,
        'verbose': -1, 'n_jobs': -1, 'random_state': 123,
    }

    oof2 = np.zeros(len(X))
    test2 = np.zeros(len(test_X))

    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        dt = lgb.Dataset(X[ti], y[ti], feature_name=fnames)
        dv = lgb.Dataset(X[vi], y[vi], reference=dt, feature_name=fnames)
        m = lgb.train(lgb_p2, dt, 5000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        oof2[vi] = m.predict(X[vi])
        test2 += m.predict(test_X.values) / N_FOLDS
        _, bf = find_best_threshold(y[vi], oof2[vi])
        print(f"  Fold {fold}: F1={bf:.4f} iter={m.best_iteration}")

    t2, f1_2 = find_best_threshold(y, oof2)
    print(f"  OOF F1: {f1_2:.4f} @ {t2:.3f}")

    # XGBoost
    print(f"\n{'=' * 60}")
    print(f"[3/3] XGBoost (lossguide) {label}")
    print(f"{'=' * 60}")

    xgb_p = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'learning_rate': 0.02, 'max_depth': 0, 'max_leaves': 255,
        'grow_policy': 'lossguide', 'min_child_weight': 20,
        'subsample': 0.75, 'colsample_bytree': 0.6,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'scale_pos_weight': sp,
        'tree_method': 'hist', 'verbosity': 0, 'random_state': SEED,
    }

    oof3 = np.zeros(len(X))
    test3 = np.zeros(len(test_X))

    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        dt = xgb.DMatrix(X[ti], y[ti], feature_names=fnames)
        dv = xgb.DMatrix(X[vi], y[vi], feature_names=fnames)
        m = xgb.train(xgb_p, dt, 5000, evals=[(dv, 'v')],
                      early_stopping_rounds=200, verbose_eval=0)
        oof3[vi] = m.predict(dv)
        test3 += m.predict(xgb.DMatrix(test_X.values, feature_names=fnames)) / N_FOLDS
        _, bf = find_best_threshold(y[vi], oof3[vi])
        print(f"  Fold {fold}: F1={bf:.4f} iter={m.best_iteration}")

    t3, f1_3 = find_best_threshold(y, oof3)
    print(f"  OOF F1: {f1_3:.4f} @ {t3:.3f}")

    return oof1, oof2, oof3, test1, test2, test3, lgb_models


def optimize_ensemble(oof1, oof2, oof3, test1, test2, test3, y, X, skf,
                      m1_name="LGB", m2_name="CB", m3_name="XGB"):
    """Find best blending: weighted avg, rank blending, LR stacking, poly stacking."""

    print(f"\n{'=' * 60}")
    print("ENSEMBLE OPTIMIZATION")
    print(f"{'=' * 60}")

    # --- Weighted average ---
    best_f = 0
    best_w = (0.5, 0.2, 0.3)
    best_t = 0.5

    for w1 in np.arange(0.2, 0.75, 0.05):
        for w2 in np.arange(0.05, 0.55, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.05 or w3 > 0.5:
                continue
            oof_e = w1 * oof1 + w2 * oof2 + w3 * oof3
            t, f = find_best_threshold(y, oof_e)
            if f > best_f:
                best_f = f
                best_w = (w1, w2, w3)
                best_t = t

    print(f"Weighted avg:  {m1_name}={best_w[0]:.2f} {m2_name}={best_w[1]:.2f} {m3_name}={best_w[2]:.2f}")
    print(f"  F1={best_f:.4f} @ thresh={best_t:.3f}")

    # --- Rank-based blending ---
    n = len(y)
    rank1 = rankdata(oof1) / n
    rank2 = rankdata(oof2) / n
    rank3 = rankdata(oof3) / n

    best_rf = 0
    best_rw = (0.5, 0.2, 0.3)
    best_rt = 0.5

    for w1 in np.arange(0.2, 0.75, 0.05):
        for w2 in np.arange(0.05, 0.55, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.05 or w3 > 0.5:
                continue
            oof_r = w1 * rank1 + w2 * rank2 + w3 * rank3
            t, f = find_best_threshold(y, oof_r)
            if f > best_rf:
                best_rf = f
                best_rw = (w1, w2, w3)
                best_rt = t

    print(f"Rank blend:    {m1_name}={best_rw[0]:.2f} {m2_name}={best_rw[1]:.2f} {m3_name}={best_rw[2]:.2f}")
    print(f"  F1={best_rf:.4f} @ thresh={best_rt:.3f}")

    # --- LR stacking ---
    oof_stack = np.column_stack([oof1, oof2, oof3])
    test_stack = np.column_stack([test1, test2, test3])
    lr = LogisticRegression(C=1.0, max_iter=1000)
    oof_lr = np.zeros(len(y))
    test_lr = np.zeros(len(test1))
    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        lr.fit(oof_stack[ti], y[ti])
        oof_lr[vi] = lr.predict_proba(oof_stack[vi])[:, 1]
        test_lr += lr.predict_proba(test_stack)[:, 1] / N_FOLDS

    t_lr, f_lr = find_best_threshold(y, oof_lr)
    print(f"LR stacking:   F1={f_lr:.4f} @ thresh={t_lr:.3f}")

    # --- Polynomial stacking ---
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    oof_poly = poly.fit_transform(oof_stack)
    test_poly = poly.transform(test_stack)
    lr2 = LogisticRegression(C=0.5, max_iter=1000)
    oof_plr = np.zeros(len(y))
    test_plr = np.zeros(len(test1))
    for fold, (ti, vi) in enumerate(skf.split(X, y)):
        lr2.fit(oof_poly[ti], y[ti])
        oof_plr[vi] = lr2.predict_proba(oof_poly[vi])[:, 1]
        test_plr += lr2.predict_proba(test_poly)[:, 1] / N_FOLDS

    t_plr, f_plr = find_best_threshold(y, oof_plr)
    print(f"Poly stacking: F1={f_plr:.4f} @ thresh={t_plr:.3f}")

    # --- Pick winner ---
    methods = {
        'weighted_avg': (best_f, best_t,
                         best_w[0] * oof1 + best_w[1] * oof2 + best_w[2] * oof3,
                         best_w[0] * test1 + best_w[1] * test2 + best_w[2] * test3,
                         best_w),
        'rank_blend': (best_rf, best_rt,
                       best_rw[0] * rank1 + best_rw[1] * rank2 + best_rw[2] * rank3,
                       best_rw[0] * rankdata(test1)/len(test1) + best_rw[1] * rankdata(test2)/len(test1) + best_rw[2] * rankdata(test3)/len(test1),
                       best_rw),
        'lr_stacking': (f_lr, t_lr, oof_lr, test_lr, None),
        'poly_stacking': (f_plr, t_plr, oof_plr, test_plr, None),
    }

    winner = max(methods.items(), key=lambda x: x[1][0])
    print(f"\n=> Winner: {winner[0]} (F1={winner[1][0]:.4f})")

    return winner[0], winner[1][0], winner[1][1], winner[1][2], winner[1][3]


def threshold_sweep(y_true, y_prob, label=""):
    """Print threshold -> precision/recall/f1 table."""
    print(f"\n{'=' * 60}")
    print(f"THRESHOLD SWEEP TABLE {label}")
    print(f"{'=' * 60}")
    print(f"  {'Thresh':>8s}  {'Prec':>8s}  {'Recall':>8s}  {'F1':>8s}  {'#Pred':>8s}  {'TP':>6s}  {'FP':>6s}  {'FN':>6s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")

    for t in np.arange(0.10, 0.95, 0.05):
        yp = (y_prob > t).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        n_pred = int(yp.sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-8)
        print(f"  {t:8.3f}  {p:8.4f}  {r:8.4f}  {f:8.4f}  {n_pred:8d}  {tp:6d}  {fp:6d}  {fn:6d}")


def two_stage_decision(y_true, y_prob):
    """Propose two-stage decision thresholds: auto-fraud / review / pass."""
    print(f"\n{'=' * 60}")
    print("TWO-STAGE DECISION PROPOSAL")
    print(f"{'=' * 60}")

    best_combo = None
    best_combo_f1 = 0

    for t_high in np.arange(0.90, 0.99, 0.01):
        for t_low in np.arange(0.30, t_high - 0.05, 0.05):
            auto_fraud = y_prob > t_high
            review = (y_prob > t_low) & (y_prob <= t_high)
            auto_pass = y_prob <= t_low

            tp_auto = int(((auto_fraud) & (y_true == 1)).sum())
            tp_review = int(((review) & (y_true == 1)).sum())
            fp_auto = int(((auto_fraud) & (y_true == 0)).sum())
            fp_review = int(((review) & (y_true == 0)).sum())
            fn_pass = int(((auto_pass) & (y_true == 1)).sum())

            total_review = int(review.sum())

            eff_tp = tp_auto + 0.7 * tp_review
            eff_fp = fp_auto + 0.3 * fp_review
            eff_fn = fn_pass + 0.3 * tp_review
            eff_prec = eff_tp / max(eff_tp + eff_fp, 1)
            eff_rec = eff_tp / max(eff_tp + eff_fn, 1)
            eff_f1 = 2 * eff_prec * eff_rec / max(eff_prec + eff_rec, 1e-8)

            if eff_f1 > best_combo_f1 and total_review < len(y_true) * 0.1:
                best_combo_f1 = eff_f1
                best_combo = (t_high, t_low, tp_auto, fp_auto, tp_review, fp_review,
                              fn_pass, total_review, eff_prec, eff_rec, eff_f1)

    if best_combo:
        t_high, t_low = best_combo[0], best_combo[1]
        print(f"  Auto-block threshold:  > {t_high:.2f}")
        print(f"  Manual review zone:    {t_low:.2f} - {t_high:.2f}")
        print(f"  Auto-pass threshold:   < {t_low:.2f}")
        print(f"")
        print(f"  Auto-block: {best_combo[2]} TP, {best_combo[3]} FP")
        print(f"  Review:     {best_combo[4]} TP, {best_combo[5]} FP ({best_combo[7]} total to review)")
        print(f"  Missed:     {best_combo[6]} FN (pass through)")
        print(f"")
        print(f"  Effective metrics (assuming 70% review catch rate):")
        print(f"    Precision: {best_combo[8]:.4f}")
        print(f"    Recall:    {best_combo[9]:.4f}")
        print(f"    F1:        {best_combo[10]:.4f}")
        print(f"    Review volume: {best_combo[7]} ({best_combo[7]/len(y_true)*100:.2f}% of users)")


def main():
    # ================================================================
    # 1. LOAD DATA
    # ================================================================
    print("Loading data...")
    train_users = pd.read_csv(os.path.join(DATA, "train_users.csv"))
    test_users = pd.read_csv(os.path.join(DATA, "test_users.csv"))
    train_tx = pd.read_csv(os.path.join(DATA, "train_transactions.csv"))
    test_tx = pd.read_csv(os.path.join(DATA, "test_transactions.csv"))

    for df, col in [(train_users, 'timestamp_reg'), (test_users, 'timestamp_reg'),
                    (train_tx, 'timestamp_tr'), (test_tx, 'timestamp_tr')]:
        df[col] = pd.to_datetime(df[col], format='ISO8601', utc=True)

    # ================================================================
    # 2. BUILD v9 FEATURES
    # ================================================================
    print("\n" + "=" * 60)
    print("BUILDING v9 FEATURES")
    print("=" * 60)
    train_X9, test_X9, y, fnames9 = build_v9_features(
        train_users, test_users, train_tx, test_tx
    )
    X9 = train_X9.values

    sp = (len(y) - y.sum()) / y.sum()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # ================================================================
    # 3. TRAIN v9 ENSEMBLE (LGB + CatBoost + XGB)
    # ================================================================
    oof1_9, oof2_9, oof3_9, test1_9, test2_9, test3_9, lgb_models_9 = train_ensemble_v9(
        X9, y, test_X9, fnames9, skf, sp, label="[v9]"
    )

    method_9, f1_9, thresh_9, final_oof_9, final_test_9 = optimize_ensemble(
        oof1_9, oof2_9, oof3_9, test1_9, test2_9, test3_9, y, X9, skf,
        m1_name="LGB", m2_name="CB", m3_name="XGB"
    )

    # ================================================================
    # 4. BUILD v8 FEATURES & TRAIN v8 ENSEMBLE (optional, for comparison)
    # ================================================================
    if HAS_V8:
        print("\n" + "#" * 60)
        print("# BUILDING v8 FEATURES (for honest comparison)")
        print("#" * 60)
        train_X8, test_X8, y8, fnames8 = build_v8_features(
            train_users, test_users, train_tx, test_tx
        )
        X8 = train_X8.values

        oof1_8, oof2_8, oof3_8, test1_8, test2_8, test3_8, lgb_models_8 = train_ensemble_v8(
            X8, y8, test_X8, fnames8, skf, sp, label="[v8]"
        )

        method_8, f1_8, thresh_8, final_oof_8, final_test_8 = optimize_ensemble(
            oof1_8, oof2_8, oof3_8, test1_8, test2_8, test3_8, y8, X8, skf,
            m1_name="LGB1", m2_name="LGB2", m3_name="XGB"
        )

        # ================================================================
        # 5. HONEST v8 vs v9 COMPARISON
        # ================================================================
        print("\n" + "=" * 60)
        print("HONEST COMPARISON: v8 vs v9 (same CV, same seed)")
        print("=" * 60)

        metrics_8 = evaluate_full(y, final_oof_8, thresh_8, prefix="v8 ")
        metrics_9 = evaluate_full(y, final_oof_9, thresh_9, prefix="v9 ")

        print(f"\n{'Metric':<20s}  {'v8':>10s}  {'v9':>10s}  {'Delta':>10s}  {'Better':>8s}")
        print(f"{'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
        for m in ['f1', 'precision', 'recall', 'pr_auc', 'brier']:
            v8_val = metrics_8[m]
            v9_val = metrics_9[m]
            delta = v9_val - v8_val
            if m == 'brier':
                better = 'v9' if delta < 0 else ('v8' if delta > 0 else 'tie')
            else:
                better = 'v9' if delta > 0 else ('v8' if delta < 0 else 'tie')
            print(f"  {m:<18s}  {v8_val:10.4f}  {v9_val:10.4f}  {delta:+10.4f}  {better:>8s}")

        print(f"\n  v8 features: {len(fnames8)}, v9 features: {len(fnames9)} (+{len(fnames9)-len(fnames8)})")
        print(f"  v8 method: {method_8}, v9 method: {method_9}")
        print(f"  v8 threshold: {thresh_8:.3f}, v9 threshold: {thresh_9:.3f}")
        print(f"  v8 models: LGB1+LGB2+XGB, v9 models: LGB+CatBoost+XGB")

        # Per-model comparison (v9 individual models)
        print(f"\n--- v9 Per-model OOF F1 ---")
        print(f"  {'Model':<20s}  {'v9 F1':>8s}")
        print(f"  {'-'*20}  {'-'*8}")
        for name, oof in [("LGB", oof1_9), ("CatBoost", oof2_9), ("XGB", oof3_9)]:
            _, f = find_best_threshold(y, oof)
            print(f"  {name:<20s}  {f:8.4f}")

        # v8 individual models
        print(f"\n--- v8 Per-model OOF F1 ---")
        print(f"  {'Model':<20s}  {'v8 F1':>8s}")
        print(f"  {'-'*20}  {'-'*8}")
        for name, oof in [("LGB v1", oof1_8), ("LGB v2", oof2_8), ("XGB", oof3_8)]:
            _, f = find_best_threshold(y, oof)
            print(f"  {name:<20s}  {f:8.4f}")

        v9_wins = sum(1 for m in ['f1', 'pr_auc', 'precision', 'recall']
                      if metrics_9[m] > metrics_8[m])
        if metrics_9['brier'] < metrics_8['brier']:
            v9_wins += 1

        if v9_wins >= 3:
            print(f"\n=> VERDICT: v9 WINS ({v9_wins}/5 metrics better)")
            use_v9 = True
        elif v9_wins <= 2:
            print(f"\n=> VERDICT: v9 is NOT convincingly better than v8 ({v9_wins}/5 metrics)")
            if metrics_9['f1'] > metrics_8['f1']:
                print("   However, F1 (primary metric) improved, so using v9 for submission.")
                use_v9 = True
            else:
                print("   F1 did not improve. Using v8 for submission.")
                use_v9 = False
        else:
            print(f"\n=> VERDICT: Mixed results ({v9_wins}/5 metrics)")
            use_v9 = metrics_9['f1'] >= metrics_8['f1']
    else:
        print("\n[features_v8 not found -- skipping v8 comparison, using v9 only]")
        evaluate_full(y, final_oof_9, thresh_9, prefix="v9 Final ")
        print(f"\n--- v9 Per-model OOF F1 ---")
        print(f"  {'Model':<20s}  {'v9 F1':>8s}")
        print(f"  {'-'*20}  {'-'*8}")
        for name, oof in [("LGB", oof1_9), ("CatBoost", oof2_9), ("XGB", oof3_9)]:
            _, f = find_best_threshold(y, oof)
            print(f"  {name:<20s}  {f:8.4f}")
        use_v9 = True
        final_oof_8 = None
        final_test_8 = None
        thresh_8 = None

    # ================================================================
    # 6. ABLATION ANALYSIS (v9 new feature blocks)
    # ================================================================
    print("\n" + "=" * 60)
    print("ABLATION ANALYSIS (v9 new feature blocks)")
    print("=" * 60)

    v9_new_features = {
        'cross_user_card': ['max_card_n_users', 'mean_card_n_users', 'cards_shared_2plus',
                            'cards_shared_3plus', 'pct_shared_cards',
                            'shared_cards_x_fail_switch'],
        'card_toxicity': ['max_card_toxicity', 'mean_card_toxicity', 'sum_card_toxicity',
                          'has_toxic_card', 'card_toxicity_x_cards', 'card_toxicity_x_fail_rate'],
        'holder_toxicity': ['max_holder_toxicity', 'mean_holder_toxicity', 'sum_holder_toxicity',
                            'has_toxic_holder'],
        'cross_user_holder': ['max_holder_n_users', 'mean_holder_n_users'],
        'tx_level_te': [f'txte_{f}_{a}' for f in ['card_country', 'card_brand', 'currency', 'payment_country']
                        for a in ['mean', 'max']],
        'low_card': ['single_card_country_mm', 'single_card_antifraud', 'single_card_fraud_err',
                     'init_only_user', 'low_tx_fail_init'],
        'velocity_accel': ['mean_acceleration', 'max_acceleration'],
        'v9_interactions': ['antifraud_x_country_mm'],
    }

    # For each block, measure v9 LGB F1 without that block
    print(f"\n  {'Block':<25s}  {'Full v9 F1':>12s}  {'Without F1':>12s}  {'Drop':>10s}  {'Contrib':>10s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}")

    _, f1_full_lgb1 = find_best_threshold(y, oof1_9)

    for block_name, block_feats in v9_new_features.items():
        drop_cols = [f for f in block_feats if f in fnames9]
        if not drop_cols:
            print(f"  {block_name:<25s}  (no matching features found)")
            continue

        keep_idx = [i for i, f in enumerate(fnames9) if f not in drop_cols]
        X9_abl = X9[:, keep_idx]
        fnames_abl = [fnames9[i] for i in keep_idx]

        oof_abl = np.zeros(len(X9_abl))
        for fold, (ti, vi) in enumerate(skf.split(X9_abl, y)):
            dt = lgb.Dataset(X9_abl[ti], y[ti], feature_name=fnames_abl)
            dv = lgb.Dataset(X9_abl[vi], y[vi], reference=dt, feature_name=fnames_abl)
            m = lgb.train({
                'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
                'learning_rate': 0.02, 'num_leaves': 255, 'max_depth': -1,
                'min_child_samples': 20, 'subsample': 0.75, 'colsample_bytree': 0.6,
                'reg_alpha': 1.0, 'reg_lambda': 5.0, 'scale_pos_weight': sp,
                'verbose': -1, 'n_jobs': -1, 'random_state': SEED,
            }, dt, 5000, valid_sets=[dv],
                callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
            oof_abl[vi] = m.predict(X9_abl[vi])

        _, f1_abl = find_best_threshold(y, oof_abl)
        drop_val = f1_full_lgb1 - f1_abl
        verdict = 'CRITICAL' if drop_val > 0.005 else 'useful' if drop_val > 0.001 else 'marginal' if drop_val > 0 else 'noise'
        print(f"  {block_name:<25s}  {f1_full_lgb1:12.4f}  {f1_abl:12.4f}  {drop_val:+10.4f}  {verdict}")

    # ================================================================
    # 7. FEATURE IMPORTANCE
    # ================================================================
    print("\n" + "=" * 60)
    print("TOP-30 FEATURES (v9 LightGBM gain)")
    print("=" * 60)

    imp = np.zeros(len(fnames9))
    for m in lgb_models_9:
        imp += m.feature_importance(importance_type='gain')
    imp /= len(lgb_models_9)
    fi = pd.Series(imp, index=fnames9).sort_values(ascending=False)
    for i, (feat, v) in enumerate(fi.head(30).items()):
        print(f"  {i + 1:2d}. {feat:45s} {v:12.1f}")

    # v9 new features importance
    v9_all_new = []
    for feats in v9_new_features.values():
        v9_all_new.extend(feats)
    v9_imp = fi[fi.index.isin(v9_all_new)].sort_values(ascending=False)
    print(f"\n--- v9 NEW features importance ---")
    for i, (feat, v) in enumerate(v9_imp.head(20).items()):
        print(f"  {i + 1:2d}. {feat:45s} {v:12.1f}")

    # ================================================================
    # 8. THRESHOLD SWEEP
    # ================================================================
    final_oof = final_oof_9 if use_v9 else final_oof_8
    final_test = final_test_9 if use_v9 else final_test_8
    final_t = thresh_9 if use_v9 else thresh_8
    version = "v9" if use_v9 else "v8"

    threshold_sweep(y, final_oof, label=f"({version})")

    # ================================================================
    # 9. TWO-STAGE DECISION
    # ================================================================
    two_stage_decision(y, final_oof)

    # ================================================================
    # 10. SUBMISSION
    # ================================================================
    print("\n" + "=" * 60)
    print(f"SUBMISSION (using {version})")
    print("=" * 60)

    test_pred = (final_test > final_t).astype(int)
    print(f"Predicted fraud: {test_pred.sum()}/{len(test_pred)} ({test_pred.mean() * 100:.2f}%)")

    sub = pd.DataFrame({'id_user': test_X9.index if use_v9 else test_X8.index, 'is_fraud': test_pred})
    out_path = os.path.join(OUT, "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(sub.head(10))
    print("\nDone!")


if __name__ == '__main__':
    main()
