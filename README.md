[**English**](README.md) | [Українська](README.uk.md)

# Anti-Fraud User Detection - SKELAR x mono AI

A machine-learning pipeline for the SKELAR x mono AI competition that predicts
whether a user is fraudulent (`is_fraud ∈ {0, 1}`) from registration metadata
and the user's transaction history. The pipeline builds a rich set of
behavioural features per user and trains a 3-model gradient-boosting ensemble
(LightGBM + CatBoost + XGBoost) with out-of-fold blending and threshold tuning
optimised for F1.

## Project structure

```
Anti_fraud/
├── data/
│   ├── train_users.csv          # 395k users with is_fraud label
│   ├── test_users.csv           # 169k users to predict
│   ├── train_transactions.csv   # 3.1M transactions
│   └── test_transactions.csv    # 1.35M transactions
├── features_v9.py               # Feature engineering (~250 features per user)
├── ensemble_v9.py               # Full pipeline + optional v8 vs v9 comparison
├── submission.csv               # Output: id_user, is_fraud
├── sub.csv                      # Previous submission snapshot
├── catboost_info/               # CatBoost training artifacts
└── README.md
```

## Data schema

**Users** (`*_users.csv`):
`id_user, timestamp_reg, email, gender, reg_country, traffic_type, is_fraud`

**Transactions** (`*_transactions.csv`):
`id_user, timestamp_tr, amount, status, transaction_type, error_group,
currency, card_brand, card_type, card_country, card_holder, card_mask_hash,
payment_country`

The fraud rate in train is approximately 4–5%, so the pipeline uses
`scale_pos_weight = N_neg / N_pos` for class balance instead of resampling.

## Requirements

- Python 3.9+
- `pandas`, `numpy`, `scikit-learn`, `scipy`
- `lightgbm`, `xgboost`, `catboost`

```bash
pip install pandas numpy scikit-learn scipy lightgbm xgboost catboost
```

## How to run

The pipeline is self-contained and resolves paths relative to its own
location. From the project root:

```bash
python ensemble_v9.py
```

`ensemble_v9.py` runs the v9 pipeline and, **if `features_v8.py` is present
in the same folder**, additionally trains a v8 ensemble for honest side-by-
side comparison. If `features_v8.py` is missing, the v8 block is skipped
automatically and v9 is used for the final submission.

The output is written to `submission.csv` next to the script (`id_user, is_fraud`).

## Feature engineering (`features_v9.py`)

Approximately 250 features per user, grouped into:

1. **Transaction aggregates** - counts, amount statistics (mean/max/min/std/
   median/skew/IQR), time-since-registration stats, transaction-type counts,
   error-group counts, country mismatch flags (card vs registration vs
   payment), night activity, holder name length.
2. **Status streaks & switches** - max consecutive failures, fail-then-switch-
   card patterns, fails before first success.
3. **Per-card / per-holder dynamics** - card lifespan, fail rate per card,
   cards with all-failed history, holders sharing multiple cards, sessions
   split by 30-min idle gaps.
4. **Early activity windows** - counts/fails/cards/holders observed within
   1 / 6 / 24 / 72 / 168 hours after registration.
5. **Email–holder match** - heuristic similarity between email local part and
   the dominant card-holder name.
6. **Behavioural shift** - first-half vs second-half fail rate, card diversity.
7. **v9 cross-user signals**
   - Card sharing across users (a card used by 2+ users is a strong fraud
     signal).
   - Holder name sharing across users.
   - **Card toxicity** - leave-one-out target encoding of how fraudulent the
     other users sharing a given card are.
   - **Holder toxicity** - same idea on holder names.
8. **v9 transaction-level target encoding** - OOF target encoding for
   `card_country`, `card_brand`, `currency`, `payment_country`, aggregated per
   user with mean and max (5-fold, smoothing = 50).
9. **v9 specialised low-card features** - single-card country mismatch,
   single-card with antifraud/fraud errors, init-only users, low-activity
   users with failed `card_init`.
10. **Velocity acceleration** - first derivative of inter-transaction time
    gaps; fraudsters tend to accelerate as they find working cards.
11. **User-level target encoding** - for `reg_country`, `traffic_type`,
    `email_domain`, `gender × traffic`, `country × traffic`.

All target encodings use 5-fold OOF on the training set and full-train
statistics applied to the test set, so there is no label leakage.

## Model ensemble (`ensemble_v9.py`)

5-fold `StratifiedKFold` (seed 42) with three diverse boosters:

| Model        | Tree shape                       | Learning rate | Notes                                  |
| ------------ | -------------------------------- | ------------- | -------------------------------------- |
| LightGBM     | 255 leaves, depth = -1           | 0.02          | Aggressive, low feature subsampling    |
| CatBoost     | depth 8                          | 0.03          | Native handling of high-cardinality TE |
| XGBoost      | 255 leaves, `lossguide`          | 0.02          | Histogram-based                        |

All three use `scale_pos_weight` for class imbalance and 300-round early
stopping on AUC.

### Blending

Four blending methods are evaluated on OOF predictions; the best F1 wins:

- Weighted average (grid over weights summing to 1.0).
- Rank-based blending (same grid, applied to per-model rank-percentiles).
- Logistic-regression stacking (level-2 LR on raw OOFs).
- Polynomial stacking (degree-2 LR with interactions).

Threshold is tuned by sweeping `0.05 → 0.95` (step 0.005) for max F1 on the
final OOF probabilities.

## What the pipeline reports

In addition to writing the submission, the run prints:

- Per-model and blended OOF F1, precision, recall, PR-AUC, Brier score.
- Top-k precision/recall at 1 / 3 / 5 / 10 % of users.
- Confusion matrix and full classification report at the chosen threshold.
- Threshold sweep table from 0.10 to 0.95.
- **Ablation analysis** - drop in F1 when each v9 feature block is removed
  (cross-user card, card toxicity, holder toxicity, cross-user holder, tx-
  level TE, low-card features, velocity acceleration, v9 interactions).
- **Top-30 feature importance** by LightGBM gain.
- **Two-stage decision proposal** - auto-block / manual review / auto-pass
  thresholds optimised under a fixed review-volume budget (≤ 10 % of users).

## Output format

`submission.csv`:

```
id_user,is_fraud
16318030,0
...
```

One row per `id_user` in `test_users.csv` (169 449 rows + header).

## Version history

- **v9** (current) - added cross-user card/holder sharing, LOO card/holder
  toxicity, transaction-level target encoding, low-card user signals, velocity
  acceleration. CatBoost replaces the second LightGBM for genuine model
  diversity. Early stopping bumped from 200 to 300 rounds.
- **v8** (reference, optional) - base feature set without cross-user signals;
  triggered only if `features_v8.py` is present alongside `ensemble_v9.py`.
