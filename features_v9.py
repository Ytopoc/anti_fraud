"""
features_v9.py - Anti-Fraud Feature Engineering v9
SKELAR x mono AI Competition

v9 changes over v8:
- ADDED: Cross-user card sharing (computed on combined train+test tx)
- ADDED: Card toxicity (LOO target encoding, no leakage)
- ADDED: Holder toxicity (LOO target encoding, no leakage)
- ADDED: Cross-user holder sharing
- ADDED: Transaction-level target encoding (card_country, card_brand, currency, payment_country)
- ADDED: Specialized low-card user features (single-card, init-only)
- ADDED: Velocity acceleration features
- ADDED: Enhanced interactions with new features
- REMOVED: Error per-card diversity (noise, ablation = -0.0005)
- REMOVED: Log transforms (marginal, 8 features for +0.0002)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


# ================================================================
# TARGET ENCODING (OOF-safe)
# ================================================================
def target_encode_cv(train_s, test_s, target, smoothing=50, n_splits=5):
    gm = target.mean()
    tr_enc = pd.Series(np.nan, index=train_s.index)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tri, vai in skf.split(train_s, target):
        fold = pd.DataFrame({'col': train_s.iloc[tri], 'y': target.iloc[tri]})
        st = fold.groupby('col')['y'].agg(['mean', 'count'])
        sm = (st['count'] * st['mean'] + smoothing * gm) / (st['count'] + smoothing)
        tr_enc.iloc[vai] = train_s.iloc[vai].map(sm).fillna(gm).values
    full = pd.DataFrame({'col': train_s, 'y': target})
    st = full.groupby('col')['y'].agg(['mean', 'count'])
    sm = (st['count'] * st['mean'] + smoothing * gm) / (st['count'] + smoothing)
    te_enc = test_s.map(sm).fillna(gm)
    return tr_enc, te_enc


# ================================================================
# TRANSACTION FEATURES (per-user, from their own transactions)
# v8 base minus error_diversity and log_transforms,
# plus velocity acceleration and low-card user features
# ================================================================
def build_tx_features(tx_df, users_df):
    tx = tx_df.merge(
        users_df[['id_user', 'timestamp_reg', 'reg_country']],
        on='id_user', how='left'
    )

    tx['h_since_reg'] = (tx['timestamp_tr'] - tx['timestamp_reg']).dt.total_seconds() / 3600
    tx['tx_hour'] = tx['timestamp_tr'].dt.hour
    tx['tx_dow'] = tx['timestamp_tr'].dt.dayofweek
    tx['tx_date'] = tx['timestamp_tr'].dt.date

    tx['mm_card_pay'] = (tx['card_country'] != tx['payment_country']).astype(int)
    tx['mm_card_reg'] = (tx['card_country'] != tx['reg_country']).astype(int)
    tx['mm_pay_reg'] = (tx['payment_country'] != tx['reg_country']).astype(int)
    tx['mm_total'] = tx['mm_card_pay'] + tx['mm_card_reg'] + tx['mm_pay_reg']

    tx = tx.sort_values(['id_user', 'timestamp_tr'])
    tx['tdiff'] = tx.groupby('id_user')['timestamp_tr'].diff().dt.total_seconds()
    tx['is_fail'] = (tx['status'] == 'fail').astype(int)
    tx['is_night'] = tx['tx_hour'].between(0, 5).astype(int)
    tx['holder_len'] = tx['card_holder'].str.len().fillna(0)
    tx['holder_words'] = tx['card_holder'].str.split().str.len().fillna(0)

    g = tx.groupby('id_user')
    F = {}

    # ================================================================
    # 1-11: Core counts, amounts, time, types, errors, uniques,
    #        country mismatch, hour/dow, rapid retries, daily patterns,
    #        holder name
    # ================================================================
    F['tx_count'] = g.size()
    F['tx_success'] = tx[tx['status'] == 'success'].groupby('id_user').size()
    F['tx_fail'] = tx[tx['status'] == 'fail'].groupby('id_user').size()

    F['amt_mean'] = g['amount'].mean()
    F['amt_max'] = g['amount'].max()
    F['amt_min'] = g['amount'].min()
    F['amt_sum'] = g['amount'].sum()
    F['amt_std'] = g['amount'].std()
    F['amt_median'] = g['amount'].median()
    F['amt_range'] = F['amt_max'] - F['amt_min']
    F['amt_skew'] = g['amount'].skew()
    F['amt_iqr'] = g['amount'].quantile(0.75) - g['amount'].quantile(0.25)
    F['succ_amt_sum'] = tx[tx['status'] == 'success'].groupby('id_user')['amount'].sum()
    F['succ_amt_mean'] = tx[tx['status'] == 'success'].groupby('id_user')['amount'].mean()
    F['fail_amt_sum'] = tx[tx['status'] == 'fail'].groupby('id_user')['amount'].sum()
    F['fail_amt_mean'] = tx[tx['status'] == 'fail'].groupby('id_user')['amount'].mean()
    F['unique_amounts'] = g['amount'].nunique()

    F['first_tx_h'] = g['h_since_reg'].min()
    F['last_tx_h'] = g['h_since_reg'].max()
    F['h_reg_mean'] = g['h_since_reg'].mean()
    F['h_reg_std'] = g['h_since_reg'].std()
    F['tx_span_h'] = (g['timestamp_tr'].max() - g['timestamp_tr'].min()).dt.total_seconds() / 3600
    F['tdiff_min'] = g['tdiff'].min()
    F['tdiff_mean'] = g['tdiff'].mean()
    F['tdiff_std'] = g['tdiff'].std()
    F['tdiff_median'] = g['tdiff'].median()
    F['tdiff_q10'] = g['tdiff'].quantile(0.1)

    for tt in ['card_init', 'card_recurring', 'google-pay', 'apple-pay', 'resign']:
        F[f'tt_{tt.replace("-", "_")}'] = tx[tx['transaction_type'] == tt].groupby('id_user').size()
    F['init_fail'] = tx[(tx['transaction_type'] == 'card_init') & (tx['status'] == 'fail')].groupby('id_user').size()
    F['init_succ'] = tx[(tx['transaction_type'] == 'card_init') & (tx['status'] == 'success')].groupby('id_user').size()
    F['recur_fail'] = tx[(tx['transaction_type'] == 'card_recurring') & (tx['status'] == 'fail')].groupby('id_user').size()
    F['recur_succ'] = tx[(tx['transaction_type'] == 'card_recurring') & (tx['status'] == 'success')].groupby('id_user').size()

    tx_err = tx[tx['error_group'].notna() & (tx['error_group'] != '')]
    F['err_count'] = tx_err.groupby('id_user').size()
    F['err_unique'] = tx_err.groupby('id_user')['error_group'].nunique()
    for eg in ['fraud', 'antifraud', '3ds error', 'insufficient funds error',
               'do not honor', 'card problem', 'cvv error', 'issuer decline',
               'invalid data', 'expired error']:
        F[f'err_{eg.replace(" ", "_")}'] = tx_err[tx_err['error_group'] == eg].groupby('id_user').size()
    F['fraud_err_cards'] = tx_err[tx_err['error_group'] == 'fraud'].groupby('id_user')['card_mask_hash'].nunique()
    F['antifraud_err_cards'] = tx_err[tx_err['error_group'] == 'antifraud'].groupby('id_user')['card_mask_hash'].nunique()

    F['u_cards'] = g['card_mask_hash'].nunique()
    F['u_brands'] = g['card_brand'].nunique()
    F['u_card_ctry'] = g['card_country'].nunique()
    F['u_curr'] = g['currency'].nunique()
    F['u_pay_ctry'] = g['payment_country'].nunique()
    F['u_holders'] = g['card_holder'].nunique()
    F['u_tx_types'] = g['transaction_type'].nunique()
    for ct in ['DEBIT', 'CREDIT', 'PREPAID']:
        F[f'ct_{ct}'] = tx[tx['card_type'] == ct].groupby('id_user').size()

    for mc in ['mm_card_pay', 'mm_card_reg', 'mm_pay_reg', 'mm_total']:
        F[f'{mc}_cnt'] = g[mc].sum()
        F[f'{mc}_rate'] = g[mc].mean()

    F['hour_mean'] = g['tx_hour'].mean()
    F['hour_std'] = g['tx_hour'].std()
    F['night_cnt'] = g['is_night'].sum()
    F['night_rate'] = g['is_night'].mean()

    tx_f = tx[tx['status'] == 'fail'].copy()
    tx_f['ftd'] = tx_f.groupby('id_user')['timestamp_tr'].diff().dt.total_seconds()
    F['rapid_fail_10s'] = tx_f[tx_f['ftd'] < 10].groupby('id_user').size()
    F['rapid_fail_30s'] = tx_f[tx_f['ftd'] < 30].groupby('id_user').size()
    F['rapid_fail_60s'] = tx_f[tx_f['ftd'] < 60].groupby('id_user').size()
    F['rapid_fail_300s'] = tx_f[tx_f['ftd'] < 300].groupby('id_user').size()
    tx_i = tx[tx['transaction_type'] == 'card_init'].copy()
    tx_i['itd'] = tx_i.groupby('id_user')['timestamp_tr'].diff().dt.total_seconds()
    F['rapid_init_60s'] = tx_i[tx_i['itd'] < 60].groupby('id_user').size()
    F['rapid_init_300s'] = tx_i[tx_i['itd'] < 300].groupby('id_user').size()

    dc = tx.groupby(['id_user', 'tx_date'])['card_mask_hash'].nunique().reset_index(name='v')
    F['max_daily_cards'] = dc.groupby('id_user')['v'].max()
    F['mean_daily_cards'] = dc.groupby('id_user')['v'].mean()
    dtx = tx.groupby(['id_user', 'tx_date']).size().reset_index(name='v')
    F['max_daily_tx'] = dtx.groupby('id_user')['v'].max()
    F['mean_daily_tx'] = dtx.groupby('id_user')['v'].mean()
    F['active_days'] = dtx.groupby('id_user').size()
    dfl = tx[tx['status'] == 'fail'].groupby(['id_user', 'tx_date']).size().reset_index(name='v')
    F['max_daily_fails'] = dfl.groupby('id_user')['v'].max()
    ac = tx.groupby(['id_user', 'amount']).size().reset_index(name='v')
    F['max_same_amt'] = ac.groupby('id_user')['v'].max()

    F['holder_len_mean'] = g['holder_len'].mean()
    F['holder_len_std'] = g['holder_len'].std()
    F['holder_words_mean'] = g['holder_words'].mean()

    # ================================================================
    # 12. Status streaks
    # ================================================================
    print("  [v7] Status streaks...")
    tx['new_streak'] = (tx['is_fail'] != tx.groupby('id_user')['is_fail'].shift()).astype(int)
    tx['streak_id'] = tx.groupby('id_user')['new_streak'].cumsum()
    fail_only = tx[tx['is_fail'] == 1]
    streak_lens = fail_only.groupby(['id_user', 'streak_id']).size()
    F['max_consec_fails'] = streak_lens.groupby('id_user').max()
    F['num_fail_streaks'] = streak_lens.groupby('id_user').size()
    F['avg_fail_streak'] = streak_lens.groupby('id_user').mean()

    first_succ_idx = tx[tx['status'] == 'success'].groupby('id_user').head(1).set_index('id_user')['timestamp_tr']
    tx_with_fs = tx.join(first_succ_idx.rename('first_succ_time'), on='id_user')
    fb = tx_with_fs[(tx_with_fs['is_fail'] == 1) & (tx_with_fs['timestamp_tr'] < tx_with_fs['first_succ_time'])]
    F['fails_before_first_succ'] = fb.groupby('id_user').size()

    # ================================================================
    # 13. Fail-then-switch
    # ================================================================
    print("  [v7] Fail-then-switch...")
    tx['prev_card'] = tx.groupby('id_user')['card_mask_hash'].shift()
    tx['prev_status'] = tx.groupby('id_user')['status'].shift()
    tx['card_changed'] = (tx['card_mask_hash'] != tx['prev_card']).astype(int)
    tx['fail_then_switch'] = ((tx['prev_status'] == 'fail') & (tx['card_changed'] == 1)).astype(int)
    F['fail_switch_cnt'] = tx.groupby('id_user')['fail_then_switch'].sum()
    tx['succ_then_switch'] = ((tx['prev_status'] == 'success') & (tx['card_changed'] == 1)).astype(int)
    F['succ_switch_cnt'] = tx.groupby('id_user')['succ_then_switch'].sum()

    # ================================================================
    # 14. Per-card stats
    # ================================================================
    print("  [v7] Per-card stats...")
    card_stats = tx.groupby(['id_user', 'card_mask_hash']).agg(
        card_tx=('status', 'size'), card_fails=('is_fail', 'sum')
    ).reset_index()
    card_stats['card_fail_rate'] = card_stats['card_fails'] / card_stats['card_tx']
    F['mean_fail_rate_per_card'] = card_stats.groupby('id_user')['card_fail_rate'].mean()
    F['max_fail_rate_per_card'] = card_stats.groupby('id_user')['card_fail_rate'].max()
    F['cards_all_failed'] = card_stats[card_stats['card_fail_rate'] == 1.0].groupby('id_user').size()
    F['cards_any_success'] = card_stats[card_stats['card_fail_rate'] < 1.0].groupby('id_user').size()
    F['cards_used_once'] = card_stats[card_stats['card_tx'] == 1].groupby('id_user').size()

    # ================================================================
    # 15. Early windows
    # ================================================================
    print("  [v7] Early windows...")
    for hours in [1, 6, 24, 72, 168]:
        w = tx[tx['h_since_reg'] <= hours]
        w_g = w.groupby('id_user')
        F[f'ew{hours}_count'] = w_g.size()
        F[f'ew{hours}_fails'] = w[w['status'] == 'fail'].groupby('id_user').size()
        F[f'ew{hours}_cards'] = w_g['card_mask_hash'].nunique()
        F[f'ew{hours}_holders'] = w_g['card_holder'].nunique()
        if hours <= 24:
            F[f'ew{hours}_init_fail'] = w[
                (w['transaction_type'] == 'card_init') & (w['status'] == 'fail')
            ].groupby('id_user').size()

    # ================================================================
    # 16. Type transitions
    # ================================================================
    print("  [v7] Type transitions...")
    tx['prev_type'] = tx.groupby('id_user')['transaction_type'].shift()
    tx['init_to_init'] = ((tx['transaction_type'] == 'card_init') & (tx['prev_type'] == 'card_init')).astype(int)
    tx['init_after_fail'] = ((tx['transaction_type'] == 'card_init') & (tx['prev_status'] == 'fail')).astype(int)
    F['init_to_init_cnt'] = tx.groupby('id_user')['init_to_init'].sum()
    F['init_after_fail_cnt'] = tx.groupby('id_user')['init_after_fail'].sum()

    # ================================================================
    # 17. Amount patterns
    # ================================================================
    print("  [v7] Amount patterns...")
    F['small_amt_ratio'] = tx[tx['amount'] < 5].groupby('id_user').size() / F['tx_count'].clip(lower=1)
    F['tiny_amt_ratio'] = tx[tx['amount'] < 1].groupby('id_user').size() / F['tx_count'].clip(lower=1)
    fail_amt_counts = tx[tx['status'] == 'fail'].groupby(['id_user', 'amount']).size().reset_index(name='v')
    F['max_same_amt_fails'] = fail_amt_counts.groupby('id_user')['v'].max()

    # ================================================================
    # 18. Hourly velocity
    # ================================================================
    print("  [v7] Hourly velocity...")
    tx['hour_bucket'] = tx['timestamp_tr'].dt.floor('h')
    hourly = tx.groupby(['id_user', 'hour_bucket']).agg(
        htx=('status', 'size'), hcards=('card_mask_hash', 'nunique'), hfails=('is_fail', 'sum')
    ).reset_index()
    F['max_hourly_tx'] = hourly.groupby('id_user')['htx'].max()
    F['max_hourly_cards'] = hourly.groupby('id_user')['hcards'].max()
    F['max_hourly_fails'] = hourly.groupby('id_user')['hfails'].max()

    # ================================================================
    # 19. Email-holder match
    # ================================================================
    print("  [v7] Email-holder match...")
    def _safe_mode(x):
        if len(x) == 0: return ''
        m = x.mode()
        return m.iloc[0] if len(m) > 0 else (x.iloc[0] if len(x) > 0 else '')

    holder_mode = tx.groupby('id_user')['card_holder'].agg(_safe_mode)
    user_info = users_df[['id_user', 'email']].set_index('id_user')
    user_info['holder'] = holder_mode
    user_info['email_local'] = user_info['email'].str.split('@').str[0].str.lower().fillna('')
    user_info['email_alpha'] = user_info['email_local'].str.replace(r'[^a-z]', '', regex=True)
    user_info['holder_lower'] = user_info['holder'].str.lower().fillna('')

    def _match(row):
        e, h = row['email_alpha'], row['holder_lower']
        if len(e) < 3 or len(h) < 3: return 0
        prefix = e[:min(5, len(e))]
        if prefix in h: return 1
        for w in h.split():
            if len(w) >= 3 and w in e: return 1
        return 0

    user_info['email_holder_match'] = user_info.apply(_match, axis=1)
    F['email_holder_match'] = user_info['email_holder_match']

    # ================================================================
    # 20. Card lifecycle
    # ================================================================
    print("  [v7] Card lifecycle...")
    card_usage = tx.groupby(['id_user', 'card_mask_hash']).agg(
        first_use=('timestamp_tr', 'min'), last_use=('timestamp_tr', 'max'), n_uses=('status', 'size')
    ).reset_index()
    card_usage['card_lifespan_h'] = (card_usage['last_use'] - card_usage['first_use']).dt.total_seconds() / 3600
    F['mean_card_lifespan_h'] = card_usage.groupby('id_user')['card_lifespan_h'].mean()
    F['max_card_uses'] = card_usage.groupby('id_user')['n_uses'].max()
    F['mean_card_uses'] = card_usage.groupby('id_user')['n_uses'].mean()

    # ================================================================
    # 21. Error sequences
    # ================================================================
    print("  [v7] Error sequences...")
    tx['prev_error'] = tx.groupby('id_user')['error_group'].shift()
    tx['antifraud_then_newcard'] = ((tx['prev_error'] == 'antifraud') & (tx['card_changed'] == 1)).astype(int)
    tx['fraud_then_newcard'] = ((tx['prev_error'] == 'fraud') & (tx['card_changed'] == 1)).astype(int)
    F['antifraud_then_switch'] = tx.groupby('id_user')['antifraud_then_newcard'].sum()
    F['fraud_then_switch'] = tx.groupby('id_user')['fraud_then_newcard'].sum()

    # ================================================================
    # 22. Per-holder card dynamics (v8, kept)
    # ================================================================
    print("  [v8] Per-holder card dynamics...")
    holder_card = tx.groupby(['id_user', 'card_holder']).agg(
        hc_cards=('card_mask_hash', 'nunique'),
        hc_tx=('status', 'size'),
        hc_fails=('is_fail', 'sum')
    ).reset_index()
    holder_card['hc_fail_rate'] = holder_card['hc_fails'] / holder_card['hc_tx']

    F['max_cards_per_holder'] = holder_card.groupby('id_user')['hc_cards'].max()
    F['mean_cards_per_holder'] = holder_card.groupby('id_user')['hc_cards'].mean()
    F['holders_multi_cards'] = holder_card[holder_card['hc_cards'] > 1].groupby('id_user').size()

    card_holder_cnt = tx.groupby(['id_user', 'card_mask_hash'])['card_holder'].nunique().reset_index(name='n_holders')
    F['cards_multi_holders'] = card_holder_cnt[card_holder_cnt['n_holders'] > 1].groupby('id_user').size()

    F['max_holder_fail_rate'] = holder_card.groupby('id_user')['hc_fail_rate'].max()
    F['mean_holder_fail_rate'] = holder_card.groupby('id_user')['hc_fail_rate'].mean()

    # ================================================================
    # 23. Session features (v8, kept)
    # ================================================================
    print("  [v8] Session features...")
    tx['new_session'] = ((tx['tdiff'] > 1800) | tx['tdiff'].isna()).astype(int)
    tx['session_id'] = tx.groupby('id_user')['new_session'].cumsum()

    sess = tx.groupby(['id_user', 'session_id']).agg(
        s_tx=('status', 'size'),
        s_cards=('card_mask_hash', 'nunique'),
        s_fails=('is_fail', 'sum'),
        s_holders=('card_holder', 'nunique')
    ).reset_index()
    sess['s_fail_rate'] = sess['s_fails'] / sess['s_tx']

    F['num_sessions'] = sess.groupby('id_user').size()
    F['max_session_tx'] = sess.groupby('id_user')['s_tx'].max()
    F['max_session_cards'] = sess.groupby('id_user')['s_cards'].max()
    F['max_session_fails'] = sess.groupby('id_user')['s_fails'].max()
    F['max_session_fail_rate'] = sess.groupby('id_user')['s_fail_rate'].max()
    F['sessions_multi_cards'] = sess[sess['s_cards'] > 1].groupby('id_user').size()

    # ================================================================
    # 24. Behavioral shift (v8, kept)
    # ================================================================
    print("  [v8] Behavioral shift...")
    tx['tx_rank'] = tx.groupby('id_user').cumcount()
    tx['tx_total'] = tx.groupby('id_user')['status'].transform('size')
    tx['is_first_half'] = (tx['tx_rank'] < tx['tx_total'] / 2).astype(int)

    fh = tx[tx['is_first_half'] == 1]
    sh = tx[tx['is_first_half'] == 0]

    F['fh_fail_rate'] = fh.groupby('id_user')['is_fail'].mean()
    F['sh_fail_rate'] = sh.groupby('id_user')['is_fail'].mean()
    F['fail_rate_shift'] = F.get('fh_fail_rate', pd.Series(dtype=float)).fillna(0) - F.get('sh_fail_rate', pd.Series(dtype=float)).fillna(0)

    F['fh_cards'] = fh.groupby('id_user')['card_mask_hash'].nunique()
    F['sh_cards'] = sh.groupby('id_user')['card_mask_hash'].nunique()

    # ================================================================
    # 25. Same-card retry (v8, kept)
    # ================================================================
    print("  [v8] Same-card retry...")
    tx['fail_same_card'] = (
        (tx['prev_status'] == 'fail') & (tx['card_changed'] == 0)
    ).astype(int)
    F['fail_same_card_cnt'] = tx.groupby('id_user')['fail_same_card'].sum()

    # [v8 Section 26 - Error per-card diversity: REMOVED (noise, ablation = -0.0005)]

    # ================================================================
    # 27. Success characteristics (v8, kept)
    # ================================================================
    print("  [v8] Success characteristics...")
    succ_tx = tx[tx['status'] == 'success']
    F['succ_on_init'] = succ_tx[succ_tx['transaction_type'] == 'card_init'].groupby('id_user').size()
    F['succ_mm_rate'] = succ_tx.groupby('id_user')['mm_total'].mean()
    F['succ_cards'] = succ_tx.groupby('id_user')['card_mask_hash'].nunique()

    # ================================================================
    # 28. Velocity acceleration (v9 NEW)
    # Anti-fraud: fraudsters ACCELERATE -- gaps between transactions
    # get shorter as they find working cards and rush to exploit them.
    # Legit users have steady or random pacing.
    # ================================================================
    print("  [v9] Velocity acceleration...")
    # accel = tdiff[i-1] - tdiff[i]; positive means speeding up
    tx['accel'] = -tx.groupby('id_user')['tdiff'].diff()
    F['mean_acceleration'] = tx.groupby('id_user')['accel'].mean()
    F['max_acceleration'] = tx.groupby('id_user')['accel'].max()

    # ================================================================
    # COMBINE
    # ================================================================
    features = pd.DataFrame(F).fillna(0)

    # ================================================================
    # DERIVED RATIOS
    # ================================================================
    tc = features['tx_count'].clip(lower=1)
    uc = features['u_cards'].clip(lower=1)

    # v7 ratios
    features['fail_rate'] = features['tx_fail'] / tc
    features['success_rate'] = features['tx_success'] / tc
    features['cards_per_tx'] = features['u_cards'] / tc
    features['init_rate'] = features['tt_card_init'] / tc
    features['recur_rate'] = features['tt_card_recurring'] / tc
    features['fraud_err_rate'] = features['err_fraud'] / tc
    features['antifraud_err_rate'] = features['err_antifraud'] / tc
    features['err_rate'] = features['err_count'] / tc
    features['tx_per_hour'] = tc / features['tx_span_h'].clip(lower=0.01)
    features['holders_per_card'] = features['u_holders'] / uc
    features['init_fail_rate'] = features['init_fail'] / features['tt_card_init'].clip(lower=1)
    features['init_succ_rate'] = features['init_succ'] / features['tt_card_init'].clip(lower=1)
    features['fail_to_succ'] = features['tx_fail'] / features['tx_success'].clip(lower=1)
    features['tx_per_day'] = tc / features['active_days'].clip(lower=1)
    features['amt_per_card'] = features['amt_sum'] / uc
    features['cards_per_day'] = features['u_cards'] / features['active_days'].clip(lower=1)
    features['fail_per_card'] = features['tx_fail'] / uc
    features['err_per_card'] = features['err_count'] / uc
    features['has_success'] = (features['tx_success'] > 0).astype(int)
    features['has_recurring'] = (features['tt_card_recurring'] > 0).astype(int)
    features['has_fraud_err'] = (features['err_fraud'] > 0).astype(int)
    features['has_antifraud_err'] = (features['err_antifraud'] > 0).astype(int)
    features['pct_prepaid'] = features['ct_PREPAID'] / tc
    features['pct_credit'] = features['ct_CREDIT'] / tc
    features['err_diversity'] = features['err_unique'] / features['err_count'].clip(lower=1)
    features['amt_cv'] = features['amt_std'] / features['amt_mean'].clip(lower=0.01)
    features['rapid_fail_per_tx'] = features['rapid_fail_60s'] / tc
    features['init_fail_per_card'] = features['init_fail'] / uc
    features['fail_switch_rate'] = features['fail_switch_cnt'] / tc
    features['cards_all_failed_pct'] = features['cards_all_failed'] / uc
    features['cards_once_pct'] = features['cards_used_once'] / uc
    features['consec_fails_per_card'] = features['max_consec_fails'] / uc
    features['init_to_init_rate'] = features['init_to_init_cnt'] / tc
    features['init_after_fail_rate'] = features['init_after_fail_cnt'] / tc

    for hours in [1, 6, 24, 72, 168]:
        cnt = features[f'ew{hours}_count'].clip(lower=1)
        features[f'ew{hours}_fail_rate'] = features[f'ew{hours}_fails'] / cnt
        features[f'ew{hours}_cards_rate'] = features[f'ew{hours}_cards'] / cnt
        if hours <= 24:
            features[f'ew{hours}_init_fail_rate'] = features[f'ew{hours}_init_fail'] / cnt

    # v8 ratios (kept)
    features['fail_same_card_rate'] = features['fail_same_card_cnt'] / tc
    features['switch_vs_retry_ratio'] = features['fail_switch_cnt'] / features['fail_same_card_cnt'].clip(lower=1)
    features['sessions_multi_cards_pct'] = features['sessions_multi_cards'] / features['num_sessions'].clip(lower=1)
    features['cards_multi_holders_pct'] = features['cards_multi_holders'] / uc
    features['succ_on_init_rate'] = features['succ_on_init'] / features['tx_success'].clip(lower=1)
    features['succ_cards_pct'] = features['succ_cards'] / uc

    # ================================================================
    # v9 NEW: Specialized low-card user features
    # Anti-fraud: 4,114 of 4,843 FN have only 1 card -- card-switching
    # features are zero for them. Need targeted signals for these users.
    # ================================================================
    print("  [v9] Low-card user features...")
    is_single_card = (features['u_cards'] == 1)

    # For single-card users: country mismatch between card and registration
    user_card_ctry = tx.groupby('id_user')['card_country'].first()
    user_reg_ctry = tx.groupby('id_user')['reg_country'].first()
    has_ctry_mm = (user_card_ctry != user_reg_ctry).reindex(features.index).fillna(False)
    features['single_card_country_mm'] = (is_single_card & has_ctry_mm).astype(int)

    # Single-card user with antifraud error
    has_antifraud = (features['err_antifraud'] > 0)
    features['single_card_antifraud'] = (is_single_card & has_antifraud).astype(int)

    # Single-card user with fraud error
    has_fraud_e = (features['err_fraud'] > 0)
    features['single_card_fraud_err'] = (is_single_card & has_fraud_e).astype(int)

    # User whose ALL transactions are card_init only
    features['init_only_user'] = (
        (features['tt_card_init'] > 0) & (features['tt_card_init'] == features['tx_count'])
    ).astype(int)

    # Low-activity user with failed card_init
    features['low_tx_fail_init'] = (
        (features['tx_count'] <= 5) & (features['init_fail'] > 0)
    ).astype(int)

    # [v8 Log transforms: REMOVED (marginal, 8 features for +0.0002)]

    # ================================================================
    # INTERACTIONS (v7 + v8 kept)
    # ================================================================
    features['fail_x_cards'] = features['tx_fail'] * features['u_cards']
    features['fail_x_mm'] = features['tx_fail'] * features['mm_total_cnt']
    features['err_x_cards'] = features['err_count'] * features['u_cards']
    features['antifraud_x_cards'] = features['err_antifraud'] * features['u_cards']
    features['init_fail_x_holders'] = features['init_fail'] * features['u_holders']
    features['fail_switch_x_cards'] = features['fail_switch_cnt'] * features['u_cards']
    features['consec_fails_x_cards'] = features['max_consec_fails'] * features['u_cards']
    features['ew24_fails_x_cards'] = features['ew24_fails'] * features['u_cards']
    features['antifraud_x_switch'] = features['err_antifraud'] * features['fail_switch_cnt']

    # v8 interactions (kept)
    features['session_cards_x_fails'] = features['max_session_cards'] * features['max_session_fails']
    features['holders_multi_x_cards'] = features['holders_multi_cards'] * features['u_cards']
    features['fail_shift_x_cards'] = features['fail_rate_shift'] * features['u_cards']

    return features


# ================================================================
# A. CROSS-USER CARD FEATURES
# Anti-fraud: 94,403 cards used by 2+ users across train+test.
# Shared cards are the #1 indicator of fraud rings --
# fraudsters distribute stolen card numbers across accounts.
# ================================================================
def build_cross_user_card_features(train_tx, test_tx):
    print("  [v9] Cross-user card sharing...")
    all_tx = pd.concat([
        train_tx[['id_user', 'card_mask_hash']],
        test_tx[['id_user', 'card_mask_hash']]
    ])

    # Per card: how many unique users
    card_n_users = all_tx.groupby('card_mask_hash')['id_user'].nunique().rename('card_n_users')

    # Per user: aggregate over their cards
    user_cards = all_tx[['id_user', 'card_mask_hash']].drop_duplicates()
    user_cards = user_cards.merge(card_n_users.reset_index(), on='card_mask_hash')

    result = user_cards.groupby('id_user')['card_n_users'].agg(
        max_card_n_users='max',
        mean_card_n_users='mean'
    )

    total_cards = user_cards.groupby('id_user').size().rename('_total')
    shared2 = user_cards[user_cards['card_n_users'] >= 2].groupby('id_user').size()
    shared3 = user_cards[user_cards['card_n_users'] >= 3].groupby('id_user').size()

    result['cards_shared_2plus'] = shared2
    result['cards_shared_3plus'] = shared3
    result = result.fillna(0)
    result['pct_shared_cards'] = result['cards_shared_2plus'] / total_cards.clip(lower=1)

    return result


# ================================================================
# C. CROSS-USER HOLDER FEATURES
# Anti-fraud: same holder name across multiple users suggests
# name reuse in fraud rings or shared identity fraud.
# ================================================================
def build_cross_user_holder_features(train_tx, test_tx):
    print("  [v9] Cross-user holder sharing...")
    all_tx = pd.concat([
        train_tx[['id_user', 'card_holder']],
        test_tx[['id_user', 'card_holder']]
    ])

    holder_n_users = all_tx.groupby('card_holder')['id_user'].nunique().rename('holder_n_users')

    user_holders = all_tx[['id_user', 'card_holder']].drop_duplicates()
    user_holders = user_holders.merge(holder_n_users.reset_index(), on='card_holder')

    result = user_holders.groupby('id_user')['holder_n_users'].agg(
        max_holder_n_users='max',
        mean_holder_n_users='mean'
    )
    return result


# ================================================================
# B. CARD TOXICITY (LOO target encoding)
# Anti-fraud: if OTHER users who share the same card are fraudsters,
# this user is likely part of the same fraud ring.
# LOO on train to avoid leakage; direct stats for test.
# ================================================================
def build_card_toxicity(train_tx, test_tx, train_users):
    print("  [v9] Card toxicity (LOO)...")

    # Step 1: unique (card, user, is_fraud) from train
    user_card = train_tx[['id_user', 'card_mask_hash']].drop_duplicates()
    user_card = user_card.merge(train_users[['id_user', 'is_fraud']], on='id_user')

    # Step 2: per card stats
    card_stats = user_card.groupby('card_mask_hash').agg(
        n_users=('id_user', 'count'),
        n_fraud=('is_fraud', 'sum')
    )

    # Step 3: LOO for train -- exclude this user's own label
    user_card = user_card.merge(card_stats, on='card_mask_hash')
    user_card['loo_fraud_rate'] = np.where(
        user_card['n_users'] > 1,
        (user_card['n_fraud'] - user_card['is_fraud']) / (user_card['n_users'] - 1),
        0.0
    )

    train_tox = user_card.groupby('id_user')['loo_fraud_rate'].agg(
        max_card_toxicity='max',
        mean_card_toxicity='mean',
        sum_card_toxicity='sum'
    )
    train_tox['has_toxic_card'] = (train_tox['max_card_toxicity'] > 0).astype(int)

    # Step 4: For test, use full train card stats (no leakage)
    card_fraud_rate = (card_stats['n_fraud'] / card_stats['n_users']).rename('card_fraud_rate')

    test_user_card = test_tx[['id_user', 'card_mask_hash']].drop_duplicates()
    test_user_card = test_user_card.merge(card_fraud_rate.reset_index(), on='card_mask_hash', how='left')
    test_user_card['card_fraud_rate'] = test_user_card['card_fraud_rate'].fillna(0)

    test_tox = test_user_card.groupby('id_user')['card_fraud_rate'].agg(
        max_card_toxicity='max',
        mean_card_toxicity='mean',
        sum_card_toxicity='sum'
    )
    test_tox['has_toxic_card'] = (test_tox['max_card_toxicity'] > 0).astype(int)

    return train_tox, test_tox


# ================================================================
# HOLDER TOXICITY (LOO target encoding)
# Anti-fraud: same logic as card toxicity but for holder names.
# If other users with the same holder name are fraudsters,
# the name is likely fake/shared in a fraud operation.
# ================================================================
def build_holder_toxicity(train_tx, test_tx, train_users):
    print("  [v9] Holder toxicity (LOO)...")

    user_holder = train_tx[['id_user', 'card_holder']].drop_duplicates()
    user_holder = user_holder.merge(train_users[['id_user', 'is_fraud']], on='id_user')

    holder_stats = user_holder.groupby('card_holder').agg(
        n_users=('id_user', 'count'),
        n_fraud=('is_fraud', 'sum')
    )

    user_holder = user_holder.merge(holder_stats, on='card_holder')
    user_holder['loo_fraud_rate'] = np.where(
        user_holder['n_users'] > 1,
        (user_holder['n_fraud'] - user_holder['is_fraud']) / (user_holder['n_users'] - 1),
        0.0
    )

    train_htox = user_holder.groupby('id_user')['loo_fraud_rate'].agg(
        max_holder_toxicity='max',
        mean_holder_toxicity='mean',
        sum_holder_toxicity='sum'
    )
    train_htox['has_toxic_holder'] = (train_htox['max_holder_toxicity'] > 0).astype(int)

    holder_fraud_rate = (holder_stats['n_fraud'] / holder_stats['n_users']).rename('holder_fraud_rate')

    test_user_holder = test_tx[['id_user', 'card_holder']].drop_duplicates()
    test_user_holder = test_user_holder.merge(holder_fraud_rate.reset_index(), on='card_holder', how='left')
    test_user_holder['holder_fraud_rate'] = test_user_holder['holder_fraud_rate'].fillna(0)

    test_htox = test_user_holder.groupby('id_user')['holder_fraud_rate'].agg(
        max_holder_toxicity='max',
        mean_holder_toxicity='mean',
        sum_holder_toxicity='sum'
    )
    test_htox['has_toxic_holder'] = (test_htox['max_holder_toxicity'] > 0).astype(int)

    return train_htox, test_htox


# ================================================================
# D. TRANSACTION-LEVEL TARGET ENCODING
# Anti-fraud: card_country, card_brand, currency, payment_country
# have very different fraud rates (Indonesia 95%, Nigeria 86%,
# DISCOVER 27%, USD 17.6%). Encoding at tx level then aggregating
# captures the risk profile of a user's transaction mix.
# OOF by id_user to avoid leakage.
# ================================================================
def build_tx_level_te(train_tx, test_tx, train_users):
    print("  [v9] Transaction-level target encoding...")
    gm = train_users['is_fraud'].mean()
    smoothing = 50

    train_tx_m = train_tx.merge(train_users[['id_user', 'is_fraud']], on='id_user')

    # Build user -> fold mapping once
    unique_users = train_users[['id_user', 'is_fraud']].copy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(unique_users['id_user'], unique_users['is_fraud']))

    user_fold = {}
    for fold_idx, (tri, vai) in enumerate(folds):
        for uid in unique_users.iloc[vai]['id_user'].values:
            user_fold[uid] = fold_idx

    train_tx_m['_fold'] = train_tx_m['id_user'].map(user_fold)

    train_agg = {}
    test_agg = {}

    for field in ['card_country', 'card_brand', 'currency', 'payment_country']:
        print(f"    TE: {field}...")

        # OOF encoding for train transactions
        encoded = np.full(len(train_tx_m), gm)

        for fold_idx in range(5):
            # Stats from all OTHER folds
            train_fold_mask = train_tx_m['_fold'] != fold_idx
            fold_tx = train_tx_m[train_fold_mask]
            st = fold_tx.groupby(field)['is_fraud'].agg(['mean', 'count'])
            sm = (st['count'] * st['mean'] + smoothing * gm) / (st['count'] + smoothing)

            # Apply to this fold's transactions
            val_mask = (train_tx_m['_fold'] == fold_idx).values
            encoded[val_mask] = train_tx_m.loc[val_mask, field].map(sm).fillna(gm).values

        # Full stats for test
        st_full = train_tx_m.groupby(field)['is_fraud'].agg(['mean', 'count'])
        sm_full = (st_full['count'] * st_full['mean'] + smoothing * gm) / (st_full['count'] + smoothing)

        # Train aggregation per user
        train_tx_m[f'_te_{field}'] = encoded
        train_agg[f'txte_{field}_mean'] = train_tx_m.groupby('id_user')[f'_te_{field}'].mean()
        train_agg[f'txte_{field}_max'] = train_tx_m.groupby('id_user')[f'_te_{field}'].max()

        # Test aggregation per user
        test_te_vals = test_tx[field].map(sm_full).fillna(gm)
        _tmp = pd.DataFrame({'id_user': test_tx['id_user'].values, 'v': test_te_vals.values})
        test_agg[f'txte_{field}_mean'] = _tmp.groupby('id_user')['v'].mean()
        test_agg[f'txte_{field}_max'] = _tmp.groupby('id_user')['v'].max()

    return pd.DataFrame(train_agg), pd.DataFrame(test_agg)


# ================================================================
# USER FEATURES (same as v8)
# ================================================================
def build_user_features(users_df):
    df = users_df.copy()
    df['email_domain'] = df['email'].str.split('@').str[1].fillna('unknown')
    df['email_local'] = df['email'].str.split('@').str[0].fillna('')

    major = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com',
             'mail.com', 'protonmail.com', 'aol.com', 'zoho.com', 'yandex.com',
             'live.com', 'gmx.com']
    df['email_prov'] = df['email_domain'].apply(lambda x: x if x in major else 'other')
    df['email_has_num'] = df['email_local'].str.contains(r'\d', regex=True).fillna(False).astype(int)
    df['email_len'] = df['email_local'].str.len().fillna(0)
    df['email_has_uscore'] = df['email_local'].str.contains(r'_', regex=True).fillna(False).astype(int)
    df['email_ndigits'] = df['email_local'].str.count(r'\d').fillna(0)
    df['email_digit_ratio'] = df['email_ndigits'] / df['email_len'].clip(lower=1)
    df['reg_hour'] = df['timestamp_reg'].dt.hour
    df['reg_dow'] = df['timestamp_reg'].dt.dayofweek
    df['reg_month'] = df['timestamp_reg'].dt.month
    df['reg_night'] = df['reg_hour'].between(0, 5).astype(int)
    df['gender_enc'] = (df['gender'] == 'male').astype(int)
    tmap = {'organic': 0, 'ppc': 1, 'cpa': 2, 'remarketing': 3, 'unknown': 4}
    df['traffic_enc'] = df['traffic_type'].map(tmap).fillna(4)
    pmap = {'gmail.com': 0, 'yahoo.com': 1, 'outlook.com': 2, 'hotmail.com': 3,
            'icloud.com': 4, 'protonmail.com': 5, 'other': 6}
    df['prov_enc'] = df['email_prov'].map(pmap).fillna(6)
    cf = df['reg_country'].value_counts()
    df['country_freq'] = df['reg_country'].map(cf)
    domf = df['email_domain'].value_counts()
    df['domain_freq'] = df['email_domain'].map(domf)

    cols = ['gender_enc', 'traffic_enc', 'prov_enc',
            'email_has_num', 'email_len', 'email_has_uscore',
            'email_ndigits', 'email_digit_ratio',
            'reg_hour', 'reg_dow', 'reg_month', 'reg_night',
            'country_freq', 'domain_freq']
    return df.set_index('id_user')[cols]


# ================================================================
# MAIN ENTRY POINT
# ================================================================
def build_all_features(train_users, test_users, train_tx, test_tx):
    # 1. Per-user transaction features
    print("Building transaction features...")
    train_tx_f = build_tx_features(train_tx, train_users)
    test_tx_f = build_tx_features(test_tx, test_users)

    # 2. Cross-user card features (A) -- computed on combined tx
    cross_card = build_cross_user_card_features(train_tx, test_tx)

    # 3. Cross-user holder features (C)
    cross_holder = build_cross_user_holder_features(train_tx, test_tx)

    # 4. Card toxicity (B) -- LOO for train, direct for test
    train_card_tox, test_card_tox = build_card_toxicity(train_tx, test_tx, train_users)

    # 5. Holder toxicity -- LOO for train, direct for test
    train_holder_tox, test_holder_tox = build_holder_toxicity(train_tx, test_tx, train_users)

    # 6. Transaction-level target encoding (D)
    train_txte, test_txte = build_tx_level_te(train_tx, test_tx, train_users)

    # 7. User features
    print("Building user features...")
    train_uf = build_user_features(train_users)
    test_uf = build_user_features(test_users)

    # 8. User-level target encoding (same as v8)
    print("Target encoding...")
    tui = train_users.set_index('id_user')
    tei = test_users.set_index('id_user')
    yt = tui['is_fraud']

    for col in ['reg_country', 'traffic_type']:
        tr, te = target_encode_cv(tui[col], tei[col], yt)
        train_uf[f'{col}_te'] = tr
        test_uf[f'{col}_te'] = te

    tui['edom'] = tui['email'].str.split('@').str[1].fillna('unk')
    tei['edom'] = tei['email'].str.split('@').str[1].fillna('unk')
    tr, te = target_encode_cv(tui['edom'], tei['edom'], yt)
    train_uf['domain_te'] = tr
    test_uf['domain_te'] = te

    tui['gt'] = tui['gender'] + '_' + tui['traffic_type'].fillna('unk')
    tei['gt'] = tei['gender'] + '_' + tei['traffic_type'].fillna('unk')
    tr, te = target_encode_cv(tui['gt'], tei['gt'], yt)
    train_uf['gender_traffic_te'] = tr
    test_uf['gender_traffic_te'] = te

    tui['ct'] = tui['reg_country'].fillna('unk') + '_' + tui['traffic_type'].fillna('unk')
    tei['ct'] = tei['reg_country'].fillna('unk') + '_' + tei['traffic_type'].fillna('unk')
    tr, te = target_encode_cv(tui['ct'], tei['ct'], yt, smoothing=100)
    train_uf['country_traffic_te'] = tr
    test_uf['country_traffic_te'] = te

    # 9. Split cross-user features by train/test user IDs
    train_ids = set(train_users['id_user'].values)
    test_ids = set(test_users['id_user'].values)

    train_cross_card = cross_card[cross_card.index.isin(train_ids)]
    test_cross_card = cross_card[cross_card.index.isin(test_ids)]

    train_cross_holder = cross_holder[cross_holder.index.isin(train_ids)]
    test_cross_holder = cross_holder[cross_holder.index.isin(test_ids)]

    # 10. Join everything
    train_X = train_uf.join(train_tx_f, how='left') \
                       .join(train_cross_card, how='left') \
                       .join(train_cross_holder, how='left') \
                       .join(train_card_tox, how='left') \
                       .join(train_holder_tox, how='left') \
                       .join(train_txte, how='left') \
                       .fillna(0)

    test_X = test_uf.join(test_tx_f, how='left') \
                     .join(test_cross_card, how='left') \
                     .join(test_cross_holder, how='left') \
                     .join(test_card_tox, how='left') \
                     .join(test_holder_tox, how='left') \
                     .join(test_txte, how='left') \
                     .fillna(0)

    # 11. v9 Enhanced interactions (G) -- combining new features with existing
    for df in [train_X, test_X]:
        df['card_toxicity_x_cards'] = df['max_card_toxicity'] * df['u_cards']
        df['card_toxicity_x_fail_rate'] = df['max_card_toxicity'] * df['fail_rate']
        df['shared_cards_x_fail_switch'] = df['cards_shared_2plus'] * df['fail_switch_cnt']
        df['antifraud_x_country_mm'] = df['err_antifraud'] * df['mm_card_reg_cnt']

    train_X['has_tx'] = (train_X['tx_count'] > 0).astype(int)
    test_X['has_tx'] = (test_X['tx_count'] > 0).astype(int)

    common = sorted(set(train_X.columns) & set(test_X.columns))
    train_X = train_X[common]
    test_X = test_X[common]

    y = train_users.set_index('id_user')['is_fraud'].loc[train_X.index].values
    feature_names = list(train_X.columns)

    print(f"\nFeatures: {len(feature_names)}, Train: {len(train_X)}, Fraud: {y.mean():.4f}")
    return train_X, test_X, y, feature_names
