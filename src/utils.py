"""Shared utilities for the Amazon Electronics Recommendation System.
Group 5 — IE University, 2025-26
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────
K = 10
RELEVANCE_THRESHOLD = 4.0
DAMPING_FACTOR = 50


# ── Data loading ─────────────────────────────────────────────

def load_data(data_dir=None):
    """Load reviews and metadata CSVs. IDs are read as strings to preserve leading zeros."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_dir = Path(data_dir)
    df = pd.read_csv(
        data_dir / 'amazon_electronics_filtered.csv',
        dtype={'user_id': str, 'item_id': str}
    )
    df_meta = pd.read_csv(
        data_dir / 'amazon_meta_filtered.csv',
        dtype={'asin': str}
    )
    return df, df_meta


def temporal_train_test_split(df, train_frac=0.8):
    """Sort by timestamp and split at train_frac. Returns (train_df, test_df, info_dict)."""
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_frac)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df  = df_sorted.iloc[split_idx:].copy()

    train_users = set(train_df['user_id'].unique())
    train_items = set(train_df['item_id'].unique())
    test_users  = set(test_df['user_id'].unique())
    test_items  = set(test_df['item_id'].unique())

    info = {
        'cold_users':  test_users - train_users,
        'cold_items':  test_items - train_items,
        'train_users': train_users,
        'train_items': train_items,
        'test_users':  test_users,
        'test_items':  test_items,
    }
    print('=== TRAIN / TEST SPLIT ===')
    print(f'Train: {len(train_df):,} interactions | {len(train_users):,} users | {len(train_items):,} items')
    print(f'Test:  {len(test_df):,} interactions  | {len(test_users):,} users  | {len(test_items):,} items')
    print(f'Cold-start users (test only): {len(info["cold_users"]):,}')
    print(f'Cold-start items (test only): {len(info["cold_items"]):,}')
    return train_df, test_df, info


def add_temporal_features(df):
    """Add datetime, year, month, dayofweek columns to a copy of df."""
    df = df.copy()
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['year']       = df['datetime'].dt.year
    df['month']      = df['datetime'].dt.month
    df['dayofweek']  = df['datetime'].dt.dayofweek
    return df


# ── Lookup structures ─────────────────────────────────────────

def build_lookup_structures(train_df, test_df):
    """Build dicts used by evaluation functions across all notebooks."""
    train_user_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    global_mean = float(train_df['rating'].mean())
    test_user_relevant = (
        test_df[test_df['rating'] >= RELEVANCE_THRESHOLD]
        .groupby('user_id')['item_id'].apply(set).to_dict()
    )
    test_user_ratings = (
        test_df.groupby('user_id')
        .apply(lambda x: dict(zip(x['item_id'], x['rating'])))
        .to_dict()
    )
    return {
        'train_user_items':   train_user_items,
        'global_mean':        global_mean,
        'test_user_relevant': test_user_relevant,
        'test_user_ratings':  test_user_ratings,
        'all_items':          set(train_df['item_id'].unique()),
    }


def sample_eval_users(lookups, n=1000, random_state=42):
    """Return n users who have ≥1 relevant test item and ≥5 train interactions."""
    np.random.seed(random_state)
    train_user_items   = lookups['train_user_items']
    test_user_relevant = lookups['test_user_relevant']
    eligible = [
        u for u in test_user_relevant
        if u in train_user_items
        and len(train_user_items[u]) >= 5
        and len(test_user_relevant[u]) >= 1
    ]
    selected = list(np.random.choice(eligible, size=min(n, len(eligible)), replace=False))
    print(f'Sampled {len(selected):,} users for ranking evaluation (from {len(eligible):,} eligible)')
    return selected


# ── Metric functions ──────────────────────────────────────────

def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def mae(actual, predicted):
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))


def precision_at_k(recommended, relevant, k=K):
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    return len(set(rec_k) & set(relevant)) / len(rec_k)


def recall_at_k(recommended, relevant, k=K):
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended, relevant_with_ratings, k=K):
    """relevant_with_ratings: dict {item_id: rating}"""
    rec_k = recommended[:k]
    dcg = sum(
        relevant_with_ratings[item] / np.log2(i + 2)
        for i, item in enumerate(rec_k)
        if item in relevant_with_ratings
    )
    ideal = sorted(relevant_with_ratings.values(), reverse=True)[:k]
    idcg  = sum(s / np.log2(i + 2) for i, s in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def coverage(all_recommendations, total_items, k=K):
    unique_rec = set()
    for recs in all_recommendations:
        unique_rec.update(recs[:k])
    return len(unique_rec) / len(total_items)


def diversity_intra_list(recommendations, item_feature_matrix, item_to_idx, k=K):
    """Mean intra-list diversity (1 − avg cosine similarity) across all recommendation lists."""
    if item_feature_matrix is None:
        return None
    distances = []
    for rec_list in recommendations:
        rec_k = rec_list[:k]
        idxs = [item_to_idx[it] for it in rec_k if it in item_to_idx]
        if len(idxs) < 2:
            continue
        feat = item_feature_matrix[idxs]
        if issparse(feat):
            feat = feat.toarray()
        sim = sk_cosine(feat)
        n = len(idxs)
        avg_dist = 1.0 - (sim.sum() - n) / (n * (n - 1))
        distances.append(avg_dist)
    return float(np.mean(distances)) if distances else 0.0


def serendipity(rec_dict, train_user_items, test_user_relevant,
                item_feature_matrix, item_to_idx, k=K):
    """Mean serendipity: avg over users of (1-sim_to_profile) * is_relevant, averaged over top-K.
    rec_dict: {user_id: [item_id, ...]}
    """
    if item_feature_matrix is None:
        return None
    scores = []
    for user_id, rec_list in rec_dict.items():
        rec_k = rec_list[:k]
        relevant = test_user_relevant.get(user_id, set())
        history_idxs = [
            item_to_idx[it]
            for it in train_user_items.get(user_id, set())
            if it in item_to_idx
        ]
        if not history_idxs or not rec_k:
            continue
        prof = item_feature_matrix[history_idxs]
        if issparse(prof):
            prof = prof.toarray()
        user_profile = np.asarray(prof.mean(axis=0)).reshape(1, -1)
        user_score = 0.0
        for item in rec_k:
            if item not in item_to_idx:
                continue
            feat = item_feature_matrix[item_to_idx[item]]
            if issparse(feat):
                feat = np.asarray(feat.todense())
            feat = np.asarray(feat).reshape(1, -1)
            sim  = float(sk_cosine(user_profile, feat)[0, 0])
            user_score += (1.0 - sim) * (1.0 if item in relevant else 0.0)
        scores.append(user_score / k)
    return float(np.mean(scores)) if scores else 0.0


# ── Unified ranking evaluation ────────────────────────────────

def evaluate_ranking(recommend_fn, sample_users, lookups, k=K):
    """Call recommend_fn(user_id) for each user; return (all_recs, metrics_dict).

    recommend_fn must accept a user_id and return a list of item_ids.
    """
    tui  = lookups['train_user_items']
    tur  = lookups['test_user_relevant']
    turr = lookups['test_user_ratings']
    all_items = lookups['all_items']

    all_recs, prec, rec, ndcg = [], [], [], []
    for user_id in sample_users:
        recs = recommend_fn(user_id)
        all_recs.append(recs)
        relevant    = tur.get(user_id, set())
        rel_ratings = {it: r for it, r in turr.get(user_id, {}).items() if r >= RELEVANCE_THRESHOLD}
        prec.append(precision_at_k(recs, relevant, k))
        rec.append(recall_at_k(recs, relevant, k))
        ndcg.append(ndcg_at_k(recs, rel_ratings, k))

    metrics = {
        f'Precision@{k}': float(np.mean(prec)),
        f'Recall@{k}':    float(np.mean(rec)),
        f'NDCG@{k}':      float(np.mean(ndcg)),
        'Coverage':       coverage(all_recs, all_items, k),
    }
    return all_recs, metrics


def print_metrics(name, rmse_val, mae_val, ranking_metrics):
    """Pretty-print evaluation results."""
    print(f'\n{"="*55}')
    print(f'  {name}')
    print(f'{"="*55}')
    print(f'  RMSE : {rmse_val:.4f}')
    print(f'  MAE  : {mae_val:.4f}')
    for key, val in ranking_metrics.items():
        if val is not None:
            print(f'  {key:<18s}: {val:.4f}')
        else:
            print(f'  {key:<18s}: N/A')
