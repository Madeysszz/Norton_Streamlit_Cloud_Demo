import json
from itertools import combinations
from math import ceil

import numpy as np
import pandas as pd


def build_demo_data(seed=2026):
    rng = np.random.default_rng(seed)
    models = ("if", "ecod", "lof")
    parts = []
    for channel in ("cobrandgen", "retail"):
        count = 64
        data = pd.DataFrame({
            "log_key": [f"demo-{channel}-log-{i + 1:04d}" for i in range(count)],
            "channel": channel,
            "account_key": [f"demo-{channel}-account-{i % 8 + 1:02d}" for i in range(count)],
            "reference_key": [f"demo-{channel}-reference-{i // 2 + 1:03d}" for i in range(count)],
            "created_at": pd.Timestamp("2030-01-01T08:00:00Z")
                + pd.to_timedelta(np.arange(count) * 360 + rng.integers(0, 60, count), unit="m"),
            "status_snapshot": np.resize(["ACTIVE", "CANCEL", "PENDING", "ACTIVE"], count),
            "amount": rng.integers(20, 501, count),
        })
        accounts = data.groupby("account_key", sort=False)
        data["prior_history_count"] = accounts.cumcount()
        data["seconds_since_previous_log"] = accounts.created_at.diff().dt.total_seconds()
        data["prior_channel_median_amount"] = data.amount.expanding().median().shift(1)
        latent = rng.uniform(0, 1, count)
        for model, noise in zip(models, (.15, .25, .75)):
            scores = latent + rng.normal(0, noise, count)
            # Satu skenario per kanal membuat contoh pada ambang 0,99 tetap terlihat.
            scores[-1] = scores.max() + 1
            data[f"{model}_percentile"] = pd.Series(scores).rank(method="average", pct=True)
        data["consensus_score"] = data[[f"{model}_percentile" for model in models]].mean(axis=1)
        data["anomaly_rank"] = data.consensus_score.rank(method="first", ascending=False).astype(int)
        data["investigation_priority"] = np.select(
            [data.anomaly_rank.le(ceil(count * .01)), data.anomaly_rank.le(ceil(count * .05))],
            ["tinggi", "sedang"], default="rendah",
        )
        data["reasons"] = "Log demonstrasi sintetis; skor model merupakan simulasi untuk tampilan dashboard."
        data.loc[data.anomaly_rank.eq(1), "reasons"] = (
            "Skenario sintetis dengan skor simulasi tertinggi; bukan hasil penilaian model Norton."
        )
        data["supporting_evidence"] = json.dumps({
            "synthetic": True, "simulation_seed": int(seed), "score_origin": "simulated_not_trained",
        })
        parts.append(data)
    data = pd.concat(parts, ignore_index=True)
    agreement, thresholds, parameters = [], [], []
    for channel, group in data.groupby("channel", sort=False):
        for left, right in combinations(models, 2):
            left_column, right_column = f"{left}_percentile", f"{right}_percentile"
            a = set(group.sort_values([left_column, "log_key"], ascending=[False, True]).log_key.head(25))
            b = set(group.sort_values([right_column, "log_key"], ascending=[False, True]).log_key.head(25))
            ranks = group[[left_column, right_column]].rank(method="average")
            agreement.append({
                "channel": channel, "model_left": left, "model_right": right,
                "k_requested": 25, "k_effective": 25, "intersection_count": len(a & b),
                "top_k_overlap": len(a & b) / 25, "jaccard": len(a & b) / len(a | b),
                "spearman": float(np.corrcoef(ranks[left_column], ranks[right_column])[0, 1]),
                "score_origin": "simulated_scores",
            })
        for model in (*models, "consensus"):
            column = "consensus_score" if model == "consensus" else f"{model}_percentile"
            ordered = group.sort_values([column, "log_key"], ascending=[False, True])
            for percentage in (1, 5, 10):
                capacity = ceil(len(group) * percentage / 100)
                boundary = float(ordered[column].iloc[capacity - 1])
                thresholds.append({
                    "channel": channel, "model": model, "capacity_percentage": percentage,
                    "selected_count": capacity, "score_at_boundary": boundary,
                    "rows_tied_at_boundary": int(group[column].eq(boundary).sum()),
                    "rows_above_boundary": int(group[column].gt(boundary).sum()),
                    "selection_rule": "ordinal_capacity_with_log_key_tiebreak",
                    "score_origin": "simulated_scores",
                })
        for model in models:
            parameters.append({
                "channel": channel, "model": model, "mode": "simulated_scores",
                "simulation_seed": int(seed), "trained": False,
            })
    return data, {
        "model_agreement": pd.DataFrame(agreement),
        "model_stability": pd.DataFrame(columns=["channel", "model", "comparison", "spearman"]),
        "threshold_sensitivity": pd.DataFrame(thresholds),
        "experiment_parameters": pd.DataFrame(parameters),
    }
