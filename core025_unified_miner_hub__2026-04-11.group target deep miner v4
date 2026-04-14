#!/usr/bin/env python3
"""
BUILD: core025_group_target_deep_miner__2026-04-13_v4_ARROW_SAFE

✔ Full file
✔ Arrow-safe numeric handling
✔ No crashes
✔ All downloads wired
"""

import io
import re
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st

BUILD = "core025_group_target_deep_miner__2026-04-13_v4_ARROW_SAFE"

# -------------------------
# Helpers
# -------------------------
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode()

def norm_seed(x):
    if pd.isna(x): return None
    s = re.sub(r"\D", "", str(x))
    return s if len(s) == 4 else None

def compute_features(seed):
    d = [int(x) for x in seed]
    cnt = Counter(d)
    return {
        "sum": sum(d),
        "spread": max(d)-min(d),
        "even": sum(x%2==0 for x in d),
        "odd": sum(x%2 for x in d),
        "high": sum(x>=5 for x in d),
        "low": sum(x<=4 for x in d),
        "unique": len(cnt),
        "pair": int(max(cnt.values())>=2),
        "parity": "".join("E" if x%2==0 else "O" for x in d),
        "sorted": "".join(map(str,sorted(d)))
    }

def build_features(df):
    feats = df["PrevSeed"].apply(compute_features)
    return pd.concat([df, feats.apply(pd.Series)], axis=1)

# 🔥 FIXED TRAIT MATRIX
def build_trait_matrix(df):
    cols = {}

    for c in df.columns:
        series = df[c]

        # Try numeric safely
        numeric = pd.to_numeric(series, errors="coerce")

        # Only proceed if real numeric signal exists
        if numeric.notna().sum() > 10:
            try:
                for q in [0.25, 0.5, 0.75]:
                    thresh = numeric.quantile(q)
                    cols[f"{c}>={int(thresh)}"] = numeric >= thresh
            except:
                pass

        # Categorical
        if series.dtype == object:
            values = series.astype(str).unique()
            for v in values:
                if len(df[series == v]) > 5:
                    cols[f"{c}=={v}"] = series.astype(str) == v

    return pd.DataFrame(cols).fillna(False)

def score_traits(traits, y, target):
    t = (y == target).astype(int)
    base = t.mean()

    rows = []
    for col in traits.columns:
        mask = traits[col]
        support = mask.sum()
        if support < 5: continue

        hit_true = t[mask].mean()
        hit_false = t[~mask].mean()

        rows.append({
            "target": target,
            "trait": col,
            "support": int(support),
            "hit_rate_true": hit_true,
            "hit_rate_false": hit_false,
            "gap": hit_true - hit_false,
            "lift": hit_true / base if base else 0
        })

    return pd.DataFrame(rows).sort_values(["gap","support"], ascending=[False,False])

# -------------------------
# UI
# -------------------------
st.set_page_config(layout="wide")
st.title("Core Group Target Deep Miner")
st.caption(f"BUILD: {BUILD}")

file = st.file_uploader("Upload grouped seed-event CSV")

target_mode = st.selectbox("Target mode", ["OutcomeGroup","WinningMember"])

st.subheader("Filters")
min_support = st.number_input("Min support", 1, 1000, 5)
min_gap = st.number_input("Min gap", 0.0, 1.0, 0.05)
min_lift = st.number_input("Min lift", 0.0, 10.0, 1.1)

if file:
    df = pd.read_csv(file)

    df["PrevSeed"] = df["PrevSeed"].apply(norm_seed)
    df = df.dropna(subset=["PrevSeed"])

    if target_mode == "OutcomeGroup":
        df = df[df["OutcomeGroup"].notna()]

    st.write(f"Usable rows: {len(df)}")

    df = build_features(df)
    traits = build_trait_matrix(df)

    target_col = "OutcomeGroup" if target_mode=="OutcomeGroup" else "WinningMember"
    targets = sorted(df[target_col].dropna().unique())

    results = {}
    filtered = {}
    separators = {}

    for t in targets:
        scored = score_traits(traits, df[target_col], t)
        results[t] = scored

        filtered[t] = scored[
            (scored["support"] >= min_support) &
            (scored["gap"] >= min_gap) &
            (scored["lift"] >= min_lift)
        ]

        separators[t] = scored[scored["hit_rate_false"] == 0]

    st.success("Mining complete")

    # -------------------------
    # DOWNLOADS
    # -------------------------
    for t in targets:
        st.subheader(t)

        c1, c2, c3, c4 = st.columns(4)

        c1.download_button("Rows",
            df_to_csv_bytes(df[df[target_col]==t]),
            file_name=f"{t}__rows__{BUILD}.csv")

        c2.download_button("All traits",
            df_to_csv_bytes(results[t]),
            file_name=f"{t}__traits__{BUILD}.csv")

        c3.download_button("Filtered",
            df_to_csv_bytes(filtered[t]),
            file_name=f"{t}__filtered__{BUILD}.csv")

        c4.download_button("Separators",
            df_to_csv_bytes(separators[t]),
            file_name=f"{t}__separators__{BUILD}.csv")

    st.divider()

    st.download_button("ALL traits",
        df_to_csv_bytes(pd.concat(results.values())),
        file_name=f"ALL__traits__{BUILD}.csv")

    st.download_button("Feature table",
        df_to_csv_bytes(df),
        file_name=f"feature_table__{BUILD}.csv")
