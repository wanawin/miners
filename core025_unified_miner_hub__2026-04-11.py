#!/usr/bin/env python3
"""
BUILD: core025_group_target_deep_miner__2026-04-13_v3_FULLY_WIRED

✔ Full file
✔ No placeholders
✔ Grouped exports
✔ Candidate filtering
✔ Multiple downloads (no reset)
✔ Visible build marker
"""

import io
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

BUILD = "core025_group_target_deep_miner__2026-04-13_v3_FULLY_WIRED"

# -------------------------
# Helpers
# -------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def norm_seed(x):
    if pd.isna(x): return None
    s = re.sub(r"\D", "", str(x))
    return s if len(s) == 4 else None

def compute_features(seed: str):
    d = [int(x) for x in seed]
    cnt = Counter(d)
    s = sum(d)
    spread = max(d) - min(d)
    return {
        "seed_sum": s,
        "seed_spread": spread,
        "even_cnt": sum(x % 2 == 0 for x in d),
        "odd_cnt": sum(x % 2 == 1 for x in d),
        "high_cnt": sum(x >= 5 for x in d),
        "low_cnt": sum(x <= 4 for x in d),
        "unique": len(cnt),
        "has_pair": int(max(cnt.values()) >= 2),
        "parity": "".join("E" if x % 2 == 0 else "O" for x in d),
        "sorted": "".join(map(str, sorted(d))),
    }

def build_features(df):
    feats = df["PrevSeed"].apply(compute_features)
    return pd.concat([df, feats.apply(pd.Series)], axis=1)

def build_trait_matrix(df):
    cols = {}
    for c in df.columns:
        if c.startswith("seed_") or c in ["parity", "sorted"]:
            if df[c].dtype == object:
                for v in df[c].unique():
                    cols[f"{c}=={v}"] = (df[c] == v)
            else:
                vals = df[c]
                for q in [0.25, 0.5, 0.75]:
                    thresh = vals.quantile(q)
                    cols[f"{c}>={int(thresh)}"] = vals >= thresh
    return pd.DataFrame(cols).astype(bool)

def score_traits(traits, y, target):
    t = (y == target).astype(int)
    rows = []
    base = t.mean()
    for col in traits.columns:
        mask = traits[col]
        support = mask.sum()
        if support < 2: continue
        hit_true = t[mask].mean()
        hit_false = t[~mask].mean()
        rows.append({
            "target": target,
            "trait": col,
            "support": int(support),
            "hit_rate_true": hit_true,
            "hit_rate_false": hit_false,
            "gap": hit_true - hit_false,
            "lift": (hit_true / base) if base > 0 else 0
        })
    return pd.DataFrame(rows).sort_values(["gap","support"], ascending=[False,False])

# -------------------------
# UI
# -------------------------
st.set_page_config(layout="wide")
st.title("Core Group Target Deep Miner")
st.caption(f"BUILD: {BUILD}")

uploaded = st.file_uploader("Upload grouped seed-event CSV")

target_mode = st.selectbox("Target mode", ["OutcomeGroup","WinningMember"])
mine_level = st.selectbox("Mine level", ["basic","expanded"])

st.subheader("Candidate filters")
min_support = st.number_input("Min support", 1, 1000, 5)
min_gap = st.number_input("Min gap", 0.0, 1.0, 0.05)
min_lift = st.number_input("Min lift", 0.0, 10.0, 1.1)

if uploaded:
    df = pd.read_csv(uploaded)
    df["PrevSeed"] = df["PrevSeed"].apply(norm_seed)
    df = df.dropna(subset=["PrevSeed"])

    st.write(f"Usable rows: {len(df)}")

    if target_mode == "OutcomeGroup":
        df = df[df["OutcomeGroup"].notna()]

    df = build_features(df)
    traits = build_trait_matrix(df)

    target_col = "OutcomeGroup" if target_mode == "OutcomeGroup" else "WinningMember"
    targets = sorted(df[target_col].dropna().unique())

    results = {}
    filtered = {}
    separators = {}

    for t in targets:
        scored = score_traits(traits, df[target_col], t)
        results[t] = scored

        filt = scored[
            (scored["support"] >= min_support) &
            (scored["gap"] >= min_gap) &
            (scored["lift"] >= min_lift)
        ]
        filtered[t] = filt

        sep = scored[scored["hit_rate_false"] == 0]
        separators[t] = sep

    st.success("Mining complete")

    # -------------------------
    # DOWNLOADS (FULLY WIRED)
    # -------------------------
    for t in targets:
        st.subheader(f"{t} Outputs")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.download_button(
                f"{t} rows",
                df_to_csv_bytes(df[df[target_col] == t]),
                file_name=f"{t}__rows__{BUILD}.csv"
            )

        with col2:
            st.download_button(
                f"{t} single traits",
                df_to_csv_bytes(results[t]),
                file_name=f"{t}__single_traits__{BUILD}.csv"
            )

        with col3:
            st.download_button(
                f"{t} filtered candidates",
                df_to_csv_bytes(filtered[t]),
                file_name=f"{t}__filtered_candidates__{BUILD}.csv"
            )

        with col4:
            st.download_button(
                f"{t} separators",
                df_to_csv_bytes(separators[t]),
                file_name=f"{t}__separator_traits__{BUILD}.csv"
            )

    st.divider()

    st.subheader("Global downloads")

    st.download_button(
        "All traits",
        df_to_csv_bytes(pd.concat(results.values())),
        file_name=f"ALL__traits__{BUILD}.csv"
    )

    st.download_button(
        "Feature table",
        df_to_csv_bytes(df),
        file_name=f"feature_table__{BUILD}.csv"
    )
