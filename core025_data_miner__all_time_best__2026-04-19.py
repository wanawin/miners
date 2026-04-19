#!/usr/bin/env python3
# BUILD: core025_data_miner__final_v6__2026-04-19

import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Core025 Miner v6", layout="wide")
st.title("🛠 Core025 Data Miner - Final v6 (Bulletproof)")
st.caption("BUILD: core025_data_miner__final_v6__2026-04-19")

history_file = st.file_uploader("Full Raw History txt", type=["txt"], key="hist")
prepared_file = st.file_uploader("prepared_training_rows__core025_precompute_builder__2026-04-16_v1.csv", type="csv", key="prep")

if not history_file or not prepared_file:
    st.info("Upload both files.")
    st.stop()

# Load history
history_df = pd.read_csv(history_file, sep='\t', header=None, engine='python', on_bad_lines='skip')
history_df.columns = ['date_text'] + [f'col_{i}' for i in range(1, history_df.shape[1])]
history_df["date"] = pd.to_datetime(history_df["date_text"], errors="coerce", format="%a, %b %d, %Y")

def extract_win(text):
    if pd.isna(text): return None
    s = str(text).replace("-", "").replace(",", "").strip()
    digits = ''.join(filter(str.isdigit, s))
    return digits[:4] if len(digits) >= 4 else None

history_df["winning_4digit"] = history_df.iloc[:, 3].apply(extract_win)
history_df = history_df.dropna(subset=["date", "winning_4digit"]).drop_duplicates(subset=["date"])

# Load prepared
prepared_df = pd.read_csv(prepared_file)
if "PlayDate" in prepared_df.columns:
    prepared_df["date"] = pd.to_datetime(prepared_df["PlayDate"], errors="coerce")
prepared_df = prepared_df.sort_values("date").reset_index(drop=True)

# Merge
merged = prepared_df.merge(history_df[["date", "winning_4digit"]], on="date", how="left")

def normalize_win(x):
    if pd.isna(x) or str(x).strip() == "": return ""
    s = str(x).strip()
    if s in ["25", "0025"]: return "0025"
    if s in ["225", "0225"]: return "0225"
    if s in ["255", "0255"]: return "0255"
    return s.zfill(4) if s.isdigit() else ""

merged["TrueMember"] = merged["winning_4digit"].apply(normalize_win)

# Bulletproof stream stats
if "StreamKey" in merged.columns:
    stream_stats = merged.groupby("StreamKey").agg(
        total_plays=("TrueMember", "count"),
        top1_hits=("TrueMember", lambda x: (x == x.mode()[0]).sum() if not x.mode().empty and len(x) > 0 else 0)
    ).reset_index()
    
    # Zero-division safe
    stream_stats["hit_density"] = np.where(
        stream_stats["total_plays"] > 0,
        stream_stats["top1_hits"].astype(float) / stream_stats["total_plays"].astype(float),
        0.0
    )
    merged = merged.merge(stream_stats[["StreamKey", "hit_density", "total_plays"]], on="StreamKey", how="left")
else:
    merged["hit_density"] = 0.0
    merged["total_plays"] = 1

merged["hit_density"] = merged["hit_density"].fillna(0.0).round(4)

st.success(f"✅ Merged {len(merged)} rows!")
st.write("TrueMember distribution:", merged["TrueMember"].value_counts().to_dict())

if st.button("Generate v6 File"):
    output = "prepared_full_truth_with_stream_stats_v6.csv"
    merged.to_csv(output, index=False)
    with open(output, "rb") as f:
        st.download_button(
            "📥 Download v6 file (real TrueMember + safe stats)",
            f.read(),
            output,
            "text/csv",
            key="v6_download"
        )
    st.balloons()
    st.success("v6 ready — use this in the walk-forward app.")
