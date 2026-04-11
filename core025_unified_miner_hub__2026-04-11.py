
#!/usr/bin/env python3
"""
core025_unified_miner_hub__2026-04-11.py

Unified miner hub for Core025-style work.

Purpose
-------
One Streamlit app that combines:
1) Quick formatter for per-event / miss / waste / needed files into a standard event-CSV style
2) Deep trait mining on seed-event CSVs
3) Member / Top2-needed / skip-danger mining from full history
4) Pairwise separator mining from full history

This app uses the real uploaded miner code when available.
No placeholders. No simulations.
"""

from __future__ import annotations

import io
import os
import sys
import types
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st

APP_BUILD = "core025_unified_miner_hub__2026-04-11"

# ---------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
SEARCH_DIRS = [ROOT_DIR, Path("/mnt/data")]

REQUIRED_FILES = {
    "deep": "core025_deep_trait_miner_streamlit_ready_v4__2026-03-24.py",
    "member": "core025_member_trait_miner_v2_strict_fix1__2026-03-28.py",
    "pairwise": "core025_pairwise_separator_miner_v1__2026-03-28.py",
}


def find_file(name: str) -> Optional[Path]:
    for d in SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource
def load_real_modules() -> Dict[str, Any]:
    mods: Dict[str, Any] = {}

    # deep miner imports without streamlit already
    deep_path = find_file(REQUIRED_FILES["deep"])
    if deep_path is None:
        raise FileNotFoundError(f"Missing required file: {REQUIRED_FILES['deep']}")
    mods["deep"] = load_module_from_path(deep_path, "core025_deep_trait_miner_v4")

    # member/pairwise miner import streamlit; in a streamlit app that's fine
    member_path = find_file(REQUIRED_FILES["member"])
    if member_path is None:
        raise FileNotFoundError(f"Missing required file: {REQUIRED_FILES['member']}")
    mods["member"] = load_module_from_path(member_path, "core025_member_trait_miner_v2")

    pair_path = find_file(REQUIRED_FILES["pairwise"])
    if pair_path is None:
        raise FileNotFoundError(f"Missing required file: {REQUIRED_FILES['pairwise']}")
    mods["pairwise"] = load_module_from_path(pair_path, "core025_pairwise_separator_miner_v1")
    return mods


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = uploaded_file.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t")
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def standardize_event_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a per-event-like file into the shared miner-ready format:
    PrevSeed, WinningMember, PlayDate, StreamKey, plus pass-through fields.
    """
    cols = {str(c).lower(): c for c in df_raw.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    seed_col = pick("PrevSeed", "seed", "seed_result", "previous_result", "prev_result", "prior_result")
    member_col = pick("WinningMember", "winning_member", "winner_member", "true_member", "member", "actual_member", "result_member", "target_member")
    date_col = pick("PlayDate", "transition_date", "event_date", "date", "draw date", "play_date", "target_date")
    stream_col = pick("StreamKey", "stream", "stream_id", "state_game", "streamkey")

    if seed_col is None:
        raise ValueError("Could not find a seed column. Expected one of PrevSeed/seed/previous_result/etc.")
    if member_col is None:
        raise ValueError("Could not find a winning member column. Expected one of WinningMember/winning_member/member/etc.")

    out = pd.DataFrame()
    out["PrevSeed"] = df_raw[seed_col].astype(str).str.extract(r'(\d+)')[0].str.zfill(4)
    out["WinningMember"] = df_raw[member_col].astype(str).str.extract(r'(\d+)')[0].str.zfill(4)
    out["PlayDate"] = pd.to_datetime(df_raw[date_col], errors="coerce").dt.strftime("%Y-%m-%d") if date_col else ""
    out["StreamKey"] = df_raw[stream_col].astype(str) if stream_col else ""

    preserve = [c for c in df_raw.columns if c not in {seed_col, member_col, date_col, stream_col}]
    for c in preserve:
        out[c] = df_raw[c]
    out = out.dropna(subset=["PrevSeed", "WinningMember"]).copy()
    out = out[(out["PrevSeed"].str.len() == 4) & (out["WinningMember"].str.len().isin([2, 3, 4]))].copy()
    out["WinningMember"] = out["WinningMember"].str.zfill(4)
    return out.reset_index(drop=True)


def preview_download(df: pd.DataFrame, file_name: str, label: str, rows: int = 25) -> None:
    st.dataframe(df.head(rows), use_container_width=True)
    st.download_button(label, data=df_to_csv_bytes(df), file_name=file_name, mime="text/csv")


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

def tab_formatter():
    st.subheader("Quick formatter")
    st.caption("Convert per-event / miss / waste / needed files into the standard seed-event CSV style used by the deep miner.")
    upl = st.file_uploader("Upload file to format", key="formatter_upload")
    if not upl:
        st.info("Upload a CSV/TXT/XLSX file to convert it into the miner-ready format.")
        return
    try:
        df_raw = read_uploaded_table(upl)
        out = standardize_event_csv(df_raw)
        st.success(f"Created standardized file with {len(out):,} usable rows.")
        preview_download(
            out,
            file_name=f"{Path(upl.name).stem}__miner_ready__{APP_BUILD}.csv",
            label="Download standardized miner-ready CSV",
            rows=20,
        )
    except Exception as e:
        st.exception(e)


def tab_deep_miner(mods: Dict[str, Any]):
    deep = mods["deep"]
    st.subheader("Deep trait miner")
    st.caption("Run the real deep trait miner on a seed-event CSV. Recommended headers: PrevSeed, WinningMember, PlayDate, StreamKey.")

    c1, c2, c3, c4 = st.columns(4)
    min_support = c1.number_input("Separator min support", min_value=3, value=8, step=1, key="deep_min_support")
    bucket_min_support = c2.number_input("Bucket min support", min_value=3, value=6, step=1, key="deep_bucket_min_support")
    bucket_top_k = c3.number_input("Bucket top K traits", min_value=10, value=80, step=5, key="deep_bucket_top_k")
    bucket_max_depth = c4.number_input("Bucket max depth", min_value=2, value=4, step=1, key="deep_bucket_max_depth")

    c5, c6 = st.columns(2)
    mine_level = c5.selectbox("Mine level", ["standard", "expanded"], index=1, key="deep_mine_level")
    objective = c6.selectbox(
        "Objective selector",
        ["positive_buckets", "separators", "member_one_vs_rest"],
        index=0,
        key="deep_objective",
    )

    upl = st.file_uploader("Upload seed-event CSV", key="deep_upload")
    if not upl:
        st.info("Upload a seed-event CSV to run the deep miner.")
        return

    try:
        df_raw = read_uploaded_table(upl)
        if not any(str(c).lower() in {"prevseed", "winningmember", "playdate", "streamkey"} for c in df_raw.columns):
            df_raw = standardize_event_csv(df_raw)
        if st.button("Run Deep Trait Miner", type="primary", key="run_deep"):
            with st.spinner("Running deep miner..."):
                results = deep.run_mining(
                    df_raw=df_raw,
                    min_support=int(min_support),
                    bucket_min_support=int(bucket_min_support),
                    bucket_top_k=int(bucket_top_k),
                    bucket_max_depth=int(bucket_max_depth),
                    mine_level=str(mine_level),
                    pass_label="Unified App Pass 1",
                    objective=str(objective),
                )

            st.success("Deep miner finished.")
            st.text_area("Summary", results["summary_text"], height=320)

            events = results["events"]
            all_scores = results["all_scores"]
            st.markdown("### Feature table")
            preview_download(events, f"feature_table__{APP_BUILD}.csv", "Download feature table CSV")
            st.markdown("### All-member trait scores")
            preview_download(all_scores, f"trait_scores_all_members__{APP_BUILD}.csv", "Download all-member trait scores CSV")

            for member in deep.MEMBERS:
                st.markdown(f"### Member {deep.member_label(member)}")
                t1, t2, t3 = st.tabs(["Top single traits", "Separator traits", "Stacked buckets"])
                with t1:
                    preview_download(
                        results["per_member_scores"][member],
                        f"{deep.member_label(member)}__single_traits__{APP_BUILD}.csv",
                        f"Download {deep.member_label(member)} single traits CSV",
                    )
                with t2:
                    preview_download(
                        results["per_member_separators"][member],
                        f"{deep.member_label(member)}__separator_traits__{APP_BUILD}.csv",
                        f"Download {deep.member_label(member)} separator traits CSV",
                    )
                with t3:
                    preview_download(
                        results["per_member_buckets"][member],
                        f"{deep.member_label(member)}__stacked_buckets__{APP_BUILD}.csv",
                        f"Download {deep.member_label(member)} stacked buckets CSV",
                    )
    except Exception as e:
        st.exception(e)


def tab_history_miners(mods: Dict[str, Any]):
    mem = mods["member"]
    pair = mods["pairwise"]

    st.subheader("History-based member + pairwise miners")
    st.caption("Run the real full-history miners on a full history file.")

    with st.expander("Controls", expanded=True):
        c1, c2, c3 = st.columns(3)
        min_support = c1.number_input("Minimum trait support", min_value=3, value=8, step=1, key="hist_min_support")
        preferred_support = c2.number_input("Preferred support", min_value=3, value=12, step=1, key="hist_pref_support")
        min_streams = c3.number_input("Minimum streams", min_value=1, value=5, step=1, key="hist_min_streams")

        c4, c5, c6 = st.columns(3)
        min_months = c4.number_input("Minimum months", min_value=1, value=3, step=1, key="hist_min_months")
        min_dom_rate = c5.slider("Minimum dominance rate", min_value=0.34, max_value=0.95, value=0.60, step=0.01, key="hist_min_dom_rate")
        min_gap = c6.slider("Minimum gap", min_value=0.00, max_value=1.50, value=0.40, step=0.01, key="hist_min_gap")

        c7, c8, c9, c10 = st.columns(4)
        min_member_rate = c7.slider("Minimum member rate", min_value=0.34, max_value=0.95, value=0.60, step=0.01, key="hist_min_member_rate")
        min_top2_rate = c8.slider("Minimum Top2-needed rate", min_value=0.05, max_value=0.95, value=0.30, step=0.01, key="hist_min_top2_rate")
        min_skip_danger_rate = c9.slider("Minimum skip-danger rate", min_value=0.05, max_value=0.95, value=0.30, step=0.01, key="hist_min_skip_rate")
        min_stream_history = c10.number_input("Minimum stream-specific history", min_value=0, value=20, step=5, key="hist_min_stream_history")

        c11, c12 = st.columns(2)
        top1_only_score_threshold = c11.slider("Top1-only score threshold", min_value=0.33, max_value=0.95, value=0.55, step=0.005, key="hist_top1_only")
        top1_top2_score_threshold = c12.slider("Top1+Top2 score threshold", min_value=0.33, max_value=0.95, value=0.42, step=0.005, key="hist_top1_top2")

    upl = st.file_uploader("Upload FULL history file", key="history_upload")
    if not upl:
        st.info("Upload the full history file to run the history-based miners.")
        return

    try:
        hist = mem.prepare_history(mem.load_table(upl))
        transitions = mem.build_transitions(hist)
        core_hits = transitions[transitions["is_core025_hit"] == 1].copy()

        a, b, c = st.columns(3)
        a.metric("Transitions", f"{len(transitions):,}")
        b.metric("Core025 hit events", f"{len(core_hits):,}")
        c.metric("Core025 base rate", f"{transitions['is_core025_hit'].mean():.4f}")

        if st.button("Run History Miners", type="primary", key="run_hist_miners"):
            with st.spinner("Running member trait miner and pairwise separator miner..."):
                sep_traits = mem.build_member_separation_traits(
                    core_hits, int(min_support), int(preferred_support),
                    float(min_dom_rate), float(min_gap), int(min_streams), int(min_months)
                )
                traits_0025 = mem.build_member_specific_traits(
                    core_hits, int(min_support), int(preferred_support),
                    "0025", float(min_member_rate), int(min_streams), int(min_months)
                )
                traits_0225 = mem.build_member_specific_traits(
                    core_hits, int(min_support), int(preferred_support),
                    "0225", float(min_member_rate), int(min_streams), int(min_months)
                )
                traits_0255 = mem.build_member_specific_traits(
                    core_hits, int(min_support), int(preferred_support),
                    "0255", float(min_member_rate), int(min_streams), int(min_months)
                )
                pred_hits = mem.build_hit_event_predictions(
                    transitions=transitions,
                    min_global_history=100,
                    min_stream_history=int(min_stream_history),
                    top1_only_score_threshold=float(top1_only_score_threshold),
                    top1_top2_score_threshold=float(top1_top2_score_threshold),
                )
                top2_needed = mem.build_top2_needed_traits(
                    pred_hits, transitions, int(min_support), int(preferred_support),
                    float(min_top2_rate), int(min_streams), int(min_months)
                )
                skip_danger = mem.build_skip_danger_traits(
                    pred_hits, transitions, int(min_support), int(preferred_support),
                    float(min_skip_danger_rate), int(min_streams), int(min_months)
                )
                pair_0025_0225 = pair.build_pairwise_separator_traits(
                    core_hits, "0025", "0225", int(min_support), int(preferred_support),
                    float(min_dom_rate), float(min_gap), int(min_streams), int(min_months)
                )
                pair_0025_0255 = pair.build_pairwise_separator_traits(
                    core_hits, "0025", "0255", int(min_support), int(preferred_support),
                    float(min_dom_rate), float(min_gap), int(min_streams), int(min_months)
                )
                pair_0225_0255 = pair.build_pairwise_separator_traits(
                    core_hits, "0225", "0255", int(min_support), int(preferred_support),
                    float(min_dom_rate), float(min_gap), int(min_streams), int(min_months)
                )

            st.success("History miners finished.")

            tabs = st.tabs([
                "Separation traits", "0025 traits", "0225 traits", "0255 traits",
                "Top2-needed", "Skip-danger", "Pair 0025 vs 0225",
                "Pair 0025 vs 0255", "Pair 0225 vs 0255"
            ])

            outputs = [
                ("separation_traits", sep_traits),
                ("traits_0025", traits_0025),
                ("traits_0225", traits_0225),
                ("traits_0255", traits_0255),
                ("top2_needed", top2_needed),
                ("skip_danger", skip_danger),
                ("pair_0025_vs_0225", pair_0025_0225),
                ("pair_0025_vs_0255", pair_0025_0255),
                ("pair_0225_vs_0255", pair_0225_0255),
            ]
            for tab, (name, df) in zip(tabs, outputs):
                with tab:
                    preview_download(df, f"{name}__{APP_BUILD}.csv", f"Download {name} CSV")
    except Exception as e:
        st.exception(e)


def main():
    st.set_page_config(page_title="Core025 Unified Miner Hub", layout="wide")
    st.title("Core025 Unified Miner Hub")
    st.caption("One app to standardize seed-event CSVs and run the real deep, member, and pairwise miners.")
    st.markdown(f"**BUILD:** `{APP_BUILD}`")

    try:
        mods = load_real_modules()
    except Exception as e:
        st.exception(e)
        st.stop()

    t1, t2, t3 = st.tabs(["Formatter", "Deep Trait Miner", "History Miners"])
    with t1:
        tab_formatter()
    with t2:
        tab_deep_miner(mods)
    with t3:
        tab_history_miners(mods)


if __name__ == "__main__":
    main()
