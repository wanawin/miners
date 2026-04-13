
#!/usr/bin/env python3
"""
core025_group_target_deep_miner__2026-04-12.py

Streamlit app for mining seed traits against either:
1) WinningMember targets (0025 / 0225 / 0255), or
2) OutcomeGroup targets (WASTE / NEEDED / MISS)
"""
from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_group_target_deep_miner__2026-04-12"
DIGITS = list(range(10))
MIRROR_PAIRS = {(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)}

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def safe_display_df(df: pd.DataFrame, rows: int = 100) -> pd.DataFrame:
    return df.head(int(rows)).copy()

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)
    nmap = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in nmap:
            return nmap[key]
    for cand in candidates:
        key = _norm(cand)
        for k, c in nmap.items():
            if key and key in k:
                return c
    if required:
        raise KeyError(f"Required column not found. Tried {list(candidates)}. Available columns: {cols}")
    return None

def load_table(uploaded_file) -> pd.DataFrame:
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

def canonical_seed(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = re.sub(r"\D", "", str(x))
    return s if len(s) == 4 else None

def coerce_member_text(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    nums = re.findall(r"\d+", s)
    if nums:
        for token in reversed(nums):
            v = token.zfill(4)
            if v in {"0025", "0225", "0255"}:
                return v
            if token in {"25", "225", "255"}:
                return {"25": "0025", "225": "0225", "255": "0255"}[token]
    s_up = s.upper()
    return s_up if s_up in {"0025", "0225", "0255"} else None

def coerce_outcome_group(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    aliases = {
        "WASTE": "WASTE",
        "WASTE_TOP2": "WASTE",
        "NEEDED": "NEEDED",
        "NEEDED_TOP2": "NEEDED",
        "MISS": "MISS",
        "MISSED": "MISS",
    }
    return aliases.get(s, s if s else None)

def digit_list(seed: str) -> List[int]:
    return [int(ch) for ch in seed]

def as_pair_tokens(seed: str) -> List[str]:
    ds = list(seed)
    out = []
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            out.append("".join(sorted((ds[i], ds[j]))))
    return out

def as_ordered_adj_pairs(seed: str) -> List[str]:
    return [seed[i:i + 2] for i in range(len(seed) - 1)]

def as_unordered_adj_pairs(seed: str) -> List[str]:
    return ["".join(sorted(seed[i:i + 2])) for i in range(len(seed) - 1)]

def compute_features(seed: str) -> Dict[str, object]:
    d = digit_list(seed)
    cnt = Counter(d)
    s = sum(d)
    spread = max(d) - min(d)
    parity = "".join("E" if x % 2 == 0 else "O" for x in d)
    highlow = "".join("H" if x >= 5 else "L" for x in d)
    ordered_adj = as_ordered_adj_pairs(seed)
    consec_links = 0
    unique_sorted = sorted(set(d))
    for a, b in zip(unique_sorted[:-1], unique_sorted[1:]):
        if b - a == 1:
            consec_links += 1
    mirrorpair_cnt = sum(1 for a, b in MIRROR_PAIRS if a in cnt and b in cnt)
    pairwise_absdiff = [abs(d[i] - d[j]) for i in range(4) for j in range(i + 1, 4)]
    adj_absdiff = [abs(d[i] - d[i + 1]) for i in range(3)]
    features: Dict[str, object] = {
        "seed_sum": s,
        "seed_sum_lastdigit": s % 10,
        "seed_sum_mod3": s % 3,
        "seed_sum_mod4": s % 4,
        "seed_sum_mod5": s % 5,
        "seed_spread": spread,
        "seed_unique_digits": len(cnt),
        "seed_has_pair": int(max(cnt.values()) >= 2),
        "seed_no_pair": int(max(cnt.values()) == 1),
        "seed_has_trip": int(max(cnt.values()) >= 3),
        "seed_has_quad": int(max(cnt.values()) >= 4),
        "seed_even_cnt": int(sum(x % 2 == 0 for x in d)),
        "seed_odd_cnt": int(sum(x % 2 == 1 for x in d)),
        "seed_high_cnt": int(sum(x >= 5 for x in d)),
        "seed_low_cnt": int(sum(x <= 4 for x in d)),
        "seed_consec_links": consec_links,
        "seed_mirrorpair_cnt": mirrorpair_cnt,
        "seed_pairwise_absdiff_sum": int(sum(pairwise_absdiff)),
        "seed_pairwise_absdiff_max": int(max(pairwise_absdiff)),
        "seed_pairwise_absdiff_min": int(min(pairwise_absdiff)),
        "seed_adj_absdiff_sum": int(sum(adj_absdiff)),
        "seed_adj_absdiff_max": int(max(adj_absdiff)),
        "seed_adj_absdiff_min": int(min(adj_absdiff)),
        "seed_pos1": d[0], "seed_pos2": d[1], "seed_pos3": d[2], "seed_pos4": d[3],
        "seed_first_last_sum": d[0] + d[3], "seed_middle_sum": d[1] + d[2],
        "seed_absdiff_outer_inner": abs((d[0] + d[3]) - (d[1] + d[2])),
        "seed_parity_pattern": parity, "seed_highlow_pattern": highlow,
        "seed_sorted": "".join(map(str, sorted(d))),
        "seed_pair_tokens": "|".join(sorted(as_pair_tokens(seed))),
        "seed_adj_pairs_ordered": "|".join(ordered_adj),
        "seed_adj_pairs_unordered": "|".join(sorted(as_unordered_adj_pairs(seed))),
        "seed_outer_equal": int(d[0] == d[3]),
        "seed_inner_equal": int(d[1] == d[2]),
        "seed_palindrome_like": int(d[0] == d[3] and d[1] == d[2]),
        "seed_same_adjacent_count": int(sum(d[i] == d[i + 1] for i in range(3))),
        "seed_pos1_lt_pos2": int(d[0] < d[1]), "seed_pos2_lt_pos3": int(d[1] < d[2]),
        "seed_pos3_lt_pos4": int(d[2] < d[3]), "seed_pos1_lt_pos3": int(d[0] < d[2]),
        "seed_pos2_lt_pos4": int(d[1] < d[3]), "seed_pos1_eq_pos2": int(d[0] == d[1]),
        "seed_pos2_eq_pos3": int(d[1] == d[2]), "seed_pos3_eq_pos4": int(d[2] == d[3]),
        "seed_outer_gt_inner": int((d[0] + d[3]) > (d[1] + d[2])),
        "seed_sum_even": int(s % 2 == 0), "seed_sum_high_20plus": int(s >= 20),
    }
    for k in DIGITS:
        features[f"seed_has{k}"] = int(k in cnt)
        features[f"seed_cnt{k}"] = int(cnt.get(k, 0))
    shape = "".join(map(str, sorted(cnt.values(), reverse=True)))
    features["seed_repeat_shape"] = {
        "1111": "all_unique",
        "211": "one_pair",
        "22": "two_pair",
        "31": "trip",
        "4": "quad",
    }.get(shape, f"shape_{shape}")
    features["cnt_0_3"] = int(sum(0 <= x <= 3 for x in d))
    features["cnt_4_6"] = int(sum(4 <= x <= 6 for x in d))
    features["cnt_7_9"] = int(sum(7 <= x <= 9 for x in d))
    pair_counts = Counter(as_pair_tokens(seed))
    for a in range(10):
        for b in range(a, 10):
            tok = f"{a}{b}"
            features[f"pair_has_{tok}"] = int(pair_counts.get(tok, 0) > 0)
    for a in range(10):
        for b in range(10):
            tok = f"{a}{b}"
            features[f"adj_ord_has_{tok}"] = int(tok in ordered_adj)
    return features

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: Dict[str, int] = {}
    new_cols: List[str] = []
    for col in df.columns:
        name = str(col)
        if name not in seen:
            seen[name] = 0
            new_cols.append(name)
        else:
            seen[name] += 1
            new_cols.append(f"{name}__dup{seen[name]}")
    out = df.copy()
    out.columns = new_cols
    return out

def bin_numeric_series(s: pd.Series, prefix: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().sum() == 0:
        return out
    uniques = sorted(set(vals.dropna().astype(float).tolist()))
    if len(uniques) <= 20:
        for u in uniques:
            label = str(int(u)) if float(u).is_integer() else str(u)
            out[f"{prefix}=={label}"] = vals == u
    qs = [0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9]
    quantiles = sorted(set(float(vals.quantile(q)) for q in qs if vals.notna().sum() >= 10))
    for q in quantiles:
        if np.isnan(q):
            continue
        label = int(q) if float(q).is_integer() else round(float(q), 3)
        out[f"{prefix}<={label}"] = vals <= q
        out[f"{prefix}>={label}"] = vals >= q
    return out

def categorical_series_traits(s: pd.Series, prefix: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    vc = s.astype(str).value_counts(dropna=False)
    for v, n in vc.items():
        if n >= 2:
            out[f"{prefix}=={v}"] = s.astype(str) == str(v)
    return out

def add_expanded_interactions(df_feat: pd.DataFrame, trait_cols: Dict[str, pd.Series]) -> None:
    if "seed_repeat_shape" in df_feat.columns and "seed_parity_pattern" in df_feat.columns:
        combo = df_feat["seed_repeat_shape"].astype(str) + "|" + df_feat["seed_parity_pattern"].astype(str)
        trait_cols.update(categorical_series_traits(combo, "x_repeatshape_parity"))
    if "seed_repeat_shape" in df_feat.columns and "seed_highlow_pattern" in df_feat.columns:
        combo = df_feat["seed_repeat_shape"].astype(str) + "|" + df_feat["seed_highlow_pattern"].astype(str)
        trait_cols.update(categorical_series_traits(combo, "x_repeatshape_highlow"))
    if "seed_unique_digits" in df_feat.columns and "seed_even_cnt" in df_feat.columns:
        combo = df_feat["seed_unique_digits"].astype(str) + "|" + df_feat["seed_even_cnt"].astype(str)
        trait_cols.update(categorical_series_traits(combo, "x_unique_even"))
    pairs = [("seed_has0","seed_has9"),("seed_outer_equal","seed_inner_equal"),("seed_sum_even","seed_has_pair")]
    for left, right in pairs:
        if left in trait_cols and right in trait_cols:
            trait_cols[f"{left} AND {right}"] = trait_cols[left] & trait_cols[right]

def build_trait_matrix(df_feat: pd.DataFrame, mine_level: str = "expanded") -> pd.DataFrame:
    trait_cols: Dict[str, pd.Series] = {}
    numeric_cols = [c for c in df_feat.columns if c.startswith("seed_") or c.startswith("cnt_")]
    categorical_cols = ["seed_parity_pattern", "seed_highlow_pattern", "seed_repeat_shape", "seed_sorted"]
    for c in numeric_cols:
        ser = df_feat[c]
        if pd.api.types.is_numeric_dtype(ser):
            trait_cols.update(bin_numeric_series(ser, c))
    for c in categorical_cols:
        if c in df_feat.columns:
            trait_cols.update(categorical_series_traits(df_feat[c], c))
    for c in df_feat.columns:
        if c.startswith(("pair_has_", "adj_ord_has_", "seed_has", "seed_cnt")):
            ser = pd.to_numeric(df_feat[c], errors="coerce").fillna(0).astype(int)
            if ser.sum() >= 2:
                trait_cols[c] = ser.astype(bool)
    if mine_level == "expanded":
        add_expanded_interactions(df_feat, trait_cols)
    return dedupe_columns(pd.DataFrame(trait_cols, index=df_feat.index).astype(bool))

def score_traits_one_vs_rest(trait_df: pd.DataFrame, y: pd.Series, target_value: str) -> pd.DataFrame:
    target = (y.astype(str) == str(target_value)).astype(int)
    rows = []
    total_hits = int(target.sum())
    total_n = int(len(target))
    base_rate = total_hits / total_n if total_n else 0.0
    for trait in trait_df.columns:
        mask = trait_df[trait].fillna(False).astype(bool)
        support = int(mask.sum())
        if support == 0 or support == total_n:
            continue
        hits_true = int(target[mask].sum())
        non_true = int((~mask).sum())
        hits_false = int(target[~mask].sum())
        hit_rate_true = hits_true / support if support else 0.0
        hit_rate_false = hits_false / non_true if non_true else 0.0
        rows.append({
            "target_value": str(target_value),
            "trait": trait,
            "support": support,
            "hits_true": hits_true,
            "hit_rate_true": hit_rate_true,
            "hits_false": hits_false,
            "hit_rate_false": hit_rate_false,
            "gap": hit_rate_true - hit_rate_false,
            "base_rate": base_rate,
            "lift": (hit_rate_true / base_rate) if base_rate > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["gap", "hit_rate_true", "support"], ascending=[False, False, False]).reset_index(drop=True) if len(out) else out

def build_separator_traits(scores: pd.DataFrame, target_value: str, top_n: int = 50) -> pd.DataFrame:
    if len(scores) == 0:
        return pd.DataFrame()
    df = scores[scores["target_value"] == str(target_value)].copy()
    if len(df) == 0:
        return pd.DataFrame()
    sep = df[(df["hits_false"] == 0) | (df["hit_rate_false"] == 0)].copy()
    if len(sep) == 0:
        return pd.DataFrame()
    sep["separator_strength"] = sep["hit_rate_true"] - sep["hit_rate_false"]
    return sep.sort_values(["separator_strength", "support", "hit_rate_true"], ascending=[False, False, False]).head(int(top_n)).reset_index(drop=True)

def build_stacked_buckets(trait_df: pd.DataFrame, y: pd.Series, target_value: str, base_scores: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if len(base_scores) == 0:
        return pd.DataFrame()
    top_traits = base_scores.head(15)["trait"].tolist()
    target = (y.astype(str) == str(target_value)).astype(int)
    rows = []
    for i in range(len(top_traits)):
        for j in range(i + 1, len(top_traits)):
            t1, t2 = top_traits[i], top_traits[j]
            if t1 not in trait_df.columns or t2 not in trait_df.columns:
                continue
            mask = trait_df[t1] & trait_df[t2]
            support = int(mask.sum())
            if support < 2:
                continue
            hits = int(target[mask].sum())
            rows.append({"target_value": str(target_value), "stack": f"{t1} AND {t2}", "support": support, "hits": hits, "hit_rate": hits / support if support else 0.0})
    out = pd.DataFrame(rows)
    return out.sort_values(["hit_rate", "support", "hits"], ascending=[False, False, False]).head(int(top_n)).reset_index(drop=True) if len(out) else out

def prepare_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    seed_col = find_col(df_raw, ["PrevSeed", "seed", "seed_result", "previous_result", "prev_result", "prior_result"])
    member_col = find_col(df_raw, ["WinningMember", "true_member", "winning_member", "winner_member", "member", "actual_member", "result_member", "target_member"], required=False)
    outcome_col = find_col(df_raw, ["OutcomeGroup", "outcome_group", "outcomegroup", "bucket", "class", "label"], required=False)
    date_col = find_col(df_raw, ["PlayDate", "date", "draw date", "play_date", "target_date"], required=False)
    stream_col = find_col(df_raw, ["StreamKey", "stream", "stream_id", "state_game", "streamkey"], required=False)
    out = pd.DataFrame()
    out["PrevSeed"] = df_raw[seed_col].apply(canonical_seed)
    out["WinningMember"] = df_raw[member_col].apply(coerce_member_text) if member_col is not None else None
    out["OutcomeGroup"] = df_raw[outcome_col].apply(coerce_outcome_group) if outcome_col is not None else None
    out["PlayDate"] = df_raw[date_col] if date_col is not None else ""
    out["StreamKey"] = df_raw[stream_col] if stream_col is not None else ""
    out = out[out["PrevSeed"].notna()].reset_index(drop=True)
    feat_df = out["PrevSeed"].apply(compute_features).apply(pd.Series)
    return pd.concat([out.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

def render_target_family(title: str, family: Dict[str, pd.DataFrame], build_slug: str) -> None:
    st.subheader(title)
    for target_value, df in family.items():
        with st.expander(f"{target_value}", expanded=False):
            st.dataframe(safe_display_df(df, 100), use_container_width=True)
            st.download_button(
                f"Download {target_value} / {title}",
                data=df_to_csv_bytes(df),
                file_name=f"{target_value}__{title.lower().replace(' ', '_')}__{build_slug}.csv",
                mime="text/csv",
                key=f"dl_{title}_{target_value}",
            )

def main():
    st.set_page_config(page_title="Core Group Target Deep Miner", layout="wide")
    st.title("Core Group Target Deep Miner")
    st.caption(BUILD_MARKER)
    with st.sidebar:
        st.write(BUILD_MARKER)
        uploaded = st.file_uploader("Upload grouped seed-event CSV", type=["csv", "txt", "tsv", "xlsx", "xls"])
        target_mode = st.selectbox("Target mode", ["OutcomeGroup", "WinningMember"])
        mine_level = st.selectbox("Mine level", ["standard", "expanded"], index=1)
        rows_to_show = st.number_input("Rows to show", min_value=20, max_value=500, value=100, step=20)
        run_btn = st.button("Run miner", type="primary", use_container_width=True)
    if uploaded is None:
        st.info("Upload a grouped CSV. Best format: PrevSeed, WinningMember, PlayDate, StreamKey, OutcomeGroup")
        st.stop()
    try:
        df_raw = load_table(uploaded)
        df = prepare_dataset(df_raw)
    except Exception as e:
        st.error(f"Failed to load/prepare dataset: {e}")
        st.stop()
    st.write(f"Usable rows: **{len(df)}**")
    st.dataframe(safe_display_df(df[["PrevSeed", "WinningMember", "OutcomeGroup", "PlayDate", "StreamKey"]], rows_to_show), use_container_width=True)
    if not run_btn:
        st.stop()
    target_col = "OutcomeGroup" if target_mode == "OutcomeGroup" else "WinningMember"
    if target_col not in df.columns or df[target_col].notna().sum() == 0:
        st.error(f"Target column '{target_col}' is missing or empty.")
        st.stop()
    work = df[df[target_col].notna()].copy().reset_index(drop=True)
    feature_cols = [c for c in work.columns if c.startswith("seed_") or c.startswith("cnt_") or c.startswith("pair_has_") or c.startswith("adj_ord_has_")]
    trait_df = build_trait_matrix(work[feature_cols].copy(), mine_level=mine_level)
    targets = sorted(work[target_col].astype(str).unique().tolist())
    all_scores_frames = []
    single_traits, separator_traits, stacked_buckets = {}, {}, {}
    for target in targets:
        scores = score_traits_one_vs_rest(trait_df, work[target_col], target)
        single_traits[target] = scores
        separator_traits[target] = build_separator_traits(scores, target)
        stacked_buckets[target] = build_stacked_buckets(trait_df, work[target_col], target, scores)
        if len(scores):
            all_scores_frames.append(scores)
    all_scores = pd.concat(all_scores_frames, ignore_index=True) if all_scores_frames else pd.DataFrame()
    st.success(f"Mining complete for target mode: {target_mode}")
    st.dataframe(work[target_col].astype(str).value_counts().rename_axis(target_col).reset_index(name="count"), use_container_width=True)
    if len(all_scores):
        st.subheader("Trait scores — all targets")
        st.dataframe(safe_display_df(all_scores, rows_to_show), use_container_width=True)
        st.download_button("Download trait_scores_all_targets", data=df_to_csv_bytes(all_scores), file_name="trait_scores_all_targets__core025_group_target_deep_miner__2026-04-12.csv", mime="text/csv")
    render_target_family("Single Traits", single_traits, "core025_group_target_deep_miner__2026-04-12")
    render_target_family("Separator Traits", separator_traits, "core025_group_target_deep_miner__2026-04-12")
    render_target_family("Stacked Buckets", stacked_buckets, "core025_group_target_deep_miner__2026-04-12")
    st.subheader("Feature Table")
    st.dataframe(safe_display_df(work, rows_to_show), use_container_width=True)
    st.download_button("Download feature_table", data=df_to_csv_bytes(work), file_name="feature_table__core025_group_target_deep_miner__2026-04-12.csv", mime="text/csv")

if __name__ == "__main__":
    main()
