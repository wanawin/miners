#!/usr/bin/env python3
"""
BUILD: core025_group_target_deep_miner__2026-04-13_v6_full312_autoclassify

Full replacement file.
- Upload + all settings in sidebar
- Explicit Run miner button
- Accepts raw per-event separator exports OR pre-grouped miner CSVs
- Auto-classifies the full per-event universe from actual play results
- Keeps downloads visible after run
- Separate downloads per target for rows / single traits / filtered candidates / separator traits / stacked buckets
- Global downloads for feature table / all target trait scores / run summary / full classified table
- Self-identifying filenames
"""

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_group_target_deep_miner__2026-04-13_v6_full312_autoclassify"
BUILD_SLUG = BUILD_MARKER.replace("BUILD: ", "")
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
    return s if len(s) >= 4 else None


def canonical_prevseed(x) -> Optional[str]:
    s = canonical_seed(x)
    return s[:4] if s is not None else None


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
        "TOP1_WIN": "TOP1_WIN",
        "TOP1": "TOP1_WIN",
        "TOP3_WIN": "TOP3_WIN",
        "TOP3": "TOP3_WIN",
        "OTHER_CAPTURE": "OTHER_CAPTURE",
        "SKIP": "SKIP",
    }
    return aliases.get(s, s if s else None)


def as_int01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def classify_outcome_group_from_per_event(df_raw: pd.DataFrame) -> pd.DataFrame:
    seed_col = find_col(df_raw, ["seed", "PrevSeed", "prev_seed"], required=True)
    winning_member_col = find_col(df_raw, ["winning_member", "WinningMember"], required=True)
    top1_col = find_col(df_raw, ["Top1", "top1"], required=True)
    top2_col = find_col(df_raw, ["Top2", "top2"], required=False)
    top3_col = find_col(df_raw, ["Top3", "top3"], required=False)
    play_rule_hit_col = find_col(df_raw, ["play_rule_hit", "PlayRuleHit"], required=True)
    is_play_top1_col = find_col(df_raw, ["is_play_top1", "IsPlayTop1"], required=False)
    is_play_top2_col = find_col(df_raw, ["is_play_top2", "IsPlayTop2"], required=False)
    is_skip_col = find_col(df_raw, ["is_skip", "IsSkip"], required=False)
    play_mode_col = find_col(df_raw, ["play_mode", "PlayMode"], required=False)
    transition_date_col = find_col(df_raw, ["transition_date", "PlayDate", "date"], required=False)
    stream_col = find_col(df_raw, ["stream", "StreamKey", "stream_id"], required=False)

    out = pd.DataFrame()
    out["PrevSeed"] = df_raw[seed_col].apply(canonical_prevseed)
    out["WinningMember"] = df_raw[winning_member_col].apply(coerce_member_text)
    out["Top1"] = df_raw[top1_col].apply(coerce_member_text)
    out["Top2"] = df_raw[top2_col].apply(coerce_member_text) if top2_col else None
    out["Top3"] = df_raw[top3_col].apply(coerce_member_text) if top3_col else None
    out["PlayDate"] = df_raw[transition_date_col] if transition_date_col else ""
    out["StreamKey"] = df_raw[stream_col] if stream_col else ""

    out["play_rule_hit"] = as_int01(df_raw[play_rule_hit_col])

    if is_play_top1_col:
        out["is_play_top1"] = as_int01(df_raw[is_play_top1_col])
    else:
        out["is_play_top1"] = ((df_raw[play_mode_col].astype(str) == "PLAY_TOP1").fillna(False)).astype(int) if play_mode_col else 0

    if is_play_top2_col:
        out["is_play_top2"] = as_int01(df_raw[is_play_top2_col])
    else:
        out["is_play_top2"] = ((df_raw[play_mode_col].astype(str) == "PLAY_TOP2").fillna(False)).astype(int) if play_mode_col else 0

    if is_skip_col:
        out["is_skip"] = as_int01(df_raw[is_skip_col])
    else:
        out["is_skip"] = ((df_raw[play_mode_col].astype(str) == "SKIP").fillna(False)).astype(int) if play_mode_col else 0

    if play_mode_col:
        out["PlayMode"] = df_raw[play_mode_col].astype(str)
    else:
        out["PlayMode"] = np.select(
            [out["is_play_top1"] == 1, out["is_play_top2"] == 1, out["is_skip"] == 1],
            ["PLAY_TOP1", "PLAY_TOP2", "SKIP"],
            default="UNKNOWN",
        )

    out = out[out["PrevSeed"].notna()].reset_index(drop=True)

    outcome_group = []
    outcome_detail = []
    recommended_play_count = []

    for _, r in out.iterrows():
        winning = r["WinningMember"]
        top1 = r["Top1"]
        top2 = r["Top2"]
        top3 = r["Top3"]
        is_top1 = int(r["is_play_top1"]) == 1
        is_top2 = int(r["is_play_top2"]) == 1
        is_skip = int(r["is_skip"]) == 1
        play_hit = int(r["play_rule_hit"]) == 1

        if is_top1:
            recommended_play_count.append(1)
        elif is_top2:
            recommended_play_count.append(2)
        else:
            recommended_play_count.append(0)

        if is_skip:
            outcome_group.append("SKIP")
            outcome_detail.append("SKIP")
            continue

        if is_top1 and top1 == winning:
            outcome_group.append("TOP1_WIN")
            outcome_detail.append("TOP1_WIN_FROM_PLAY_TOP1")
            continue

        if is_top2 and top1 == winning:
            outcome_group.append("WASTE")
            outcome_detail.append("TOP2_WASTE")
            continue

        if is_top2 and top1 != winning and top2 == winning:
            outcome_group.append("NEEDED")
            outcome_detail.append("TOP2_NEEDED")
            continue

        if play_hit:
            if top3 is not None and top3 == winning:
                outcome_group.append("TOP3_WIN")
                outcome_detail.append("TOP3_CAPTURE")
            else:
                outcome_group.append("OTHER_CAPTURE")
                outcome_detail.append("OTHER_CAPTURE")
            continue

        outcome_group.append("MISS")
        if is_top1:
            outcome_detail.append("MISS_FROM_PLAY_TOP1")
        elif is_top2:
            outcome_detail.append("MISS_FROM_PLAY_TOP2")
        else:
            outcome_detail.append("MISS_UNCLASSIFIED")

    out["OutcomeGroup"] = outcome_group
    out["OutcomeDetail"] = outcome_detail
    out["recommended_play_count"] = recommended_play_count
    out["BuildMarker"] = BUILD_SLUG
    return out


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


def prepare_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    normalized_cols = {_norm(c): c for c in df_raw.columns}
    looks_like_per_event = ("playrulehit" in normalized_cols) and (("top1" in normalized_cols) or ("winningmember" in normalized_cols) or ("winning_member" in normalized_cols))
    if looks_like_per_event:
        out = classify_outcome_group_from_per_event(df_raw)
        source_type = "raw_per_event_autoclassified"
    else:
        seed_col = find_col(df_raw, ["PrevSeed", "seed", "Prev Seed"], required=True)
        member_col = find_col(df_raw, ["WinningMember", "winning_member", "winner_member"], required=False)
        outcome_col = find_col(df_raw, ["OutcomeGroup", "outcome_group", "bucket", "class", "label"], required=False)
        date_col = find_col(df_raw, ["PlayDate", "date", "draw date", "play_date", "target_date"], required=False)
        stream_col = find_col(df_raw, ["StreamKey", "stream", "stream_id", "state_game"], required=False)

        out = pd.DataFrame()
        out["PrevSeed"] = df_raw[seed_col].apply(canonical_prevseed)
        out["WinningMember"] = df_raw[member_col].apply(coerce_member_text) if member_col is not None else None
        out["OutcomeGroup"] = df_raw[outcome_col].apply(coerce_outcome_group) if outcome_col is not None else None
        out["OutcomeDetail"] = out["OutcomeGroup"]
        out["PlayDate"] = df_raw[date_col] if date_col is not None else ""
        out["StreamKey"] = df_raw[stream_col] if stream_col is not None else ""
        out["BuildMarker"] = BUILD_SLUG
        out = out[out["PrevSeed"].notna()].reset_index(drop=True)
        source_type = "pre_grouped_input"

    feat_df = out["PrevSeed"].apply(compute_features).apply(pd.Series)
    out = pd.concat([out.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    return out, source_type


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
    if vals.notna().sum() >= 10:
        for q in [0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9]:
            try:
                thresh = float(vals.quantile(q))
            except Exception:
                continue
            if np.isnan(thresh):
                continue
            label = int(thresh) if float(thresh).is_integer() else round(float(thresh), 3)
            out[f"{prefix}<={label}"] = vals <= thresh
            out[f"{prefix}>={label}"] = vals >= thresh
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
    pairs = [("seed_has0", "seed_has9"), ("seed_outer_equal", "seed_inner_equal"), ("seed_sum_even", "seed_has_pair")]
    for left, right in pairs:
        if left in trait_cols and right in trait_cols:
            trait_cols[f"{left} AND {right}"] = trait_cols[left] & trait_cols[right]


def build_trait_matrix(df_feat: pd.DataFrame, mine_level: str = "expanded") -> pd.DataFrame:
    trait_cols: Dict[str, pd.Series] = {}
    numeric_cols = [c for c in df_feat.columns if c.startswith("seed_") or c.startswith("cnt_")]
    categorical_cols = ["seed_parity_pattern", "seed_highlow_pattern", "seed_repeat_shape", "seed_sorted"]
    for c in numeric_cols:
        trait_cols.update(bin_numeric_series(df_feat[c], c))
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
    trait_df = pd.DataFrame(trait_cols, index=df_feat.index)
    return dedupe_columns(trait_df.astype(bool))


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
            "separator_flag": int(hit_rate_false == 0),
            "BuildMarker": BUILD_SLUG,
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


def build_stacked_buckets(trait_df: pd.DataFrame, y: pd.Series, target_value: str, base_scores: pd.DataFrame, top_n: int = 200, top_k_traits: int = 15) -> pd.DataFrame:
    if len(base_scores) == 0:
        return pd.DataFrame()
    top_traits = [t for t in base_scores.head(int(top_k_traits))["trait"].tolist() if t in trait_df.columns]
    target = (y.astype(str) == str(target_value)).astype(int)
    rows = []
    for i in range(len(top_traits)):
        for j in range(i + 1, len(top_traits)):
            t1, t2 = top_traits[i], top_traits[j]
            mask = trait_df[t1] & trait_df[t2]
            support = int(mask.sum())
            if support < 2:
                continue
            hits = int(target[mask].sum())
            hit_rate = hits / support if support else 0.0
            rows.append({
                "target_value": str(target_value),
                "stack": f"{t1} AND {t2}",
                "support": support,
                "hits_true": hits,
                "hit_rate_true": hit_rate,
                "BuildMarker": BUILD_SLUG,
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["hit_rate_true", "support", "hits_true"], ascending=[False, False, False]).head(int(top_n)).reset_index(drop=True) if len(out) else out


def filter_candidate_traits(scores: pd.DataFrame, min_support: int, min_gap: float, min_hit_rate_true: float, min_lift: float) -> pd.DataFrame:
    if len(scores) == 0:
        return scores.copy()
    out = scores.copy()
    out = out[
        (out["support"] >= int(min_support)) &
        (out["gap"] >= float(min_gap)) &
        (out["hit_rate_true"] >= float(min_hit_rate_true))
    ]
    if "lift" in out.columns:
        out = out[(out["lift"].fillna(0) >= float(min_lift))]
    return out.reset_index(drop=True)


def build_classification_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = int(len(df))
    vc = df["OutcomeGroup"].astype(str).value_counts(dropna=False)
    for label, count in vc.items():
        rows.append({"label": label, "count": int(count), "pct_of_rows": (float(count) / total) if total else 0.0, "BuildMarker": BUILD_SLUG})
    return pd.DataFrame(rows)


def build_operational_summary_from_classified(df: pd.DataFrame) -> pd.DataFrame:
    total = int(len(df))
    return pd.DataFrame([
        {"metric": "winner_event_rows", "value": total},
        {"metric": "top1_wins__play_top1_and_top1_won", "value": int((df["OutcomeGroup"] == "TOP1_WIN").sum())},
        {"metric": "top2_waste", "value": int((df["OutcomeGroup"] == "WASTE").sum())},
        {"metric": "top2_needed", "value": int((df["OutcomeGroup"] == "NEEDED").sum())},
        {"metric": "misses", "value": int((df["OutcomeGroup"] == "MISS").sum())},
        {"metric": "play_top1_rows", "value": int(df["is_play_top1"].sum()) if "is_play_top1" in df.columns else 0},
        {"metric": "play_top2_rows", "value": int(df["is_play_top2"].sum()) if "is_play_top2" in df.columns else 0},
        {"metric": "skips", "value": int(df["is_skip"].sum()) if "is_skip" in df.columns else 0},
    ])


def main():
    st.set_page_config(page_title="Core Group Target Deep Miner", layout="wide")
    st.title("Core Group Target Deep Miner")
    st.caption(BUILD_MARKER)

    if "miner_outputs_v6" not in st.session_state:
        st.session_state["miner_outputs_v6"] = None

    with st.sidebar:
        st.write(BUILD_MARKER)
        uploaded = st.file_uploader("Upload raw per-event export or grouped CSV", type=["csv", "txt", "tsv", "xlsx", "xls"])
        target_mode = st.selectbox("Target mode", ["OutcomeGroup", "WinningMember"], index=0)
        mine_level = st.selectbox("Mine level", ["standard", "expanded"], index=1)
        rows_to_show = st.number_input("Rows to preview", min_value=20, max_value=500, value=100, step=20)

        st.markdown("### Candidate filters")
        min_support = st.number_input("Minimum support", min_value=1, max_value=5000, value=5, step=1)
        min_gap = st.number_input("Minimum gap", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")
        min_hit_rate_true = st.number_input("Minimum hit_rate_true", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
        min_lift = st.number_input("Minimum lift", min_value=0.0, max_value=100.0, value=1.10, step=0.05, format="%.2f")

        st.markdown("### Export limits")
        max_filtered_rows = st.number_input("Filtered candidate rows per target", min_value=10, max_value=5000, value=250, step=10)
        max_separator_rows = st.number_input("Separator rows per target", min_value=10, max_value=5000, value=250, step=10)
        max_stacked_rows = st.number_input("Stacked bucket rows per target", min_value=10, max_value=5000, value=250, step=10)
        stacked_top_k_traits = st.number_input("Top traits used to build stacked buckets", min_value=5, max_value=200, value=30, step=5)

        run_btn = st.button("Run miner", type="primary", use_container_width=True)

    if run_btn:
        if uploaded is None:
            st.error("Upload a file first.")
            st.session_state["miner_outputs_v6"] = None
        else:
            try:
                df_raw = load_table(uploaded)
                df, source_type = prepare_dataset(df_raw)
                target_col = "OutcomeGroup" if target_mode == "OutcomeGroup" else "WinningMember"
                if target_col not in df.columns or df[target_col].notna().sum() == 0:
                    raise ValueError(f"Target column '{target_col}' is missing or empty.")
                work = df[df[target_col].notna()].copy().reset_index(drop=True)

                feature_cols = [c for c in work.columns if c.startswith("seed_") or c.startswith("cnt_") or c.startswith("pair_has_") or c.startswith("adj_ord_has_")]
                trait_df = build_trait_matrix(work[feature_cols].copy(), mine_level=mine_level)

                targets = sorted(work[target_col].astype(str).unique().tolist())
                all_scores_frames = []
                single_traits = {}
                separator_traits = {}
                stacked_buckets = {}
                filtered_candidates = {}
                grouped_rows = {}

                for target in targets:
                    target_rows = work[work[target_col].astype(str) == str(target)].copy().reset_index(drop=True)
                    grouped_rows[target] = target_rows
                    scores = score_traits_one_vs_rest(trait_df, work[target_col], target)
                    single_traits[target] = scores
                    separator_traits[target] = build_separator_traits(scores, target, top_n=int(max_separator_rows))
                    stacked_buckets[target] = build_stacked_buckets(trait_df, work[target_col], target, scores, top_n=int(max_stacked_rows), top_k_traits=int(stacked_top_k_traits))
                    filtered_candidates[target] = filter_candidate_traits(scores, min_support, min_gap, min_hit_rate_true, min_lift).head(int(max_filtered_rows)).reset_index(drop=True)
                    if len(scores):
                        all_scores_frames.append(scores)

                all_scores = pd.concat(all_scores_frames, ignore_index=True) if all_scores_frames else pd.DataFrame()
                run_summary = pd.DataFrame([{
                    "target_value": target,
                    "grouped_rows": int(len(grouped_rows[target])),
                    "single_traits": int(len(single_traits[target])),
                    "filtered_candidates": int(len(filtered_candidates[target])),
                    "separator_traits": int(len(separator_traits[target])),
                    "stacked_buckets": int(len(stacked_buckets[target])),
                    "BuildMarker": BUILD_SLUG,
                } for target in targets])

                classification_summary = build_classification_summary(df) if "OutcomeGroup" in df.columns else pd.DataFrame()
                operational_summary = build_operational_summary_from_classified(df) if "OutcomeGroup" in df.columns else pd.DataFrame()

                st.session_state["miner_outputs_v6"] = {
                    "source_name": uploaded.name,
                    "source_type": source_type,
                    "prepared_df": df,
                    "work_df": work,
                    "target_mode": target_mode,
                    "target_col": target_col,
                    "targets": targets,
                    "trait_df_cols": int(len(trait_df.columns)),
                    "all_scores": all_scores,
                    "single_traits": single_traits,
                    "separator_traits": separator_traits,
                    "stacked_buckets": stacked_buckets,
                    "filtered_candidates": filtered_candidates,
                    "grouped_rows": grouped_rows,
                    "run_summary": run_summary,
                    "classification_summary": classification_summary,
                    "operational_summary": operational_summary,
                }
            except Exception as e:
                st.session_state["miner_outputs_v6"] = None
                st.error(f"Failed to run miner: {e}")

    outputs = st.session_state.get("miner_outputs_v6")
    if outputs is None:
        st.info("Upload the file in the sidebar, choose settings, then click Run miner.")
        return

    st.success("Mining complete")
    st.write(f"**Source file:** {outputs['source_name']}")
    st.write(f"**Source type:** {outputs['source_type']}")
    st.write(f"**Prepared rows:** {len(outputs['prepared_df'])}")
    st.write(f"**Rows used for target mode '{outputs['target_mode']}':** {len(outputs['work_df'])}")
    st.write(f"**Trait columns built:** {outputs['trait_df_cols']}")
    st.write(f"**Targets found:** {', '.join(outputs['targets'])}")

    if len(outputs["classification_summary"]) > 0:
        st.subheader("Full classification summary")
        st.dataframe(outputs["classification_summary"], use_container_width=True, hide_index=True)

    if len(outputs["operational_summary"]) > 0:
        st.subheader("Operational summary")
        st.dataframe(outputs["operational_summary"], use_container_width=True, hide_index=True)

    st.subheader("Run summary")
    st.dataframe(outputs["run_summary"], use_container_width=True, hide_index=True)

    st.subheader("Global downloads")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("Download full classified table", data=df_to_csv_bytes(outputs["prepared_df"]), file_name=f"full_classified_table__{BUILD_SLUG}.csv", mime="text/csv", use_container_width=True, key="dl_full_classified_v6")
    with c2:
        st.download_button("Download trait_scores_all_targets", data=df_to_csv_bytes(outputs["all_scores"]), file_name=f"trait_scores_all_targets__{BUILD_SLUG}.csv", mime="text/csv", use_container_width=True, key="dl_all_scores_v6")
    with c3:
        st.download_button("Download feature table", data=df_to_csv_bytes(outputs["work_df"]), file_name=f"feature_table__{BUILD_SLUG}.csv", mime="text/csv", use_container_width=True, key="dl_feature_table_v6")
    with c4:
        st.download_button("Download run summary", data=df_to_csv_bytes(outputs["run_summary"]), file_name=f"run_summary__{BUILD_SLUG}.csv", mime="text/csv", use_container_width=True, key="dl_run_summary_v6")

    for target in outputs["targets"]:
        st.markdown("---")
        st.subheader(str(target))
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            st.download_button(f"Download {target} rows", data=df_to_csv_bytes(outputs["grouped_rows"][target]), file_name=f"{target}__rows__{BUILD_SLUG}.csv", mime="text/csv", use_container_width=True, key=f"dl_rows_{target}_v6")
        with d2:
            st.download_button(f"Download {target} single traits", data=df_to_csv_bytes(outputs["single_traits"][target]), file_name=f"{target}__single_traits__{BUILD_SLUG}.csv", mime="text/csv", use_container_width=True, key=f"dl_single_{target}_v6")
        with d3:
            st.download_button(f"Download {target} filtered candidates", data=df_to_csv_bytes(outputs["filtered_candidates"][target]), file_name=f"{target}__filtered_candidates__{BUILD_SLUG}.csv", mime="text/csv", use_container_width=True, key=f"dl_filtered_{target}_v6")
        with d4:
            st.download_button(f"Download {target} separator traits", data=df_to_csv_bytes(outputs["separator_traits"][target]), file_name=f"{target}__separator_traits__{BUILD_SLUG}.csv", mime="text/csv", use_container_width=True, key=f"dl_separator_{target}_v6")
        with d5:
            st.download_button(f"Download {target} stacked buckets", data=df_to_csv_bytes(outputs["stacked_buckets"][target]), file_name=f"{target}__stacked_buckets__{BUILD_SLUG}.csv", mime="text/csv", use_container_width=True, key=f"dl_stacked_{target}_v6")

        t1, t2, t3, t4, t5 = st.tabs(["Rows preview", "Single traits preview", "Filtered candidates preview", "Separator traits preview", "Stacked buckets preview"])
        with t1:
            st.dataframe(safe_display_df(outputs["grouped_rows"][target], rows_to_show), use_container_width=True, hide_index=True)
        with t2:
            st.dataframe(safe_display_df(outputs["single_traits"][target], rows_to_show), use_container_width=True, hide_index=True)
        with t3:
            st.dataframe(safe_display_df(outputs["filtered_candidates"][target], rows_to_show), use_container_width=True, hide_index=True)
        with t4:
            st.dataframe(safe_display_df(outputs["separator_traits"][target], rows_to_show), use_container_width=True, hide_index=True)
        with t5:
            st.dataframe(safe_display_df(outputs["stacked_buckets"][target], rows_to_show), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
