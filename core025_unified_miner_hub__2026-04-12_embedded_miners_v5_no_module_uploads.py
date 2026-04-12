
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

APP_BUILD = "core025_unified_miner_hub__2026-04-12_embedded_miners_v5_no_module_uploads"

EMBEDDED_MODULES = {
    "deep": r"""#!/usr/bin/env python3
\"\"\"
core025_deep_trait_miner_streamlit_ready_v4__2026-03-24.py

Supports BOTH:
1) CLI usage:
   python core025_deep_trait_miner_streamlit_ready_v4__2026-03-24.py --input your_file.csv

2) Streamlit usage:
   streamlit run core025_deep_trait_miner_streamlit_ready_v4__2026-03-24.py

Full corrected file.
No placeholders. No simulations.

What is new in v4:
- Multi-pass mining workflow
- Second-pass subset builder from first-pass results
- Pass mode selector
- Target-member selector
- Objective selector
- Mine-level selector (standard / expanded)
- Stored bucket row coverage for residual mining
- Session-state persistence so downloads do not clear mined results
- Duplicate-column-safe display/export
\"\"\"

from __future__ import annotations

import argparse
import io
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None


MEMBERS = [25, 225, 255]
MIRROR_PAIRS = {(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)}
DIGITS = list(range(10))


def member_label(v: int) -> str:
    return {25: "0025", 225: "0225", 255: "0255"}.get(v, str(v))


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


def coerce_member(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    nums = re.findall(r"\d+", s)
    if not nums:
        return None
    for token in reversed(nums):
        try:
            v = int(token)
        except Exception:
            continue
        if v in MEMBERS:
            return v
        z = token.zfill(4)
        if z == "0025":
            return 25
        if z == "0225":
            return 225
        if z == "0255":
            return 255
    return None


def canonical_seed(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = re.sub(r"\D", "", str(x))
    return s if len(s) == 4 else None


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

    pairwise_absdiff = []
    for i in range(4):
        for j in range(i + 1, 4):
            pairwise_absdiff.append(abs(d[i] - d[j]))
    adj_absdiff = [abs(d[i] - d[i + 1]) for i in range(3)]

    features: Dict[str, object] = {
        "feat_seed": seed,
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
        "seed_pos1": d[0],
        "seed_pos2": d[1],
        "seed_pos3": d[2],
        "seed_pos4": d[3],
        "seed_first_last_sum": d[0] + d[3],
        "seed_middle_sum": d[1] + d[2],
        "seed_absdiff_outer_inner": abs((d[0] + d[3]) - (d[1] + d[2])),
        "seed_parity_pattern": parity,
        "seed_highlow_pattern": highlow,
        "seed_sorted": "".join(map(str, sorted(d))),
        "seed_pair_tokens": "|".join(sorted(as_pair_tokens(seed))),
        "seed_adj_pairs_ordered": "|".join(ordered_adj),
        "seed_adj_pairs_unordered": "|".join(sorted(as_unordered_adj_pairs(seed))),
        "seed_outer_equal": int(d[0] == d[3]),
        "seed_inner_equal": int(d[1] == d[2]),
        "seed_palindrome_like": int(d[0] == d[3] and d[1] == d[2]),
        "seed_same_adjacent_count": int(sum(d[i] == d[i + 1] for i in range(3))),
        "seed_pos1_lt_pos2": int(d[0] < d[1]),
        "seed_pos2_lt_pos3": int(d[1] < d[2]),
        "seed_pos3_lt_pos4": int(d[2] < d[3]),
        "seed_pos1_lt_pos3": int(d[0] < d[2]),
        "seed_pos2_lt_pos4": int(d[1] < d[3]),
        "seed_pos1_eq_pos2": int(d[0] == d[1]),
        "seed_pos2_eq_pos3": int(d[1] == d[2]),
        "seed_pos3_eq_pos4": int(d[2] == d[3]),
        "seed_outer_gt_inner": int((d[0] + d[3]) > (d[1] + d[2])),
        "seed_sum_even": int(s % 2 == 0),
        "seed_sum_high_20plus": int(s >= 20),
    }

    for k in DIGITS:
        features[f"seed_has{k}"] = int(k in cnt)
        features[f"seed_cnt{k}"] = int(cnt.get(k, 0))

    shape = "".join(map(str, sorted(cnt.values(), reverse=True)))
    shape_name = {
        "1111": "all_unique",
        "211": "one_pair",
        "22": "two_pair",
        "31": "trip",
        "4": "quad",
    }.get(shape, f"shape_{shape}")
    features["seed_repeat_shape"] = shape_name

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

    if len(uniques) <= 20 and all(float(x).is_integer() for x in uniques):
        int_uniques = [int(x) for x in uniques]
        for lo in int_uniques:
            for hi in int_uniques:
                if lo < hi and (hi - lo) <= 3:
                    out[f"{prefix}_in[{lo},{hi}]"] = (vals >= lo) & (vals <= hi)
    return out


def categorical_series_traits(s: pd.Series, prefix: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    vc = s.astype(str).value_counts(dropna=False)
    for v, n in vc.items():
        if n >= 3:
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

    if "seed_has9" in df_feat.columns and "seed_repeat_shape" in df_feat.columns:
        combo = df_feat["seed_has9"].astype(str) + "|" + df_feat["seed_repeat_shape"].astype(str)
        trait_cols.update(categorical_series_traits(combo, "x_has9_repeatshape"))

    binary_pairs = [
        ("seed_has0", "seed_has9"),
        ("seed_has2", "seed_has5"),
        ("seed_outer_equal", "seed_inner_equal"),
        ("seed_sum_even", "seed_has_pair"),
        ("seed_sum_high_20plus", "seed_has9"),
        ("seed_pos1_lt_pos3", "seed_pos2_lt_pos4"),
        ("seed_same_adjacent_count==0", "seed_has_pair"),
    ]

    for left, right in binary_pairs:
        if left in trait_cols and right in trait_cols:
            trait_cols[f"{left} AND {right}"] = trait_cols[left] & trait_cols[right]


def build_trait_matrix(df_feat: pd.DataFrame, mine_level: str = "standard") -> pd.DataFrame:
    trait_cols: Dict[str, pd.Series] = {}

    numeric_cols = [
        c for c in df_feat.columns
        if (c.startswith("seed_") or c.startswith("cnt_")) and c not in {"feat_seed"}
    ]
    categorical_cols = [
        "seed_parity_pattern",
        "seed_highlow_pattern",
        "seed_repeat_shape",
        "seed_sorted",
    ]

    for c in numeric_cols:
        ser = df_feat[c]
        if pd.api.types.is_numeric_dtype(ser):
            trait_cols.update(bin_numeric_series(ser, c))

    for c in categorical_cols:
        if c in df_feat.columns:
            trait_cols.update(categorical_series_traits(df_feat[c], c))

    sparse_prefixes = ("pair_has_", "adj_ord_has_", "seed_has", "seed_cnt")
    for c in df_feat.columns:
        if c.startswith(sparse_prefixes):
            ser = pd.to_numeric(df_feat[c], errors="coerce").fillna(0).astype(int)
            if ser.sum() >= 3:
                trait_cols[c] = ser.astype(bool)

    if mine_level == "expanded":
        add_expanded_interactions(df_feat, trait_cols)

    trait_df = pd.DataFrame(trait_cols, index=df_feat.index).astype(bool)
    return dedupe_columns(trait_df)


def score_traits_one_vs_rest(trait_df: pd.DataFrame, y_member: pd.Series, member: int) -> pd.DataFrame:
    target = (y_member == member).astype(int)
    rows: List[Dict[str, object]] = []
    total_hits = int(target.sum())
    total_n = int(len(target))
    base_rate = total_hits / total_n if total_n else 0.0

    for trait in trait_df.columns:
        mask = trait_df[trait].fillna(False).astype(bool)
        support = int(mask.sum())
        if support == 0 or support == total_n:
            continue

        hits_true = int(target[mask].sum())
        misses_true = int(support - hits_true)
        hit_rate_true = hits_true / support if support else 0.0

        inv = ~mask
        support_false = int(inv.sum())
        hits_false = int(target[inv].sum())
        misses_false = int(support_false - hits_false)
        hit_rate_false = hits_false / support_false if support_false else 0.0

        rows.append({
            "member": member,
            "member_label": member_label(member),
            "trait": trait,
            "support": support,
            "support_pct": support / total_n if total_n else 0.0,
            "hits_true": hits_true,
            "misses_true": misses_true,
            "hit_rate_true": hit_rate_true,
            "support_false": support_false,
            "hits_false": hits_false,
            "misses_false": misses_false,
            "hit_rate_false": hit_rate_false,
            "base_rate_member": base_rate,
            "lift_vs_base": hit_rate_true - base_rate,
            "precision_gap": hit_rate_true - hit_rate_false,
            "is_separator": int(hits_true == 0 and support > 0),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["precision_gap", "hit_rate_true", "support"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def find_separator_traits(scored_traits: pd.DataFrame, min_support: int = 8) -> pd.DataFrame:
    out = scored_traits[(scored_traits["is_separator"] == 1) & (scored_traits["support"] >= min_support)].copy()
    if not out.empty:
        out = out.sort_values(["support", "hit_rate_false"], ascending=[False, False]).reset_index(drop=True)
    return out


def bucket_mask_from_traits(trait_df: pd.DataFrame, bucket_traits: str) -> pd.Series:
    parts = [p.strip() for p in str(bucket_traits).split(" AND ") if p.strip()]
    if not parts:
        return pd.Series([False] * len(trait_df), index=trait_df.index)
    mask = pd.Series([True] * len(trait_df), index=trait_df.index)
    for t in parts:
        if t not in trait_df.columns:
            mask &= False
        else:
            mask &= trait_df[t].fillna(False).astype(bool)
    return mask


def greedy_bucket_search(
    trait_df: pd.DataFrame,
    y_member: pd.Series,
    member: int,
    scored_traits: pd.DataFrame,
    min_support: int = 6,
    top_k_traits: int = 80,
    max_depth: int = 4,
) -> pd.DataFrame:
    target = (y_member == member).astype(int)
    candidates = scored_traits[
        (scored_traits["support"] >= min_support) &
        (scored_traits["hit_rate_true"] > scored_traits["base_rate_member"])
    ].head(top_k_traits)

    rows = []
    seen = set()

    for start_trait in candidates["trait"].tolist():
        mask = trait_df[start_trait].copy()
        chosen = [start_trait]

        for depth in range(1, max_depth + 1):
            support = int(mask.sum())
            if support < min_support:
                break

            hits = int(target[mask].sum())
            hit_rate = hits / support if support else 0.0
            base_rate = float(target.mean()) if len(target) else 0.0
            misses = int(support - hits)
            key = tuple(chosen)

            if key not in seen:
                seen.add(key)
                rows.append({
                    "member": member,
                    "member_label": member_label(member),
                    "depth": len(chosen),
                    "bucket_traits": " AND ".join(chosen),
                    "support": support,
                    "hits": hits,
                    "misses": misses,
                    "hit_rate": hit_rate,
                    "base_rate_member": base_rate,
                    "lift_vs_base": hit_rate - base_rate,
                })

            if depth == max_depth:
                break

            best_next_trait = None
            best_next_rate = hit_rate
            best_next_support = support
            best_next_mask = None

            for nxt in candidates["trait"].tolist():
                if nxt in chosen:
                    continue
                nxt_mask = mask & trait_df[nxt]
                nxt_support = int(nxt_mask.sum())
                if nxt_support < min_support:
                    continue
                nxt_hits = int(target[nxt_mask].sum())
                nxt_rate = nxt_hits / nxt_support if nxt_support else 0.0

                if (nxt_rate > best_next_rate) or (
                    math.isclose(nxt_rate, best_next_rate) and nxt_support > best_next_support
                ):
                    best_next_rate = nxt_rate
                    best_next_support = nxt_support
                    best_next_trait = nxt
                    best_next_mask = nxt_mask

            if best_next_trait is None:
                break

            chosen.append(best_next_trait)
            mask = best_next_mask

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["hit_rate", "support", "depth"], ascending=[False, False, True]).reset_index(drop=True)
        out.insert(0, "bucket_id", range(1, len(out) + 1))
    return out


def read_table_from_path(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".txt", ".tsv"):
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input type: {path.suffix}")


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
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
    raise ValueError(f"Unsupported uploaded input type: {uploaded_file.name}")


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    seed_col = find_col(df, ["seed", "seed_result", "previous_result", "prev_result", "prior_result"], required=True)
    member_col = find_col(df, ["true_member", "winning_member", "winner_member", "member", "actual_member", "result_member", "target_member"], required=True)
    date_col = find_col(df, ["date", "draw date", "play_date", "target_date"], required=False)
    stream_col = find_col(df, ["stream", "stream_id", "state_game", "streamkey"], required=False)
    return {"seed_col": seed_col, "member_col": member_col, "date_col": date_col, "stream_col": stream_col}


def prepare_event_table(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    cols = detect_columns(df_raw)

    df = df_raw.copy()
    df["seed"] = df[cols["seed_col"]].apply(canonical_seed)
    df["true_member"] = df[cols["member_col"]].apply(coerce_member)
    df["date"] = pd.to_datetime(df[cols["date_col"]], errors="coerce") if cols["date_col"] else pd.NaT
    df["stream"] = df[cols["stream_col"]].astype(str) if cols["stream_col"] else ""
    df = df.dropna(subset=["seed", "true_member"]).copy()
    df["true_member"] = df["true_member"].astype(int)

    if len(df) == 0:
        raise ValueError("After cleaning, there are 0 usable rows with both seed and true member.")

    df = dedupe_columns(df.reset_index(drop=True))
    df["row_id"] = np.arange(1, len(df) + 1)
    return df, cols


def summarize_text(
    df_events: pd.DataFrame,
    detected_cols: Dict[str, Optional[str]],
    per_member_scores: Dict[int, pd.DataFrame],
    per_member_separators: Dict[int, pd.DataFrame],
    per_member_buckets: Dict[int, pd.DataFrame],
    pass_label: str,
    mine_level: str,
    objective: str,
) -> str:
    lines = []
    lines.append("CORE 025 DEEP TRAIT MINER SUMMARY")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append(f"Pass label: {pass_label}")
    lines.append(f"Mine level: {mine_level}")
    lines.append(f"Objective: {objective}")
    lines.append("")
    lines.append("Detected columns:")
    for k, v in detected_cols.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append(f"Usable rows: {len(df_events)}")
    lines.append("Member counts:")
    vc = df_events["true_member"].value_counts().sort_index()
    for m in MEMBERS:
        lines.append(f"  - {member_label(m)}: {int(vc.get(m, 0))}")
    lines.append("")

    for m in MEMBERS:
        lines.append(f"=== MEMBER {member_label(m)} ===")
        s = per_member_scores.get(m, pd.DataFrame())
        z = per_member_separators.get(m, pd.DataFrame())
        b = per_member_buckets.get(m, pd.DataFrame())

        if not s.empty:
            lines.append("Top single traits:")
            for _, r in s.head(10).iterrows():
                lines.append(f"  - {r['trait']} | support={int(r['support'])} | hit_rate_true={r['hit_rate_true']:.3f} | gap={r['precision_gap']:.3f}")
        else:
            lines.append("Top single traits: none")
        lines.append("")

        if not z.empty:
            lines.append("Top separator traits (0 hits when trait true):")
            for _, r in z.head(10).iterrows():
                lines.append(f"  - {r['trait']} | support={int(r['support'])} | hits_true=0 | member_hit_rate_when_false={r['hit_rate_false']:.3f}")
        else:
            lines.append("Top separator traits: none")
        lines.append("")

        if not b.empty:
            lines.append("Top stacked bucket candidates:")
            for _, r in b.head(10).iterrows():
                lines.append(f"  - bucket_id={int(r['bucket_id'])} | {r['bucket_traits']} | support={int(r['support'])} | hit_rate={r['hit_rate']:.3f} | lift={r['lift_vs_base']:.3f}")
        else:
            lines.append("Top stacked bucket candidates: none")
        lines.append("")

    return "\n".join(lines)


def build_bucket_row_indices(trait_df: pd.DataFrame, per_member_buckets: Dict[int, pd.DataFrame]) -> Dict[int, Dict[int, List[int]]]:
    out: Dict[int, Dict[int, List[int]]] = {}
    for member, bdf in per_member_buckets.items():
        member_map: Dict[int, List[int]] = {}
        if bdf is not None and not bdf.empty:
            for _, row in bdf.iterrows():
                bucket_id = int(row["bucket_id"])
                mask = bucket_mask_from_traits(trait_df, row["bucket_traits"])
                member_map[bucket_id] = trait_df.index[mask].tolist()
        out[member] = member_map
    return out


def run_mining(
    df_raw: pd.DataFrame,
    min_support: int = 8,
    bucket_min_support: int = 6,
    bucket_top_k: int = 80,
    bucket_max_depth: int = 4,
    mine_level: str = "standard",
    pass_label: str = "Pass 1 - Full dataset",
    objective: str = "positive_buckets",
) -> Dict[str, object]:
    events, detected_cols = prepare_event_table(df_raw)

    feat_rows = [compute_features(seed) for seed in events["seed"].astype(str)]
    df_feat = dedupe_columns(pd.DataFrame(feat_rows))
    df_all = dedupe_columns(pd.concat([events.reset_index(drop=True), df_feat.reset_index(drop=True)], axis=1))
    trait_df = build_trait_matrix(df_feat, mine_level=mine_level)

    per_member_scores: Dict[int, pd.DataFrame] = {}
    per_member_separators: Dict[int, pd.DataFrame] = {}
    per_member_buckets: Dict[int, pd.DataFrame] = {}
    all_scores = []

    for member in MEMBERS:
        scored = dedupe_columns(score_traits_one_vs_rest(trait_df, df_all["true_member"], member))
        per_member_scores[member] = scored
        all_scores.append(scored)
        per_member_separators[member] = dedupe_columns(find_separator_traits(scored, min_support=min_support))
        per_member_buckets[member] = dedupe_columns(greedy_bucket_search(
            trait_df=trait_df,
            y_member=df_all["true_member"],
            member=member,
            scored_traits=scored,
            min_support=bucket_min_support,
            top_k_traits=bucket_top_k,
            max_depth=bucket_max_depth,
        ))

    all_scores_df = dedupe_columns(pd.concat(all_scores, ignore_index=True)) if all_scores else pd.DataFrame()
    bucket_row_indices = build_bucket_row_indices(trait_df, per_member_buckets)

    summary_text = summarize_text(
        df_events=df_all,
        detected_cols=detected_cols,
        per_member_scores=per_member_scores,
        per_member_separators=per_member_separators,
        per_member_buckets=per_member_buckets,
        pass_label=pass_label,
        mine_level=mine_level,
        objective=objective,
    )

    return {
        "events": df_all,
        "features_only": df_feat,
        "trait_df": trait_df,
        "detected_cols": detected_cols,
        "all_scores": all_scores_df,
        "per_member_scores": per_member_scores,
        "per_member_separators": per_member_separators,
        "per_member_buckets": per_member_buckets,
        "bucket_row_indices": bucket_row_indices,
        "summary_text": summary_text,
        "completed_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "pass_label": pass_label,
        "mine_level": mine_level,
        "objective": objective,
    }


def minimal_event_df_from_events(events: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["seed", "true_member", "date", "stream"] if c in events.columns]
    return dedupe_columns(events[keep].copy())


def build_subset_from_results(
    base_results: Dict[str, object],
    subset_mode: str,
    target_member: int,
    selected_bucket_id: Optional[int],
) -> Tuple[pd.DataFrame, str]:
    events = dedupe_columns(base_results["events"].copy())
    mask = pd.Series([True] * len(events), index=events.index)

    if subset_mode == "full_dataset_again":
        label = "Pass 2 - Full dataset again"

    elif subset_mode == "member_only_full":
        mask = events["true_member"] == target_member
        label = f"Pass 2 - Member-only full rows for {member_label(target_member)}"

    elif subset_mode == "no9_regime":
        mask = events["seed"].astype(str).str.contains("9") == False
        label = "Pass 2 - No-9 regime"

    elif subset_mode == "has9_regime":
        mask = events["seed"].astype(str).str.contains("9") == True
        label = "Pass 2 - Has-9 regime"

    elif subset_mode == "pair_only_regime":
        mask = events["seed_repeat_shape"].astype(str).isin(["one_pair", "two_pair", "trip", "quad"])
        label = "Pass 2 - Pair-or-more regime"

    elif subset_mode == "all_unique_regime":
        mask = events["seed_repeat_shape"].astype(str) == "all_unique"
        label = "Pass 2 - All-unique regime"

    else:
        if selected_bucket_id is None:
            raise ValueError("This pass mode requires a selected bucket.")
        member_map = base_results["bucket_row_indices"].get(target_member, {})
        bucket_rows = set(member_map.get(int(selected_bucket_id), []))
        bucket_mask = events.index.to_series().isin(bucket_rows)

        if subset_mode == "unexplained_target_winners":
            mask = (events["true_member"] == target_member) & (~bucket_mask)
            label = f"Pass 2 - Unexplained {member_label(target_member)} winners after bucket {selected_bucket_id}"
        elif subset_mode == "false_positives":
            mask = bucket_mask & (events["true_member"] != target_member)
            label = f"Pass 2 - False positives for {member_label(target_member)} bucket {selected_bucket_id}"
        elif subset_mode == "covered_rows":
            mask = bucket_mask
            label = f"Pass 2 - Rows covered by {member_label(target_member)} bucket {selected_bucket_id}"
        elif subset_mode == "bucket_error_rows":
            mask = ((events["true_member"] == target_member) & (~bucket_mask)) | (bucket_mask & (events["true_member"] != target_member))
            label = f"Pass 2 - Error rows for {member_label(target_member)} bucket {selected_bucket_id}"
        else:
            raise ValueError(f"Unsupported subset_mode: {subset_mode}")

    sub = minimal_event_df_from_events(events[mask].copy())
    if len(sub) == 0:
        raise ValueError("The selected second-pass subset contains 0 rows.")
    return sub, label


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return dedupe_columns(df).to_csv(index=False).encode("utf-8")


def has_streamlit_context() -> bool:
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def safe_display_df(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    return dedupe_columns(df).head(int(rows)).copy()


def init_session_state() -> None:
    defaults = {
        "pass1_results": None,
        "pass2_results": None,
        "uploaded_name": None,
        "raw_preview_df": None,
        "pass1_signature": None,
        "pass2_signature": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def build_signature(
    uploaded_name: str,
    raw_shape: Tuple[int, int],
    min_support: int,
    bucket_min_support: int,
    bucket_top_k: int,
    bucket_max_depth: int,
    mine_level: str,
    objective: str,
    pass_mode: str,
    target_member: int,
    selected_bucket_id: str,
) -> Tuple:
    return (
        uploaded_name,
        raw_shape,
        int(min_support),
        int(bucket_min_support),
        int(bucket_top_k),
        int(bucket_max_depth),
        str(mine_level),
        str(objective),
        str(pass_mode),
        int(target_member),
        str(selected_bucket_id),
    )


def run_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to per-event CSV/XLSX/TXT")
    ap.add_argument("--min-support", type=int, default=8)
    ap.add_argument("--bucket-min-support", type=int, default=6)
    ap.add_argument("--bucket-top-k", type=int, default=80)
    ap.add_argument("--bucket-max-depth", type=int, default=4)
    ap.add_argument("--mine-level", choices=["standard", "expanded"], default="standard")
    ap.add_argument("--objective", choices=["positive_buckets", "separators", "rescue_pockets", "anti_buckets"], default="positive_buckets")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    raw = read_table_from_path(in_path)
    results = run_mining(
        raw,
        min_support=args.min_support,
        bucket_min_support=args.bucket_min_support,
        bucket_top_k=args.bucket_top_k,
        bucket_max_depth=args.bucket_max_depth,
        mine_level=args.mine_level,
        pass_label="Pass 1 - Full dataset",
        objective=args.objective,
    )

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = in_path.parent / f"{in_path.stem}__deep_trait_miner_outputs__{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dedupe_columns(results["events"]).to_csv(out_dir / f"{in_path.stem}__feature_table.csv", index=False)
    dedupe_columns(results["all_scores"]).to_csv(out_dir / f"{in_path.stem}__trait_scores_all_members.csv", index=False)

    for member in MEMBERS:
        dedupe_columns(results["per_member_scores"][member]).to_csv(out_dir / f"{in_path.stem}__trait_scores_member_{member_label(member)}.csv", index=False)
        dedupe_columns(results["per_member_separators"][member]).to_csv(out_dir / f"{in_path.stem}__separator_candidates_member_{member_label(member)}.csv", index=False)
        dedupe_columns(results["per_member_buckets"][member]).to_csv(out_dir / f"{in_path.stem}__bucket_candidates_member_{member_label(member)}.csv", index=False)

    (out_dir / f"{in_path.stem}__summary.txt").write_text(results["summary_text"], encoding="utf-8")
    print(f"Done. Output folder: {out_dir}")


def render_result_block(results: Dict[str, object], block_name: str, top_rows: int, objective: str) -> None:
    detected_cols = results["detected_cols"]
    events = results["events"]
    all_scores = results["all_scores"]

    st.markdown(f"## {block_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usable rows", f"{len(events):,}")
    c2.metric("Traits generated", f"{results['trait_df'].shape[1]:,}")
    c3.metric("Feature columns", f"{results['features_only'].shape[1]:,}")
    c4.metric("Scored rows", f"{len(all_scores):,}")

    st.caption(
        f"Completed at UTC: {results.get('completed_at_utc', '')} | "
        f"Pass label: {results.get('pass_label', '')} | "
        f"Mine level: {results.get('mine_level', '')} | "
        f"Objective: {results.get('objective', objective)}"
    )

    st.subheader("Detected columns")
    st.json(detected_cols)

    st.subheader("Summary")
    st.text_area(f"Summary text - {block_name}", results["summary_text"], height=350, key=f"summary_{block_name}")

    st.download_button(
        f"Download summary TXT - {block_name}",
        data=results["summary_text"].encode("utf-8"),
        file_name=f"{block_name.lower().replace(' ', '_')}__core025_deep_trait_miner_summary__2026-03-24.txt",
        mime="text/plain",
        key=f"dl_summary_{block_name}",
    )

    st.subheader("Feature table")
    st.dataframe(safe_display_df(events, int(top_rows)), use_container_width=True)
    st.download_button(
        f"Download feature table CSV - {block_name}",
        data=df_to_csv_bytes(events),
        file_name=f"{block_name.lower().replace(' ', '_')}__core025_feature_table__2026-03-24.csv",
        mime="text/csv",
        key=f"dl_features_{block_name}",
    )

    st.subheader("All-member trait scores")
    st.dataframe(safe_display_df(all_scores, int(top_rows)), use_container_width=True)
    st.download_button(
        f"Download all-member trait scores CSV - {block_name}",
        data=df_to_csv_bytes(all_scores),
        file_name=f"{block_name.lower().replace(' ', '_')}__core025_trait_scores_all_members__2026-03-24.csv",
        mime="text/csv",
        key=f"dl_allscores_{block_name}",
    )

    for member in MEMBERS:
        st.markdown(f"### Member {member_label(member)}")
        scores = results["per_member_scores"][member]
        seps = results["per_member_separators"][member]
        buckets = results["per_member_buckets"][member]

        tab1, tab2, tab3 = st.tabs(["Top single traits", "Separator traits", "Stacked buckets"])

        with tab1:
            st.dataframe(safe_display_df(scores, int(top_rows)), use_container_width=True)
            st.download_button(
                f"Download {member_label(member)} single traits CSV - {block_name}",
                data=df_to_csv_bytes(scores),
                file_name=f"{block_name.lower().replace(' ', '_')}__core025_trait_scores_member_{member_label(member)}__2026-03-24.csv",
                mime="text/csv",
                key=f"dl_scores_{block_name}_{member}",
            )

        with tab2:
            st.dataframe(safe_display_df(seps, int(top_rows)), use_container_width=True)
            st.download_button(
                f"Download {member_label(member)} separators CSV - {block_name}",
                data=df_to_csv_bytes(seps),
                file_name=f"{block_name.lower().replace(' ', '_')}__core025_separator_candidates_member_{member_label(member)}__2026-03-24.csv",
                mime="text/csv",
                key=f"dl_seps_{block_name}_{member}",
            )

        with tab3:
            st.dataframe(safe_display_df(buckets, int(top_rows)), use_container_width=True)
            st.download_button(
                f"Download {member_label(member)} buckets CSV - {block_name}",
                data=df_to_csv_bytes(buckets),
                file_name=f"{block_name.lower().replace(' ', '_')}__core025_bucket_candidates_member_{member_label(member)}__2026-03-24.csv",
                mime="text/csv",
                key=f"dl_buckets_{block_name}_{member}",
            )


def run_streamlit_app() -> None:
    st.set_page_config(page_title="Core 025 Deep Trait Miner", layout="wide")
    init_session_state()

    st.title("Core 025 Deep Trait Miner")
    st.caption(
        "Upload a real per-event file. Run a first pass on the full set, then use the second-pass "
        "controls to re-mine unexplained winners, false positives, regimes, or member-only subsets."
    )

    with st.sidebar:
        st.header("Global mining settings")
        min_support = st.number_input("Separator min support", min_value=1, value=8, step=1)
        bucket_min_support = st.number_input("Bucket min support", min_value=1, value=6, step=1)
        bucket_top_k = st.number_input("Bucket top K traits", min_value=10, value=80, step=5)
        bucket_max_depth = st.number_input("Bucket max depth", min_value=1, value=4, step=1)
        top_rows = st.number_input("Rows to display per table", min_value=5, value=25, step=5)
        mine_level = st.selectbox("Mine level", options=["standard", "expanded"], index=1)
        objective = st.selectbox(
            "Objective selector",
            options=["positive_buckets", "separators", "rescue_pockets", "anti_buckets"],
            index=0,
        )

        st.header("Second-pass controls")
        target_member = st.selectbox("Target member", options=MEMBERS, format_func=member_label, index=2)
        pass_mode = st.selectbox(
            "Pass 2 subset mode",
            options=[
                "full_dataset_again",
                "member_only_full",
                "no9_regime",
                "has9_regime",
                "pair_only_regime",
                "all_unique_regime",
                "unexplained_target_winners",
                "false_positives",
                "covered_rows",
                "bucket_error_rows",
            ],
            format_func=lambda x: {
                "full_dataset_again": "Full dataset again",
                "member_only_full": "Member-only full rows",
                "no9_regime": "No-9 regime",
                "has9_regime": "Has-9 regime",
                "pair_only_regime": "Pair-or-more regime",
                "all_unique_regime": "All-unique regime",
                "unexplained_target_winners": "Unexplained target winners",
                "false_positives": "False positives from selected bucket",
                "covered_rows": "Rows covered by selected bucket",
                "bucket_error_rows": "Bucket error rows",
            }[x],
            index=6,
        )

        if st.button("Clear stored results", key="clear_results_btn"):
            for key in ["pass1_results", "pass2_results", "uploaded_name", "raw_preview_df", "pass1_signature", "pass2_signature"]:
                st.session_state[key] = None
            st.rerun()

    uploaded = st.file_uploader(
        "Upload per-event CSV / TXT / TSV / XLSX",
        type=["csv", "txt", "tsv", "xlsx", "xls"],
        key="per_event_uploader",
    )

    if uploaded is None:
        st.info("Upload a file to begin.")
        return

    try:
        df_raw = dedupe_columns(read_uploaded_file(uploaded))
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        return

    st.session_state["uploaded_name"] = uploaded.name
    st.session_state["raw_preview_df"] = df_raw

    st.subheader("Raw file preview")
    st.write(f"File: {uploaded.name}")
    st.write(f"Rows: {len(df_raw):,} | Columns: {len(df_raw.columns)}")
    st.dataframe(safe_display_df(df_raw, 10), use_container_width=True)

    bucket_options = ["(none)"]
    pass1_results = st.session_state.get("pass1_results")
    if pass1_results is not None:
        target_buckets = pass1_results["per_member_buckets"].get(target_member, pd.DataFrame())
        if target_buckets is not None and not target_buckets.empty:
            for _, row in target_buckets.head(100).iterrows():
                bucket_options.append(
                    f"{int(row['bucket_id'])} | support={int(row['support'])} | hit_rate={row['hit_rate']:.3f} | {row['bucket_traits']}"
                )

    selected_bucket_text = st.selectbox("Selected first-pass bucket for Pass 2", options=bucket_options, index=0)
    selected_bucket_id = None
    if selected_bucket_text != "(none)":
        selected_bucket_id = int(str(selected_bucket_text).split("|", 1)[0].strip())

    st.markdown("## Pass 1")
    pass1_sig = build_signature(
        uploaded_name=uploaded.name,
        raw_shape=df_raw.shape,
        min_support=int(min_support),
        bucket_min_support=int(bucket_min_support),
        bucket_top_k=int(bucket_top_k),
        bucket_max_depth=int(bucket_max_depth),
        mine_level=mine_level,
        objective=objective,
        pass_mode="pass1_full",
        target_member=target_member,
        selected_bucket_id=str(selected_bucket_id),
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        run_pass1 = st.button("Run Pass 1", type="primary", key="run_pass1_btn")
    with col2:
        if st.session_state["pass1_results"] is not None:
            st.caption("Pass 1 results are stored in session state and survive downloads.")

    if run_pass1:
        try:
            with st.spinner("Running Pass 1 on the full dataset..."):
                results = run_mining(
                    df_raw,
                    min_support=int(min_support),
                    bucket_min_support=int(bucket_min_support),
                    bucket_top_k=int(bucket_top_k),
                    bucket_max_depth=int(bucket_max_depth),
                    mine_level=mine_level,
                    pass_label="Pass 1 - Full dataset",
                    objective=objective,
                )
            st.session_state["pass1_results"] = results
            st.session_state["pass1_signature"] = pass1_sig
            st.session_state["pass2_results"] = None
            st.session_state["pass2_signature"] = None
            st.rerun()
        except Exception as e:
            st.exception(e)
            return

    pass1_results = st.session_state.get("pass1_results")
    if pass1_results is None:
        st.info("Click 'Run Pass 1' to generate the first-pass results.")
        return

    if st.session_state.get("pass1_signature") != pass1_sig:
        st.warning("Current settings differ from stored Pass 1 results. Click 'Run Pass 1' again to refresh them.")

    render_result_block(pass1_results, "Pass 1 Results", int(top_rows), objective)

    st.markdown("---")
    st.markdown("## Pass 2")

    needs_bucket = pass_mode in {"unexplained_target_winners", "false_positives", "covered_rows", "bucket_error_rows"}
    if needs_bucket and selected_bucket_id is None:
        st.info("Select a first-pass bucket above to run this second-pass mode.")
        return

    pass2_sig = build_signature(
        uploaded_name=uploaded.name,
        raw_shape=df_raw.shape,
        min_support=int(min_support),
        bucket_min_support=int(bucket_min_support),
        bucket_top_k=int(bucket_top_k),
        bucket_max_depth=int(bucket_max_depth),
        mine_level=mine_level,
        objective=objective,
        pass_mode=pass_mode,
        target_member=target_member,
        selected_bucket_id=str(selected_bucket_id),
    )

    col3, col4 = st.columns([1, 3])
    with col3:
        run_pass2 = st.button("Run Pass 2", type="primary", key="run_pass2_btn")
    with col4:
        if st.session_state["pass2_results"] is not None:
            st.caption("Pass 2 results are also stored in session state and survive downloads.")

    if run_pass2:
        try:
            with st.spinner("Building second-pass subset and running mining..."):
                subset_df, pass_label = build_subset_from_results(
                    base_results=pass1_results,
                    subset_mode=pass_mode,
                    target_member=int(target_member),
                    selected_bucket_id=selected_bucket_id,
                )
                pass2_results = run_mining(
                    subset_df,
                    min_support=int(min_support),
                    bucket_min_support=int(bucket_min_support),
                    bucket_top_k=int(bucket_top_k),
                    bucket_max_depth=int(bucket_max_depth),
                    mine_level=mine_level,
                    pass_label=pass_label,
                    objective=objective,
                )
            st.session_state["pass2_results"] = pass2_results
            st.session_state["pass2_signature"] = pass2_sig
            st.rerun()
        except Exception as e:
            st.exception(e)
            return

    pass2_results = st.session_state.get("pass2_results")
    if pass2_results is None:
        st.info("Choose a second-pass mode and click 'Run Pass 2' to mine the next layer.")
        return

    if st.session_state.get("pass2_signature") != pass2_sig:
        st.warning("Current Pass 2 settings differ from the stored Pass 2 results. Click 'Run Pass 2' again to refresh them.")

    render_result_block(pass2_results, "Pass 2 Results", int(top_rows), objective)


if __name__ == "__main__":
    if has_streamlit_context():
        run_streamlit_app()
    else:
        run_cli()
""",
    "member": r"""#!/usr/bin/env python3
# core025_member_trait_miner_v2_strict_fix1__2026-03-28.py
#
# BUILD: core025_member_trait_miner_v2_strict_fix1__2026-03-28
#
# Full file. No placeholders.
# Fixes KeyError: 'year_month' by preventing merge collisions and making stability checks resilient.

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]
BUILD_MARKER = "BUILD: core025_member_trait_miner_v2_strict_fix1__2026-03-28"


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_display_df(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    return df.head(int(rows)).copy()


def load_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = uploaded_file.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t", header=None)
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=None)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def norm_result(r: object) -> Optional[str]:
    d = re.findall(r"\d", str(r))
    return "".join(d[:4]) if len(d) >= 4 else None


def to_member(r4: Optional[str]) -> Optional[str]:
    if r4 is None:
        return None
    s = "".join(sorted(r4))
    return s if s in CORE025 else None


def sum_bucket(x: int) -> str:
    if x <= 9:
        return "sum_00_09"
    if x <= 13:
        return "sum_10_13"
    if x <= 17:
        return "sum_14_17"
    if x <= 21:
        return "sum_18_21"
    return "sum_22_plus"


def spread_bucket(x: int) -> str:
    if x <= 2:
        return "spread_0_2"
    if x <= 4:
        return "spread_3_4"
    if x <= 6:
        return "spread_5_6"
    return "spread_7_plus"


def pair_token_pattern(digs: List[int]) -> str:
    tokens = []
    for i in range(4):
        for j in range(i + 1, 4):
            tokens.append("".join(sorted([str(digs[i]), str(digs[j])])))
    return "|".join(sorted(tokens))


def structure_label(digs: List[int]) -> str:
    counts = sorted(Counter(digs).values(), reverse=True)
    if counts == [1, 1, 1, 1]:
        return "ABCD"
    if counts == [2, 1, 1]:
        return "AABC"
    if counts == [2, 2]:
        return "AABB"
    if counts == [3, 1]:
        return "AAAB"
    if counts == [4]:
        return "AAAA"
    return "OTHER"


def features(seed: object) -> Optional[Dict[str, object]]:
    if seed is None:
        return None
    d = re.findall(r"\d", str(seed))
    if len(d) < 4:
        return None
    digs = [int(x) for x in d[:4]]
    cnt = Counter(digs)
    unique_sorted = sorted(set(digs))
    consec_links = 0
    for a, b in zip(unique_sorted[:-1], unique_sorted[1:]):
        if b - a == 1:
            consec_links += 1
    s = sum(digs)
    spread = max(digs) - min(digs)
    feat = {
        "sum": s,
        "sum_bucket": sum_bucket(s),
        "spread": spread,
        "spread_bucket": spread_bucket(spread),
        "even": sum(x % 2 == 0 for x in digs),
        "odd": sum(x % 2 != 0 for x in digs),
        "high": sum(x >= 5 for x in digs),
        "low": sum(x <= 4 for x in digs),
        "unique": len(set(digs)),
        "pair": int(len(set(digs)) < 4),
        "max_rep": max(cnt.values()),
        "sorted_seed": "".join(map(str, sorted(digs))),
        "first2": f"{digs[0]}{digs[1]}",
        "last2": f"{digs[2]}{digs[3]}",
        "consec_links": consec_links,
        "parity_pattern": "".join("E" if x % 2 == 0 else "O" for x in digs),
        "highlow_pattern": "".join("H" if x >= 5 else "L" for x in digs),
        "pair_token_pattern": pair_token_pattern(digs),
        "structure": structure_label(digs),
    }
    for k in range(10):
        feat[f"has{k}"] = int(k in cnt)
        feat[f"cnt{k}"] = int(cnt.get(k, 0))
    return feat


def miner_feature_columns() -> List[str]:
    return [
        "sum_bucket", "spread_bucket", "even", "odd", "high", "low", "unique", "pair",
        "max_rep", "sorted_seed", "first2", "last2", "consec_links",
        "parity_pattern", "highlow_pattern", "pair_token_pattern", "structure",
    ] + [f"has{k}" for k in range(10)] + [f"cnt{k}" for k in range(10)]


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df.columns) == 4:
        df.columns = ["date", "jurisdiction", "game", "result"]
    else:
        cols = [str(c).lower() for c in df.columns]
        df.columns = cols
        rename_map = {}
        if "date" not in df.columns:
            for c in df.columns:
                if "date" in c:
                    rename_map[c] = "date"; break
        if "jurisdiction" not in df.columns:
            for c in df.columns:
                if "jurisdiction" in c or "state" in c:
                    rename_map[c] = "jurisdiction"; break
        if "game" not in df.columns:
            for c in df.columns:
                if "game" in c or "stream" in c:
                    rename_map[c] = "game"; break
        if "result" not in df.columns:
            for c in df.columns:
                if "result" in c:
                    rename_map[c] = "result"; break
        df = df.rename(columns=rename_map)
        needed = {"date", "jurisdiction", "game", "result"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["r4"] = df["result"].apply(norm_result)
    df["member"] = df["r4"].apply(to_member)
    df["stream"] = df["jurisdiction"].astype(str) + "|" + df["game"].astype(str)
    df = df.dropna(subset=["date", "r4"]).reset_index(drop=True)

    feat_series = df["r4"].apply(features)
    valid_mask = feat_series.notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    feat_df = feat_series.loc[valid_mask].apply(pd.Series).reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    return df


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream, g in df.groupby("stream"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(1, len(g)):
            seed = g.loc[i - 1, "r4"]
            next_member = g.loc[i, "member"]
            next_r4 = g.loc[i, "r4"]
            feat = features(seed)
            if feat is None:
                continue
            rows.append({
                "stream": stream,
                "jurisdiction": g.loc[i, "jurisdiction"],
                "game": g.loc[i, "game"],
                "seed_date": g.loc[i - 1, "date"],
                "event_date": g.loc[i, "date"],
                "year_month": g.loc[i, "date"].to_period("M").strftime("%Y-%m"),
                "seed": seed,
                "next_r4": next_r4,
                "next_member": next_member,
                "is_core025_hit": int(next_member is not None),
                **feat,
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["event_date", "stream", "seed"]).reset_index(drop=True)


def transition_maps(prior_transitions: pd.DataFrame):
    exact_seed_map = {}
    sorted_seed_map = {}
    stream_member_map = {}
    global_member_map = Counter()
    for _, r in prior_transitions.iterrows():
        member = r["next_member"]
        if member is None or pd.isna(member):
            continue
        exact_seed_map.setdefault(str(r["seed"]), Counter())[member] += 1
        sorted_seed_map.setdefault(str(r["sorted_seed"]), Counter())[member] += 1
        stream_member_map.setdefault(str(r["stream"]), Counter())[member] += 1
        global_member_map[member] += 1
    return exact_seed_map, sorted_seed_map, stream_member_map, global_member_map


def counter_to_probs(c: Counter) -> Dict[str, float]:
    total = sum(c.values())
    if total <= 0:
        return {m: 1 / 3 for m in CORE025}
    return {m: c.get(m, 0) / total for m in CORE025}


def similarity(a: Dict[str, object], b: pd.Series) -> float:
    score = 0.0
    if a["sum_bucket"] == b["sum_bucket"]:
        score += 3
    if a["spread_bucket"] == b["spread_bucket"]:
        score += 2
    if a["parity_pattern"] == b["parity_pattern"]:
        score += 2
    if a["highlow_pattern"] == b["highlow_pattern"]:
        score += 2
    if a["structure"] == b["structure"]:
        score += 3
    if a["sorted_seed"] == b["sorted_seed"]:
        score += 6
    if a["pair_token_pattern"] == b["pair_token_pattern"]:
        score += 4
    if a["unique"] == b["unique"]:
        score += 1
    if a["max_rep"] == b["max_rep"]:
        score += 1
    for k in range(10):
        if a[f"has{k}"] == b[f"has{k}"]:
            score += 0.2
        if a[f"cnt{k}"] == b[f"cnt{k}"]:
            score += 0.1
    return float(score)


def score_seed_v3(seed: str, stream: str, prior_transitions: pd.DataFrame, min_stream_history: int = 20) -> List[tuple[str, float]]:
    seed_feat = features(seed)
    if seed_feat is None:
        return [(m, 1 / 3) for m in CORE025]
    exact_seed_map, sorted_seed_map, stream_member_map, global_member_map = transition_maps(prior_transitions)

    score_accum = {m: 0.0 for m in CORE025}
    global_probs = counter_to_probs(global_member_map)
    for m in CORE025:
        score_accum[m] += global_probs[m] * 0.25

    if stream is not None and str(stream) in stream_member_map and sum(stream_member_map[str(stream)].values()) >= int(min_stream_history):
        stream_probs = counter_to_probs(stream_member_map[str(stream)])
        for m in CORE025:
            score_accum[m] += stream_probs[m] * 1.20

    if str(seed) in exact_seed_map:
        exact_probs = counter_to_probs(exact_seed_map[str(seed)])
        for m in CORE025:
            score_accum[m] += exact_probs[m] * 1.50

    sorted_key = str(seed_feat["sorted_seed"])
    if sorted_key in sorted_seed_map:
        sorted_probs = counter_to_probs(sorted_seed_map[sorted_key])
        for m in CORE025:
            score_accum[m] += sorted_probs[m] * 1.10

    pool = prior_transitions.copy()
    if stream is not None:
        stream_subset = prior_transitions[prior_transitions["stream"] == str(stream)].copy()
        if len(stream_subset) >= int(min_stream_history):
            pool = stream_subset

    if len(pool):
        pool = pool.sort_values("event_date").reset_index(drop=True)
        sim_scores = {m: 0.0 for m in CORE025}
        total_sim = 0.0
        for _, r in pool.iterrows():
            member = r["next_member"]
            if member is None or pd.isna(member):
                continue
            sim = similarity(seed_feat, r)
            if sim <= 0:
                continue
            sim_scores[member] += sim
            total_sim += sim
        if total_sim > 0:
            for m in CORE025:
                score_accum[m] += (sim_scores[m] / total_sim) * 1.80

    total = sum(score_accum.values())
    if total <= 0:
        return [(m, 1 / 3) for m in CORE025]
    probs = {m: score_accum[m] / total for m in CORE025}
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)


def classify_score_tier(top1_score: float, top1_only_score_threshold: float, top1_top2_score_threshold: float) -> str:
    if top1_score >= float(top1_only_score_threshold):
        return "Top1 only"
    if top1_score >= float(top1_top2_score_threshold):
        return "Top1 + Top2"
    return "Skip member play"


def build_hit_event_predictions(transitions: pd.DataFrame, min_global_history: int, min_stream_history: int,
                                top1_only_score_threshold: float, top1_top2_score_threshold: float) -> pd.DataFrame:
    rows = []
    for i in range(len(transitions)):
        current = transitions.iloc[i]
        if int(current["is_core025_hit"]) != 1:
            continue
        prior = transitions.iloc[:i].copy()
        if len(prior) < int(min_global_history):
            continue
        ranked = score_seed_v3(
            seed=str(current["seed"]),
            stream=str(current["stream"]),
            prior_transitions=prior,
            min_stream_history=int(min_stream_history),
        )
        top1, top1_score = ranked[0]
        top2, top2_score = ranked[1]
        top3, top3_score = ranked[2]
        recommendation = classify_score_tier(top1_score, float(top1_only_score_threshold), float(top1_top2_score_threshold))
        actual_member = current["next_member"] if pd.notna(current["next_member"]) else ""
        rows.append({
            "event_date": current["event_date"],
            "year_month": current["year_month"],
            "stream": current["stream"],
            "seed": current["seed"],
            "actual_member": actual_member,
            "Top1": top1,
            "Top1_score": top1_score,
            "Top2": top2,
            "Top2_score": top2_score,
            "Top3": top3,
            "Top3_score": top3_score,
            "Top1_minus_Top2": top1_score - top2_score,
            "recommendation": recommendation,
            "top1_hit": int(actual_member == top1),
            "top2_hit": int(actual_member in [top1, top2]),
            "member_play_hit": int(
                (recommendation == "Top1 only" and actual_member == top1) or
                (recommendation == "Top1 + Top2" and actual_member in [top1, top2])
            ),
        })
    return pd.DataFrame(rows)


def stability_stats(sub: pd.DataFrame) -> Dict[str, int]:
    ym_col = next((c for c in ["year_month", "year_month_x", "year_month_y"] if c in sub.columns), None)
    if ym_col is None:
        raise KeyError("No year_month column found in subset.")
    return {"stream_count": int(sub["stream"].nunique()), "month_count": int(sub[ym_col].nunique())}


def build_member_separation_traits(core_hits: pd.DataFrame, min_support: int, preferred_support: int, min_dom_rate: float,
                                   min_gap: float, min_streams: int, min_months: int) -> pd.DataFrame:
    rows = []
    for col in miner_feature_columns():
        if col not in core_hits.columns:
            continue
        vals = core_hits[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for val in vals:
            sub = core_hits[core_hits[col] == val].copy()
            support = len(sub)
            if support < int(min_support):
                continue
            counts = sub["next_member"].value_counts()
            probs = {m: counts.get(m, 0) / support for m in CORE025}
            winner = max(probs, key=probs.get)
            winner_rate = probs[winner]
            second_rate = sorted(probs.values(), reverse=True)[1]
            separation_gap = winner_rate - second_rate
            stable = stability_stats(sub)
            if winner_rate < float(min_dom_rate) or separation_gap < float(min_gap):
                continue
            if stable["stream_count"] < int(min_streams) or stable["month_count"] < int(min_months):
                continue
            rows.append({
                "trait": f"{col}={val}",
                "support": support,
                "preferred_support_met": int(support >= int(preferred_support)),
                "winning_member": winner,
                "winning_member_rate": winner_rate,
                "second_best_rate": second_rate,
                "separation_gap": separation_gap,
                "stream_count": stable["stream_count"],
                "month_count": stable["month_count"],
                "rate_0025": probs["0025"],
                "rate_0225": probs["0225"],
                "rate_0255": probs["0255"],
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["preferred_support_met", "winning_member_rate", "separation_gap", "support", "stream_count", "month_count"],
            ascending=[False, False, False, False, False, False]
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "trait", "support", "preferred_support_met", "winning_member", "winning_member_rate",
            "second_best_rate", "separation_gap", "stream_count", "month_count",
            "rate_0025", "rate_0225", "rate_0255"
        ])
    return out


def build_member_specific_traits(core_hits: pd.DataFrame, min_support: int, preferred_support: int, target_member: str,
                                 min_member_rate: float, min_streams: int, min_months: int) -> pd.DataFrame:
    rows = []
    for col in miner_feature_columns():
        if col not in core_hits.columns:
            continue
        vals = core_hits[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for val in vals:
            sub = core_hits[core_hits[col] == val].copy()
            support = len(sub)
            if support < int(min_support):
                continue
            rate = float((sub["next_member"] == target_member).mean())
            stable = stability_stats(sub)
            if rate < float(min_member_rate):
                continue
            if stable["stream_count"] < int(min_streams) or stable["month_count"] < int(min_months):
                continue
            rows.append({
                "trait": f"{col}={val}",
                "support": support,
                "preferred_support_met": int(support >= int(preferred_support)),
                "target_member": target_member,
                "target_member_rate": rate,
                "non_target_rate": 1.0 - rate,
                "stream_count": stable["stream_count"],
                "month_count": stable["month_count"],
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["preferred_support_met", "target_member_rate", "support", "stream_count", "month_count"],
            ascending=[False, False, False, False, False]
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "trait", "support", "preferred_support_met", "target_member", "target_member_rate",
            "non_target_rate", "stream_count", "month_count"
        ])
    return out


def build_top2_needed_traits(pred_hits: pd.DataFrame, transitions: pd.DataFrame, min_support: int, preferred_support: int,
                             min_top2_rate: float, min_streams: int, min_months: int) -> pd.DataFrame:
    pockets = pred_hits[(pred_hits["top1_hit"] == 0) & (pred_hits["top2_hit"] == 1)].copy()
    if len(pockets) == 0:
        return pd.DataFrame(columns=["trait", "support_top2_needed", "hit_event_support", "top2_needed_rate"])
    trait_source = transitions[["event_date", "stream"] + [c for c in miner_feature_columns() if c in transitions.columns]].copy()
    data = pockets.merge(trait_source, on=["event_date", "stream"], how="left")
    rows = []
    cols = [c for c in miner_feature_columns() if c in data.columns]
    for col in cols:
        vals = data[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for val in vals:
            sub = data[data[col] == val].copy()
            support = len(sub)
            if support < int(min_support):
                continue
            denom_df = transitions[(transitions["is_core025_hit"] == 1) & (transitions[col] == val)].copy()
            denom = len(denom_df)
            if denom < int(min_support):
                continue
            top2_needed_rate = support / denom
            stable = stability_stats(sub)
            if top2_needed_rate < float(min_top2_rate):
                continue
            if stable["stream_count"] < int(min_streams) or stable["month_count"] < int(min_months):
                continue
            rows.append({
                "trait": f"{col}={val}",
                "support_top2_needed": support,
                "preferred_support_met": int(support >= int(preferred_support)),
                "hit_event_support": denom,
                "top2_needed_rate": top2_needed_rate,
                "stream_count": stable["stream_count"],
                "month_count": stable["month_count"],
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["preferred_support_met", "top2_needed_rate", "support_top2_needed", "stream_count", "month_count"],
            ascending=[False, False, False, False, False]
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "trait", "support_top2_needed", "preferred_support_met", "hit_event_support", "top2_needed_rate",
            "stream_count", "month_count"
        ])
    return out


def build_skip_danger_traits(pred_hits: pd.DataFrame, transitions: pd.DataFrame, min_support: int, preferred_support: int,
                             min_skip_danger_rate: float, min_streams: int, min_months: int) -> pd.DataFrame:
    danger = pred_hits[pred_hits["recommendation"] == "Skip member play"].copy()
    if len(danger) == 0:
        return pd.DataFrame(columns=["trait", "support_skipped_hits", "hit_event_support", "skip_danger_rate"])
    trait_source = transitions[["event_date", "stream"] + [c for c in miner_feature_columns() if c in transitions.columns]].copy()
    data = danger.merge(trait_source, on=["event_date", "stream"], how="left")
    rows = []
    cols = [c for c in miner_feature_columns() if c in data.columns]
    for col in cols:
        vals = data[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for val in vals:
            sub = data[data[col] == val].copy()
            support = len(sub)
            if support < int(min_support):
                continue
            denom_df = transitions[(transitions["is_core025_hit"] == 1) & (transitions[col] == val)].copy()
            denom = len(denom_df)
            if denom < int(min_support):
                continue
            skip_danger_rate = support / denom
            stable = stability_stats(sub)
            if skip_danger_rate < float(min_skip_danger_rate):
                continue
            if stable["stream_count"] < int(min_streams) or stable["month_count"] < int(min_months):
                continue
            rows.append({
                "trait": f"{col}={val}",
                "support_skipped_hits": support,
                "preferred_support_met": int(support >= int(preferred_support)),
                "hit_event_support": denom,
                "skip_danger_rate": skip_danger_rate,
                "stream_count": stable["stream_count"],
                "month_count": stable["month_count"],
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["preferred_support_met", "skip_danger_rate", "support_skipped_hits", "stream_count", "month_count"],
            ascending=[False, False, False, False, False]
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "trait", "support_skipped_hits", "preferred_support_met", "hit_event_support", "skip_danger_rate",
            "stream_count", "month_count"
        ])
    return out


def main():
    st.set_page_config(page_title="Core025 Member Trait Miner v2 Strict Fix1", layout="wide")
    st.title("Core025 Member Trait Miner v2 Strict Fix1")
    st.caption("Strict miner with stronger support/stability filters and year_month merge fix.")
    st.code(BUILD_MARKER, language="text")

    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        st.header("Strict trait filters")
        min_support = st.number_input("Minimum support", min_value=10, value=25, step=1)
        preferred_support = st.number_input("Preferred support", min_value=10, value=40, step=1)
        min_dom_rate = st.slider("Minimum dominance rate for separation", min_value=0.34, max_value=0.95, value=0.60, step=0.01)
        min_gap = st.slider("Minimum separation gap", min_value=0.00, max_value=0.50, value=0.10, step=0.01)
        min_member_rate = st.slider("Minimum member-specific rate", min_value=0.34, max_value=0.95, value=0.60, step=0.01)
        min_top2_rate = st.slider("Minimum Top2-needed rate", min_value=0.05, max_value=0.95, value=0.25, step=0.01)
        min_skip_danger_rate = st.slider("Minimum skip-danger rate", min_value=0.05, max_value=0.95, value=0.25, step=0.01)
        min_streams = st.number_input("Minimum distinct streams", min_value=1, value=3, step=1)
        min_months = st.number_input("Minimum distinct months", min_value=1, value=2, step=1)

        st.header("Prediction-context controls")
        min_global_history = st.number_input("Minimum prior transitions before prediction mining", min_value=10, value=100, step=10)
        min_stream_history = st.number_input("Minimum stream-specific history", min_value=0, value=20, step=5)
        top1_only_score_threshold = st.slider("Top1-only score threshold", min_value=0.33, max_value=0.95, value=0.48, step=0.005)
        top1_top2_score_threshold = st.slider("Top1+Top2 score threshold", min_value=0.33, max_value=0.95, value=0.36, step=0.005)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=25, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="strict_trait_hist_fix1")
    if not hist_file:
        st.info("Upload the full history file to begin.")
        return

    try:
        hist = prepare_history(load_table(hist_file))
    except Exception as e:
        st.exception(e)
        return

    transitions = build_transitions(hist)
    core_hits = transitions[transitions["is_core025_hit"] == 1].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Transitions", f"{len(transitions):,}")
    c2.metric("Core025 hit events", f"{len(core_hits):,}")
    c3.metric("Core025 base rate", f"{transitions['is_core025_hit'].mean():.4f}")

    if st.button("Run Strict Trait Miner", type="primary"):
        try:
            with st.spinner("Mining strict traits..."):
                sep_traits = build_member_separation_traits(core_hits, int(min_support), int(preferred_support), float(min_dom_rate), float(min_gap), int(min_streams), int(min_months))
                traits_0025 = build_member_specific_traits(core_hits, int(min_support), int(preferred_support), "0025", float(min_member_rate), int(min_streams), int(min_months))
                traits_0225 = build_member_specific_traits(core_hits, int(min_support), int(preferred_support), "0225", float(min_member_rate), int(min_streams), int(min_months))
                traits_0255 = build_member_specific_traits(core_hits, int(min_support), int(preferred_support), "0255", float(min_member_rate), int(min_streams), int(min_months))
                pred_hits = build_hit_event_predictions(transitions, int(min_global_history), int(min_stream_history), float(top1_only_score_threshold), float(top1_top2_score_threshold))
                top2_needed = build_top2_needed_traits(pred_hits, transitions, int(min_support), int(preferred_support), float(min_top2_rate), int(min_streams), int(min_months))
                skip_danger = build_skip_danger_traits(pred_hits, transitions, int(min_support), int(preferred_support), float(min_skip_danger_rate), int(min_streams), int(min_months))
            st.session_state["strict_miner_results_fix1"] = {
                "sep_traits": sep_traits,
                "traits_0025": traits_0025,
                "traits_0225": traits_0225,
                "traits_0255": traits_0255,
                "pred_hits": pred_hits,
                "top2_needed": top2_needed,
                "skip_danger": skip_danger,
            }
        except Exception as e:
            st.exception(e)
            return

    if "strict_miner_results_fix1" not in st.session_state or st.session_state["strict_miner_results_fix1"] is None:
        return

    results = st.session_state["strict_miner_results_fix1"]
    st.subheader("Strict separation traits")
    st.dataframe(safe_display_df(results["sep_traits"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict separation traits CSV", df_to_csv_bytes(results["sep_traits"]), "core025_member_trait_miner_v2_strict_fix1_separation_traits__2026-03-28.csv", "text/csv")

    st.subheader("0025 member-specific traits")
    st.dataframe(safe_display_df(results["traits_0025"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict 0025 traits CSV", df_to_csv_bytes(results["traits_0025"]), "core025_member_trait_miner_v2_strict_fix1_0025_traits__2026-03-28.csv", "text/csv")

    st.subheader("0225 member-specific traits")
    st.dataframe(safe_display_df(results["traits_0225"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict 0225 traits CSV", df_to_csv_bytes(results["traits_0225"]), "core025_member_trait_miner_v2_strict_fix1_0225_traits__2026-03-28.csv", "text/csv")

    st.subheader("0255 member-specific traits")
    st.dataframe(safe_display_df(results["traits_0255"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict 0255 traits CSV", df_to_csv_bytes(results["traits_0255"]), "core025_member_trait_miner_v2_strict_fix1_0255_traits__2026-03-28.csv", "text/csv")

    st.subheader("Strict Top2-needed traits")
    st.dataframe(safe_display_df(results["top2_needed"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict Top2-needed traits CSV", df_to_csv_bytes(results["top2_needed"]), "core025_member_trait_miner_v2_strict_fix1_top2_needed_traits__2026-03-28.csv", "text/csv")

    st.subheader("Strict skip-danger traits")
    st.dataframe(safe_display_df(results["skip_danger"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict skip-danger traits CSV", df_to_csv_bytes(results["skip_danger"]), "core025_member_trait_miner_v2_strict_fix1_skip_danger_traits__2026-03-28.csv", "text/csv")

    st.subheader("Historical hit-event prediction table")
    st.dataframe(safe_display_df(results["pred_hits"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict hit-event prediction table CSV", df_to_csv_bytes(results["pred_hits"]), "core025_member_trait_miner_v2_strict_fix1_hit_event_predictions__2026-03-28.csv", "text/csv")


if __name__ == "__main__":
    if "strict_miner_results_fix1" not in st.session_state:
        st.session_state["strict_miner_results_fix1"] = None
    main()
""",
    "pairwise": r"""#!/usr/bin/env python3
# core025_pairwise_separator_miner_v1__2026-03-28.py
#
# BUILD: core025_pairwise_separator_miner_v1__2026-03-28

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]
BUILD_MARKER = "BUILD: core025_pairwise_separator_miner_v1__2026-03-28"


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_display_df(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    return df.head(int(rows)).copy()


def load_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = uploaded_file.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t", header=None)
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=None)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def norm_result(r: object) -> Optional[str]:
    d = re.findall(r"\d", str(r))
    return "".join(d[:4]) if len(d) >= 4 else None


def to_member(r4: Optional[str]) -> Optional[str]:
    if r4 is None:
        return None
    s = "".join(sorted(r4))
    return s if s in CORE025 else None


def sum_bucket(x: int) -> str:
    if x <= 9:
        return "sum_00_09"
    if x <= 13:
        return "sum_10_13"
    if x <= 17:
        return "sum_14_17"
    if x <= 21:
        return "sum_18_21"
    return "sum_22_plus"


def spread_bucket(x: int) -> str:
    if x <= 2:
        return "spread_0_2"
    if x <= 4:
        return "spread_3_4"
    if x <= 6:
        return "spread_5_6"
    return "spread_7_plus"


def pair_token_pattern(digs: List[int]) -> str:
    tokens = []
    for i in range(4):
        for j in range(i + 1, 4):
            tokens.append("".join(sorted([str(digs[i]), str(digs[j])])))
    return "|".join(sorted(tokens))


def structure_label(digs: List[int]) -> str:
    counts = sorted(Counter(digs).values(), reverse=True)
    if counts == [1, 1, 1, 1]:
        return "ABCD"
    if counts == [2, 1, 1]:
        return "AABC"
    if counts == [2, 2]:
        return "AABB"
    if counts == [3, 1]:
        return "AAAB"
    if counts == [4]:
        return "AAAA"
    return "OTHER"


def features(seed: object) -> Optional[Dict[str, object]]:
    if seed is None:
        return None
    d = re.findall(r"\d", str(seed))
    if len(d) < 4:
        return None
    digs = [int(x) for x in d[:4]]
    cnt = Counter(digs)
    unique_sorted = sorted(set(digs))
    consec_links = 0
    for a, b in zip(unique_sorted[:-1], unique_sorted[1:]):
        if b - a == 1:
            consec_links += 1
    s = sum(digs)
    spread = max(digs) - min(digs)
    feat = {
        "sum": s,
        "sum_bucket": sum_bucket(s),
        "spread": spread,
        "spread_bucket": spread_bucket(spread),
        "even": sum(x % 2 == 0 for x in digs),
        "odd": sum(x % 2 != 0 for x in digs),
        "high": sum(x >= 5 for x in digs),
        "low": sum(x <= 4 for x in digs),
        "unique": len(set(digs)),
        "pair": int(len(set(digs)) < 4),
        "max_rep": max(cnt.values()),
        "sorted_seed": "".join(map(str, sorted(digs))),
        "first2": f"{digs[0]}{digs[1]}",
        "last2": f"{digs[2]}{digs[3]}",
        "consec_links": consec_links,
        "parity_pattern": "".join("E" if x % 2 == 0 else "O" for x in digs),
        "highlow_pattern": "".join("H" if x >= 5 else "L" for x in digs),
        "pair_token_pattern": pair_token_pattern(digs),
        "structure": structure_label(digs),
    }
    for k in range(10):
        feat[f"has{k}"] = int(k in cnt)
        feat[f"cnt{k}"] = int(cnt.get(k, 0))
    return feat


def miner_feature_columns() -> List[str]:
    return [
        "sum_bucket", "spread_bucket", "even", "odd", "high", "low", "unique",
        "pair", "max_rep", "sorted_seed", "first2", "last2", "consec_links",
        "parity_pattern", "highlow_pattern", "pair_token_pattern", "structure",
    ] + [f"has{k}" for k in range(10)] + [f"cnt{k}" for k in range(10)]


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df.columns) == 4:
        df.columns = ["date", "jurisdiction", "game", "result"]
    else:
        cols = [str(c).lower() for c in df.columns]
        df.columns = cols
        rename_map = {}
        if "date" not in df.columns:
            for c in df.columns:
                if "date" in c:
                    rename_map[c] = "date"; break
        if "jurisdiction" not in df.columns:
            for c in df.columns:
                if "jurisdiction" in c or "state" in c:
                    rename_map[c] = "jurisdiction"; break
        if "game" not in df.columns:
            for c in df.columns:
                if "game" in c or "stream" in c:
                    rename_map[c] = "game"; break
        if "result" not in df.columns:
            for c in df.columns:
                if "result" in c:
                    rename_map[c] = "result"; break
        df = df.rename(columns=rename_map)
        needed = {"date", "jurisdiction", "game", "result"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["r4"] = df["result"].apply(norm_result)
    df["member"] = df["r4"].apply(to_member)
    df["stream"] = df["jurisdiction"].astype(str) + "|" + df["game"].astype(str)
    df = df.dropna(subset=["date", "r4"]).reset_index(drop=True)
    feat_series = df["r4"].apply(features)
    valid_mask = feat_series.notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    feat_df = feat_series.loc[valid_mask].apply(pd.Series).reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    return df


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream, g in df.groupby("stream"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(1, len(g)):
            seed = g.loc[i - 1, "r4"]
            next_member = g.loc[i, "member"]
            feat = features(seed)
            if feat is None:
                continue
            rows.append({
                "stream": stream,
                "jurisdiction": g.loc[i, "jurisdiction"],
                "game": g.loc[i, "game"],
                "seed_date": g.loc[i - 1, "date"],
                "event_date": g.loc[i, "date"],
                "year_month": g.loc[i, "date"].to_period("M").strftime("%Y-%m"),
                "seed": seed,
                "next_member": next_member,
                "is_core025_hit": int(next_member is not None),
                **feat,
            })
    return pd.DataFrame(rows).sort_values(["event_date", "stream", "seed"]).reset_index(drop=True)


def stability_stats(sub: pd.DataFrame) -> Dict[str, int]:
    ym_col = next((c for c in ["year_month", "year_month_x", "year_month_y"] if c in sub.columns), None)
    if ym_col is None:
        raise KeyError("No year_month column found in subset.")
    return {"stream_count": int(sub["stream"].nunique()), "month_count": int(sub[ym_col].nunique())}


def build_pairwise_separator_traits(
    core_hits: pd.DataFrame, left_member: str, right_member: str, min_support: int,
    preferred_support: int, min_dom_rate: float, min_gap: float, min_streams: int, min_months: int
) -> pd.DataFrame:
    pair_df = core_hits[core_hits["next_member"].isin([left_member, right_member])].copy()
    rows = []
    for col in miner_feature_columns():
        if col not in pair_df.columns:
            continue
        vals = pair_df[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for val in vals:
            sub = pair_df[pair_df[col] == val].copy()
            support = len(sub)
            if support < int(min_support):
                continue
            left_count = int((sub["next_member"] == left_member).sum())
            right_count = int((sub["next_member"] == right_member).sum())
            left_rate = left_count / support
            right_rate = right_count / support
            winner = left_member if left_rate >= right_rate else right_member
            loser = right_member if winner == left_member else left_member
            winner_rate = max(left_rate, right_rate)
            loser_rate = min(left_rate, right_rate)
            gap = winner_rate - loser_rate
            stable = stability_stats(sub)
            if winner_rate < float(min_dom_rate):
                continue
            if gap < float(min_gap):
                continue
            if stable["stream_count"] < int(min_streams):
                continue
            if stable["month_count"] < int(min_months):
                continue
            rows.append({
                "pair": f"{left_member}_vs_{right_member}",
                "trait": f"{col}={val}",
                "support": support,
                "preferred_support_met": int(support >= int(preferred_support)),
                "winner_member": winner,
                "loser_member": loser,
                "winner_rate": winner_rate,
                "loser_rate": loser_rate,
                "pair_gap": gap,
                "stream_count": stable["stream_count"],
                "month_count": stable["month_count"],
                f"rate_{left_member}": left_rate,
                f"rate_{right_member}": right_rate,
                "left_count": left_count,
                "right_count": right_count,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["preferred_support_met", "winner_rate", "pair_gap", "support", "stream_count", "month_count"],
            ascending=[False, False, False, False, False, False]
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "pair", "trait", "support", "preferred_support_met", "winner_member", "loser_member",
            "winner_rate", "loser_rate", "pair_gap", "stream_count", "month_count",
            f"rate_{left_member}", f"rate_{right_member}", "left_count", "right_count"
        ])
    return out


def main():
    st.set_page_config(page_title="Core025 Pairwise Separator Miner v1", layout="wide")
    st.title("Core025 Pairwise Separator Miner v1")
    st.caption("Mines stronger between-member separator traits so more true winners become Top1.")
    st.code(BUILD_MARKER, language="text")

    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        st.header("Pairwise separator filters")
        min_support = st.number_input("Minimum support", min_value=10, value=20, step=1)
        preferred_support = st.number_input("Preferred support", min_value=10, value=35, step=1)
        min_dom_rate = st.slider("Minimum winner rate", min_value=0.50, max_value=0.95, value=0.62, step=0.01)
        min_gap = st.slider("Minimum pairwise gap", min_value=0.00, max_value=0.50, value=0.12, step=0.01)
        min_streams = st.number_input("Minimum distinct streams", min_value=1, value=3, step=1)
        min_months = st.number_input("Minimum distinct months", min_value=1, value=2, step=1)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=25, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="pairwise_hist")
    if not hist_file:
        st.info("Upload the full history file to begin.")
        return

    try:
        hist = prepare_history(load_table(hist_file))
    except Exception as e:
        st.exception(e)
        return

    transitions = build_transitions(hist)
    core_hits = transitions[transitions["is_core025_hit"] == 1].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Transitions", f"{len(transitions):,}")
    c2.metric("Core025 hit events", f"{len(core_hits):,}")
    c3.metric("Core025 base rate", f"{transitions['is_core025_hit'].mean():.4f}")

    if st.button("Run Pairwise Separator Miner", type="primary"):
        try:
            with st.spinner("Mining pairwise separator traits..."):
                pair_0025_0225 = build_pairwise_separator_traits(core_hits, "0025", "0225", int(min_support), int(preferred_support), float(min_dom_rate), float(min_gap), int(min_streams), int(min_months))
                pair_0225_0255 = build_pairwise_separator_traits(core_hits, "0225", "0255", int(min_support), int(preferred_support), float(min_dom_rate), float(min_gap), int(min_streams), int(min_months))
                pair_0025_0255 = build_pairwise_separator_traits(core_hits, "0025", "0255", int(min_support), int(preferred_support), float(min_dom_rate), float(min_gap), int(min_streams), int(min_months))
                all_pairwise = pd.concat([pair_0025_0225, pair_0225_0255, pair_0025_0255], ignore_index=True) if (len(pair_0025_0225) or len(pair_0225_0255) or len(pair_0025_0255)) else pd.DataFrame()
            st.session_state["pairwise_separator_results"] = {
                "pair_0025_0225": pair_0025_0225,
                "pair_0225_0255": pair_0225_0255,
                "pair_0025_0255": pair_0025_0255,
                "all_pairwise": all_pairwise,
            }
        except Exception as e:
            st.exception(e)
            return

    if "pairwise_separator_results" not in st.session_state or st.session_state["pairwise_separator_results"] is None:
        return

    results = st.session_state["pairwise_separator_results"]

    st.subheader("0025 vs 0225 separators")
    st.dataframe(safe_display_df(results["pair_0025_0225"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download 0025 vs 0225 separators CSV", df_to_csv_bytes(results["pair_0025_0225"]), "core025_pairwise_separator_miner_v1__2026-03-28__0025_vs_0225.csv", "text/csv")

    st.subheader("0225 vs 0255 separators")
    st.dataframe(safe_display_df(results["pair_0225_0255"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download 0225 vs 0255 separators CSV", df_to_csv_bytes(results["pair_0225_0255"]), "core025_pairwise_separator_miner_v1__2026-03-28__0225_vs_0255.csv", "text/csv")

    st.subheader("0025 vs 0255 separators")
    st.dataframe(safe_display_df(results["pair_0025_0255"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download 0025 vs 0255 separators CSV", df_to_csv_bytes(results["pair_0025_0255"]), "core025_pairwise_separator_miner_v1__2026-03-28__0025_vs_0255.csv", "text/csv")

    st.subheader("All pairwise separators")
    st.dataframe(safe_display_df(results["all_pairwise"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download all pairwise separators CSV", df_to_csv_bytes(results["all_pairwise"]), "core025_pairwise_separator_miner_v1__2026-03-28__all_pairwise.csv", "text/csv")


if __name__ == "__main__":
    if "pairwise_separator_results" not in st.session_state:
        st.session_state["pairwise_separator_results"] = None
    main()
""",
}


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


def load_module_from_source(source: str, module_name: str):
    mod = types.ModuleType(module_name)
    mod.__file__ = f"<embedded:{module_name}>"
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    return mod


def save_uploaded_py(uploaded_file, canonical_name: str) -> Optional[Path]:
    if uploaded_file is None:
        return None
    tmp_dir = Path("/tmp/core025_unified_miner_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / canonical_name
    out.write_bytes(uploaded_file.getvalue())
    return out


@st.cache_resource
def load_real_modules_with_overrides(deep_override: Optional[str], member_override: Optional[str], pair_override: Optional[str]) -> Dict[str, Any]:
    mods: Dict[str, Any] = {}

    deep_path = Path(deep_override) if deep_override else find_file(REQUIRED_FILES["deep"])
    if deep_path is not None and deep_path.exists():
        mods["deep"] = load_module_from_path(deep_path, "core025_deep_trait_miner_v4")
    else:
        mods["deep"] = load_module_from_source(EMBEDDED_MODULES["deep"], "core025_deep_trait_miner_v4")

    member_path = Path(member_override) if member_override else find_file(REQUIRED_FILES["member"])
    if member_path is not None and member_path.exists():
        mods["member"] = load_module_from_path(member_path, "core025_member_trait_miner_v2")
    else:
        mods["member"] = load_module_from_source(EMBEDDED_MODULES["member"], "core025_member_trait_miner_v2")

    pair_path = Path(pair_override) if pair_override else find_file(REQUIRED_FILES["pairwise"])
    if pair_path is not None and pair_path.exists():
        mods["pairwise"] = load_module_from_path(pair_path, "core025_pairwise_separator_miner_v1")
    else:
        mods["pairwise"] = load_module_from_source(EMBEDDED_MODULES["pairwise"], "core025_pairwise_separator_miner_v1")
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


def _module_status_paths(deep_override, member_override, pair_override):
    return {
        "deep": Path(str(deep_override)) if deep_override else find_file(REQUIRED_FILES["deep"]),
        "member": Path(str(member_override)) if member_override else find_file(REQUIRED_FILES["member"]),
        "pairwise": Path(str(pair_override)) if pair_override else find_file(REQUIRED_FILES["pairwise"]),
    }


def _show_module_status(paths):
    rows = []
    for key, req in REQUIRED_FILES.items():
        p = paths.get(key)
        rows.append({
            "module": key,
            "required_file": req,
            "status": "ready" if p and Path(p).exists() else "missing",
            "source_path": str(p) if p else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="Core025 Unified Miner Hub", layout="wide")
    st.title("Core025 Unified Miner Hub")
    st.caption("One app to standardize seed-event CSVs and run the embedded deep, member, and pairwise miners.")
    st.markdown(f"**BUILD:** `{APP_BUILD}__v5_no_module_uploads`")
    st.info("This build is fully embedded. Do not upload any .py miner files. Only upload your data CSVs.")

    try:
        mods = load_real_modules_with_overrides(None, None, None)
    except Exception as e:
        st.error(f"Embedded miner load failed: {e}")
        return

    t1, t2, t3 = st.tabs(["Formatter", "Deep Trait Miner", "History Miners"])
    with t1:
        tab_formatter()
    with t2:
        try:
            tab_deep_miner(mods)
        except Exception as e:
            st.exception(e)

    with t3:
        try:
            hist_mods = {"member": mods["member"], "pairwise": mods["pairwise"]}
            tab_history_miners(hist_mods)
        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()
