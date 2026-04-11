
#!/usr/bin/env python3
"""
core025_deep_trait_miner_v1__2026-03-23.py

Purpose
-------
Deep trait/pattern miner for Core 025 member assignment work.

What it does
------------
- Loads a real per-event export (seed -> true next member) from CSV/XLSX/TXT.
- Auto-detects likely columns for seed, true member, date, stream, and optional predicted ranks.
- Computes a large, programmatic feature set from the seed.
- Generates many boolean traits automatically.
- Scores traits for:
    * one-vs-rest member separation (25 / 225 / 255)
    * winner-vs-loser separation for any user-specified target bucket
- Mines stacked "bucket" candidates using greedy intersection of high-value traits.
- Exports:
    * feature table
    * single-trait scores
    * stacked bucket candidates
    * zero-hit / separator candidates
    * plain-text summary
- No placeholders, no simulations.

Usage
-----
python core025_deep_trait_miner_v1__2026-03-23.py --input your_per_event.csv
python core025_deep_trait_miner_v1__2026-03-23.py --input your_per_event.csv --target-member 255
python core025_deep_trait_miner_v1__2026-03-23.py --input your_per_event.csv --target-member 25 --min-support 8

Accepted inputs
---------------
CSV, TSV/TXT, XLSX.

Expected content
----------------
At minimum, a seed column and a true-member column.
The script tries hard to auto-detect columns from common names.

Outputs
-------
Creates a self-identifying output folder next to the input file and writes:
- *_feature_table.csv
- *_trait_scores_all_members.csv
- *_trait_scores_member_<member>.csv
- *_separator_candidates_member_<member>.csv
- *_bucket_candidates_member_<member>.csv
- *_summary.txt
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MEMBERS = [25, 225, 255]
MIRROR_PAIRS = {(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)}
DIGITS = list(range(10))


# ----------------------------
# Generic helpers
# ----------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _safe_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(str(x).strip())
    except Exception:
        return None


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
        raise KeyError(
            f"Required column not found. Tried {list(candidates)}. Available columns: {cols}"
        )
    return None


def coerce_member(x) -> Optional[int]:
    """
    Accepts:
    25, 225, 255
    0025, 0225, 0255
    strings containing those values
    """
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
        if v in (25, 225, 255):
            return v
        # 4-digit formats like 0025 / 0225 / 0255
        if token.zfill(4) == "0025":
            return 25
        if token.zfill(4) == "0225":
            return 225
        if token.zfill(4) == "0255":
            return 255
    return None


def canonical_seed(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = re.sub(r"\D", "", str(x))
    if len(s) == 4:
        return s
    return None


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
    return [seed[i:i+2] for i in range(len(seed) - 1)]


def as_unordered_adj_pairs(seed: str) -> List[str]:
    return ["".join(sorted(seed[i:i+2])) for i in range(len(seed) - 1)]


def member_label(v: int) -> str:
    return {25: "0025", 225: "0225", 255: "0255"}.get(v, str(v))


# ----------------------------
# Feature engineering
# ----------------------------

def compute_features(seed: str) -> Dict[str, object]:
    d = digit_list(seed)
    cnt = Counter(d)
    uniq = sorted(cnt.keys())
    s = sum(d)
    spread = max(d) - min(d)
    parity = "".join("E" if x % 2 == 0 else "O" for x in d)
    highlow = "".join("H" if x >= 5 else "L" for x in d)

    consec_links = 0
    for a, b in zip(sorted(set(d))[:-1], sorted(set(d))[1:]):
        if b - a == 1:
            consec_links += 1

    mirrorpair_cnt = sum(
        1
        for a, b in MIRROR_PAIRS
        if a in cnt and b in cnt
    )

    pairwise_absdiff = []
    for i in range(4):
        for j in range(i + 1, 4):
            pairwise_absdiff.append(abs(d[i] - d[j]))

    adj_absdiff = [abs(d[i] - d[i+1]) for i in range(3)]

    features: Dict[str, object] = {
        "seed": seed,
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
        "seed_adj_pairs_ordered": "|".join(as_ordered_adj_pairs(seed)),
        "seed_adj_pairs_unordered": "|".join(sorted(as_unordered_adj_pairs(seed))),
    }

    # digit presence and counts
    for k in DIGITS:
        features[f"seed_has{k}"] = int(k in cnt)
        features[f"seed_cnt{k}"] = int(cnt.get(k, 0))

    # repeat shape
    shape = "".join(map(str, sorted(cnt.values(), reverse=True)))
    shape_name = {
        "1111": "all_unique",
        "211": "one_pair",
        "22": "two_pair",
        "31": "trip",
        "4": "quad",
    }.get(shape, f"shape_{shape}")
    features["seed_repeat_shape"] = shape_name

    # digit-zone counts
    features["cnt_0_3"] = int(sum(0 <= x <= 3 for x in d))
    features["cnt_4_6"] = int(sum(4 <= x <= 6 for x in d))
    features["cnt_7_9"] = int(sum(7 <= x <= 9 for x in d))

    # exact pair indicators
    pair_counts = Counter(as_pair_tokens(seed))
    for a in range(10):
        for b in range(a, 10):
            tok = f"{a}{b}"
            features[f"pair_has_{tok}"] = int(pair_counts.get(tok, 0) > 0)

    # ordered adjacency indicators
    for a in range(10):
        for b in range(10):
            tok = f"{a}{b}"
            features[f"adj_ord_has_{tok}"] = int(tok in as_ordered_adj_pairs(seed))

    return features


# ----------------------------
# Trait generation
# ----------------------------

def bin_numeric_series(s: pd.Series, prefix: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().sum() == 0:
        return out

    uniques = sorted(set(vals.dropna().astype(float).tolist()))
    if len(uniques) <= 20:
        for u in uniques:
            if float(u).is_integer():
                label = str(int(u))
            else:
                label = str(u)
            out[f"{prefix}=={label}"] = vals == u

    # quantile bins
    qs = [0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9]
    quantiles = sorted(set(float(vals.quantile(q)) for q in qs if vals.notna().sum() >= 10))
    for q in quantiles:
        if np.isnan(q):
            continue
        label = int(q) if float(q).is_integer() else round(float(q), 3)
        out[f"{prefix}<={label}"] = vals <= q
        out[f"{prefix}>={label}"] = vals >= q

    # common compact ranges for integer-like columns
    if all(float(x).is_integer() for x in uniques[: min(len(uniques), 20)]):
        int_uniques = [int(x) for x in uniques]
        if len(int_uniques) <= 20:
            for lo in int_uniques:
                for hi in int_uniques:
                    if lo < hi and (hi - lo) <= 3:
                        out[f"{prefix}_in[{lo},{hi}]"] = (vals >= lo) & (vals <= hi)

    return out


def categorical_series_traits(s: pd.Series, prefix: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    vc = s.astype(str).value_counts(dropna=False)
    for v, n in vc.items():
        if n == 0:
            continue
        if n >= 3:
            out[f"{prefix}=={v}"] = s.astype(str) == str(v)
    return out


def build_trait_matrix(df_feat: pd.DataFrame) -> pd.DataFrame:
    trait_cols: Dict[str, pd.Series] = {}

    # Chosen base columns
    numeric_cols = [
        c for c in df_feat.columns
        if (
            c.startswith("seed_")
            or c.startswith("cnt_")
        )
        and c not in {"seed"}
    ]
    categorical_cols = [
        "seed_parity_pattern",
        "seed_highlow_pattern",
        "seed_repeat_shape",
        "seed_sorted",
    ]

    # Numeric threshold/equality traits
    for c in numeric_cols:
        ser = df_feat[c]
        if pd.api.types.is_numeric_dtype(ser):
            trait_cols.update(bin_numeric_series(ser, c))

    # Categorical exact-match traits
    for c in categorical_cols:
        if c in df_feat.columns:
            trait_cols.update(categorical_series_traits(df_feat[c], c))

    # Sparse indicator families: keep only traits that occur at least 3 times
    sparse_prefixes = ("pair_has_", "adj_ord_has_", "seed_has", "seed_cnt")
    for c in df_feat.columns:
        if c.startswith(sparse_prefixes):
            ser = pd.to_numeric(df_feat[c], errors="coerce").fillna(0).astype(int)
            if ser.sum() >= 3:
                trait_cols[c] = ser.astype(bool)

    trait_df = pd.DataFrame(trait_cols, index=df_feat.index).astype(bool)
    return trait_df


# ----------------------------
# Scoring
# ----------------------------

@dataclass
class TraitScore:
    member: int
    trait: str
    support: int
    hits_true: int
    misses_true: int
    hit_rate_true: float
    support_false: int
    hits_false: int
    misses_false: int
    hit_rate_false: float
    lift: float
    precision_gap: float
    is_separator: int


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

        lift = hit_rate_true - base_rate
        precision_gap = hit_rate_true - hit_rate_false
        is_separator = int(hits_true == 0 and support > 0)

        rows.append({
            "member": member,
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
            "lift_vs_base": lift,
            "precision_gap": precision_gap,
            "is_separator": is_separator,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["precision_gap", "hit_rate_true", "support"],
            ascending=[False, False, False]
        ).reset_index(drop=True)
    return out


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
    candidates = scored_traits.copy()
    candidates = candidates[
        (candidates["support"] >= min_support)
        & (candidates["hit_rate_true"] > candidates["base_rate_member"])
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

                # Prefer stronger precision first, then larger support
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
        out = out.sort_values(
            ["hit_rate", "support", "depth"],
            ascending=[False, False, True]
        ).reset_index(drop=True)
    return out


def find_separator_traits(scored_traits: pd.DataFrame, min_support: int = 8) -> pd.DataFrame:
    out = scored_traits[
        (scored_traits["is_separator"] == 1)
        & (scored_traits["support"] >= min_support)
    ].copy()
    if not out.empty:
        out = out.sort_values(["support", "hit_rate_false"], ascending=[False, False]).reset_index(drop=True)
    return out


# ----------------------------
# IO
# ----------------------------

def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".txt", ".tsv"):
        # Try tab first, then generic python engine
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input type: {path.suffix}")


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    seed_col = find_col(
        df,
        ["seed", "seed_result", "previous_result", "prev_result", "prior_result"],
        required=True,
    )
    member_col = find_col(
        df,
        [
            "true_member", "winning_member", "winner_member", "member",
            "actual_member", "result_member", "target_member", "winning core member"
        ],
        required=True,
    )
    date_col = find_col(df, ["date", "draw date", "play_date", "target_date"], required=False)
    stream_col = find_col(df, ["stream", "stream_id", "state_game", "streamkey"], required=False)
    return {
        "seed_col": seed_col,
        "member_col": member_col,
        "date_col": date_col,
        "stream_col": stream_col,
    }


def prepare_event_table(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    cols = detect_columns(df_raw)

    df = df_raw.copy()
    df["seed"] = df[cols["seed_col"]].apply(canonical_seed)
    df["true_member"] = df[cols["member_col"]].apply(coerce_member)

    if cols["date_col"]:
        df["date"] = pd.to_datetime(df[cols["date_col"]], errors="coerce")
    else:
        df["date"] = pd.NaT

    if cols["stream_col"]:
        df["stream"] = df[cols["stream_col"]].astype(str)
    else:
        df["stream"] = ""

    df = df.dropna(subset=["seed", "true_member"]).copy()
    df["true_member"] = df["true_member"].astype(int)

    if len(df) == 0:
        raise ValueError("After cleaning, there are 0 usable rows with both seed and true member.")

    return df.reset_index(drop=True), cols


# ----------------------------
# Summary writer
# ----------------------------

def write_summary(
    out_path: Path,
    df_events: pd.DataFrame,
    detected_cols: Dict[str, Optional[str]],
    per_member_scores: Dict[int, pd.DataFrame],
    per_member_separators: Dict[int, pd.DataFrame],
    per_member_buckets: Dict[int, pd.DataFrame],
) -> None:
    lines = []
    lines.append("CORE 025 DEEP TRAIT MINER SUMMARY")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
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
            top = s.head(10)
            for _, r in top.iterrows():
                lines.append(
                    f"  - {r['trait']} | support={int(r['support'])} | "
                    f"hit_rate_true={r['hit_rate_true']:.3f} | gap={r['precision_gap']:.3f}"
                )
        else:
            lines.append("Top single traits: none")
        lines.append("")

        if not z.empty:
            lines.append("Top separator traits (0 hits when trait true):")
            topz = z.head(10)
            for _, r in topz.iterrows():
                lines.append(
                    f"  - {r['trait']} | support={int(r['support'])} | hits_true=0 | "
                    f"member_hit_rate_when_false={r['hit_rate_false']:.3f}"
                )
        else:
            lines.append("Top separator traits: none")
        lines.append("")

        if not b.empty:
            lines.append("Top stacked bucket candidates:")
            topb = b.head(10)
            for _, r in topb.iterrows():
                lines.append(
                    f"  - {r['bucket_traits']} | support={int(r['support'])} | "
                    f"hit_rate={r['hit_rate']:.3f} | lift={r['lift_vs_base']:.3f}"
                )
        else:
            lines.append("Top stacked bucket candidates: none")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to per-event CSV/XLSX/TXT")
    ap.add_argument("--target-member", type=int, choices=MEMBERS, default=None,
                    help="Optional: focus summary on one member")
    ap.add_argument("--min-support", type=int, default=8, help="Minimum support for separator export")
    ap.add_argument("--bucket-min-support", type=int, default=6, help="Minimum support for stacked buckets")
    ap.add_argument("--bucket-top-k", type=int, default=80, help="How many top traits to consider for bucket search")
    ap.add_argument("--bucket-max-depth", type=int, default=4, help="Maximum stacked bucket depth")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = in_path.parent / f"{in_path.stem}__deep_trait_miner_outputs__{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_table(in_path)
    events, detected_cols = prepare_event_table(raw)

    feat_rows = [compute_features(seed) for seed in events["seed"].astype(str)]
    df_feat = pd.DataFrame(feat_rows)
    df_all = pd.concat([events.reset_index(drop=True), df_feat.reset_index(drop=True)], axis=1)

    trait_df = build_trait_matrix(df_feat)

    # Export feature table
    feature_table_path = out_dir / f"{in_path.stem}__feature_table.csv"
    df_all.to_csv(feature_table_path, index=False)

    all_scores = []
    per_member_scores: Dict[int, pd.DataFrame] = {}
    per_member_separators: Dict[int, pd.DataFrame] = {}
    per_member_buckets: Dict[int, pd.DataFrame] = {}

    for member in MEMBERS:
        scored = score_traits_one_vs_rest(trait_df, df_all["true_member"], member)
        per_member_scores[member] = scored
        all_scores.append(scored.assign(member_label=member_label(member)))

        sep = find_separator_traits(scored, min_support=args.min_support)
        per_member_separators[member] = sep

        buckets = greedy_bucket_search(
            trait_df=trait_df,
            y_member=df_all["true_member"],
            member=member,
            scored_traits=scored,
            min_support=args.bucket_min_support,
            top_k_traits=args.bucket_top_k,
            max_depth=args.bucket_max_depth,
        )
        per_member_buckets[member] = buckets

        scored.to_csv(out_dir / f"{in_path.stem}__trait_scores_member_{member_label(member)}.csv", index=False)
        sep.to_csv(out_dir / f"{in_path.stem}__separator_candidates_member_{member_label(member)}.csv", index=False)
        buckets.to_csv(out_dir / f"{in_path.stem}__bucket_candidates_member_{member_label(member)}.csv", index=False)

    all_scores_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    all_scores_df.to_csv(out_dir / f"{in_path.stem}__trait_scores_all_members.csv", index=False)

    summary_path = out_dir / f"{in_path.stem}__summary.txt"
    write_summary(
        out_path=summary_path,
        df_events=df_all,
        detected_cols=detected_cols,
        per_member_scores=per_member_scores,
        per_member_separators=per_member_separators,
        per_member_buckets=per_member_buckets,
    )

    print(f"Done. Output folder: {out_dir}")


if __name__ == "__main__":
    main()
