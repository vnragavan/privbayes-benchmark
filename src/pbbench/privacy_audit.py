from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


def _select_qi_cols(real: pd.DataFrame, k: int = 3) -> list[str]:
    """Heuristic QI selection: top-k numeric columns by variance."""
    try:
        cols = (
            real.select_dtypes(include=[np.number])
            .var(numeric_only=True)
            .sort_values(ascending=False)
            .head(int(k))
            .index.tolist()
        )
        return [c for c in cols if c in real.columns]
    except Exception:
        return []


def _discretize_for_audit(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    *,
    n_bins: int = 20,
    cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discretize real/syn into aligned integer codes for Hamming/pattern audits.

    - Numeric: build bin edges from REAL via qcut(retbins=True), then cut SYN using same bins.
    - Non-numeric: build categories from REAL, map SYN unseen values to -1.
    """
    use_cols = cols or [c for c in real.columns if c in syn.columns]
    r_out: dict[str, Any] = {}
    s_out: dict[str, Any] = {}

    for c in use_cols:
        r = real[c]
        s = syn[c]
        if pd.api.types.is_numeric_dtype(r):
            rr = pd.to_numeric(r, errors="coerce")
            ss = pd.to_numeric(s, errors="coerce")
            ok = rr.notna()
            if ok.sum() < 2:
                r_out[c] = np.zeros(len(real), dtype=int)
                s_out[c] = np.zeros(len(syn), dtype=int)
                continue
            try:
                _, bins = pd.qcut(rr[ok], q=int(n_bins), retbins=True, duplicates="drop")
                # pd.cut includes lowest to keep alignment
                r_codes = pd.cut(rr, bins=bins, labels=False, include_lowest=True)
                s_codes = pd.cut(ss, bins=bins, labels=False, include_lowest=True)
            except Exception:
                # fallback: equal-width bins on real range
                lo = float(np.nanmin(rr.to_numpy()))
                hi = float(np.nanmax(rr.to_numpy()))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    r_out[c] = np.zeros(len(real), dtype=int)
                    s_out[c] = np.zeros(len(syn), dtype=int)
                    continue
                bins = np.linspace(lo, hi, int(n_bins) + 1)
                r_codes = pd.cut(rr, bins=bins, labels=False, include_lowest=True)
                s_codes = pd.cut(ss, bins=bins, labels=False, include_lowest=True)

            r_out[c] = pd.to_numeric(r_codes, errors="coerce").fillna(-1).astype(int).to_numpy()
            s_out[c] = pd.to_numeric(s_codes, errors="coerce").fillna(-1).astype(int).to_numpy()
        else:
            # categories from real
            cats = pd.Series(r, copy=False).astype("string").fillna("__NA__").unique().tolist()
            cat_index = {str(v): i for i, v in enumerate(cats)}
            r_vals = pd.Series(r, copy=False).astype("string").fillna("__NA__").map(lambda v: cat_index.get(str(v), -1))
            s_vals = pd.Series(s, copy=False).astype("string").fillna("__NA__").map(lambda v: cat_index.get(str(v), -1))
            r_out[c] = pd.to_numeric(r_vals, errors="coerce").fillna(-1).astype(int).to_numpy()
            s_out[c] = pd.to_numeric(s_vals, errors="coerce").fillna(-1).astype(int).to_numpy()

    return pd.DataFrame(r_out), pd.DataFrame(s_out)


@dataclass
class MemorizationAudit:
    emr: float
    mean_dsyn: float
    mean_dreal: float


def nearest_neighbor_memorization(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    *,
    n_bins: int = 20,
    cols: Optional[list[str]] = None,
) -> MemorizationAudit:
    """
    Nearest-neighbor memorization probe (Hamming distance on discretized codes).

    - d_syn(x): min Hamming distance from real record to any synthetic record
    - d_real(x): min Hamming distance to another real record (excluding itself)
    - EMR: fraction where d_syn(x) < d_real(x)
    """
    from sklearn.neighbors import NearestNeighbors

    r_disc, s_disc = _discretize_for_audit(real, syn, n_bins=n_bins, cols=cols)
    Xr = r_disc.to_numpy(dtype=int, copy=False)
    Xs = s_disc.to_numpy(dtype=int, copy=False)

    if len(real) == 0 or len(syn) == 0:
        return MemorizationAudit(emr=float("nan"), mean_dsyn=float("nan"), mean_dreal=float("nan"))

    nn_syn = NearestNeighbors(n_neighbors=1, metric="hamming", algorithm="brute")
    nn_syn.fit(Xs)
    d_syn, _ = nn_syn.kneighbors(Xr, n_neighbors=1, return_distance=True)
    d_syn = d_syn.reshape(-1)

    # real-to-real excluding itself
    if len(real) >= 2:
        nn_real = NearestNeighbors(n_neighbors=2, metric="hamming", algorithm="brute")
        nn_real.fit(Xr)
        d_rr, _ = nn_real.kneighbors(Xr, n_neighbors=2, return_distance=True)
        d_real = d_rr[:, 1]
    else:
        d_real = np.full(len(real), np.inf)

    emr = float(np.mean(d_syn < d_real))
    return MemorizationAudit(emr=emr, mean_dsyn=float(np.mean(d_syn)), mean_dreal=float(np.mean(d_real)))


def unique_pattern_leakage(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    *,
    cols: Optional[list[str]] = None,
    n_bins: int = 20,
) -> float:
    """
    Unique Pattern Leakage: fraction of REAL patterns with support=1 that appear in SYN at least once.
    """
    r_disc, s_disc = _discretize_for_audit(real, syn, n_bins=n_bins, cols=cols)
    if r_disc.empty or s_disc.empty:
        return float("nan")
    r_tuples = list(map(tuple, r_disc.to_numpy(dtype=int, copy=False)))
    s_set = set(map(tuple, s_disc.to_numpy(dtype=int, copy=False)))
    vc = pd.Series(r_tuples).value_counts()
    uniques = set(vc[vc == 1].index.tolist())
    if not uniques:
        return 0.0
    hit = sum(1 for u in uniques if u in s_set)
    return float(hit / len(uniques))


@dataclass
class RareCombinationAudit:
    rmr: float  # rare match rate
    mae: float  # mean absolute error on rare cells


def rare_combination_leakage(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    *,
    tau: int = 3,
    cols: Optional[list[str]] = None,
    n_bins: int = 20,
) -> RareCombinationAudit:
    """
    Rare-combination leakage on discretized patterns.

    For patterns c with 0 < count_real(c) <= tau:
    - RMR_tau: fraction of such patterns that appear in SYN at least once
    - MAE_tau: mean |count_syn(c) - count_real(c)| over those patterns
    """
    r_disc, s_disc = _discretize_for_audit(real, syn, n_bins=n_bins, cols=cols)
    if r_disc.empty or s_disc.empty:
        return RareCombinationAudit(rmr=float("nan"), mae=float("nan"))
    r_tuples = list(map(tuple, r_disc.to_numpy(dtype=int, copy=False)))
    s_tuples = list(map(tuple, s_disc.to_numpy(dtype=int, copy=False)))
    r_counts = pd.Series(r_tuples).value_counts()
    s_counts = pd.Series(s_tuples).value_counts()

    rare = r_counts[(r_counts > 0) & (r_counts <= int(tau))]
    if rare.empty:
        return RareCombinationAudit(rmr=0.0, mae=0.0)

    hit = 0
    abs_err = []
    for pat, cr in rare.items():
        cs = int(s_counts.get(pat, 0))
        if cs >= 1:
            hit += 1
        abs_err.append(abs(int(cr) - cs))
    return RareCombinationAudit(rmr=float(hit / len(rare)), mae=float(np.mean(abs_err) if abs_err else 0.0))


def conditional_disclosure_l1(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    *,
    qi_cols: Optional[list[str]] = None,
    sensitive_col: str,
    n_bins: int = 20,
) -> float:
    """
    Conditional disclosure leakage:
    average L1 distance between P(S | Q=q) in real vs syn, weighted by P_real(Q=q).

    Returns value in [0, 2]. Lower is better.
    """
    if sensitive_col not in real.columns or sensitive_col not in syn.columns:
        return float("nan")

    if qi_cols is None:
        qi_cols = _select_qi_cols(real, k=3)
    qi_cols = [c for c in qi_cols if c in real.columns and c in syn.columns and c != sensitive_col]
    if not qi_cols:
        return float("nan")

    cols = qi_cols + [sensitive_col]
    r_disc, s_disc = _discretize_for_audit(real[cols], syn[cols], n_bins=n_bins, cols=cols)

    # Build conditional distributions per QI key
    r_q = list(map(tuple, r_disc[qi_cols].to_numpy(dtype=int, copy=False)))
    s_q = list(map(tuple, s_disc[qi_cols].to_numpy(dtype=int, copy=False)))
    r_s = r_disc[sensitive_col].to_numpy(dtype=int, copy=False)
    s_s = s_disc[sensitive_col].to_numpy(dtype=int, copy=False)

    # value sets (ensure same support)
    s_values = sorted(set(map(int, np.unique(np.concatenate([r_s, s_s])))))
    idx = {v: i for i, v in enumerate(s_values)}

    def cond_counts(q_list, s_arr):
        out: dict[tuple[int, ...], np.ndarray] = {}
        for q, sval in zip(q_list, s_arr):
            if q not in out:
                out[q] = np.zeros(len(s_values), dtype=float)
            out[q][idx[int(sval)]] += 1.0
        return out

    rc = cond_counts(r_q, r_s)
    sc = cond_counts(s_q, s_s)

    total_r = float(len(r_q))
    if total_r <= 0:
        return float("nan")

    l1s = []
    weights = []
    for q, rvec in rc.items():
        w = float(rvec.sum() / total_r)
        rprob = rvec / max(rvec.sum(), 1e-12)
        svec = sc.get(q, np.zeros_like(rvec))
        sprob = svec / max(svec.sum(), 1e-12)
        l1 = float(np.sum(np.abs(rprob - sprob)))
        l1s.append(l1)
        weights.append(w)

    return float(np.average(l1s, weights=weights) if l1s else float("nan"))


@dataclass
class MembershipInferenceAudit:
    auc: float
    advantage: float


def membership_inference_distance_attack(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    syn_from_train: pd.DataFrame,
    *,
    n_bins: int = 20,
    cols: Optional[list[str]] = None,
) -> MembershipInferenceAudit:
    """
    Simple membership inference attack:
    score(x) = - min_{s in SYN} HammingDistance(x, s) on discretized codes.

    Labels: 1 for records in real_train (members), 0 for real_holdout (non-members).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors

    if len(real_train) == 0 or len(real_holdout) == 0 or len(syn_from_train) == 0:
        return MembershipInferenceAudit(auc=float("nan"), advantage=float("nan"))

    # Discretize against the TRAIN distribution (important).
    r_disc, s_disc = _discretize_for_audit(real_train, syn_from_train, n_bins=n_bins, cols=cols)
    h_disc, _ = _discretize_for_audit(real_train, real_holdout, n_bins=n_bins, cols=cols)

    Xs = s_disc.to_numpy(dtype=int, copy=False)
    nn = NearestNeighbors(n_neighbors=1, metric="hamming", algorithm="brute")
    nn.fit(Xs)

    d_in, _ = nn.kneighbors(r_disc.to_numpy(dtype=int, copy=False), n_neighbors=1, return_distance=True)
    d_out, _ = nn.kneighbors(h_disc.to_numpy(dtype=int, copy=False), n_neighbors=1, return_distance=True)

    scores = np.concatenate([-d_in.reshape(-1), -d_out.reshape(-1)])
    y = np.concatenate([np.ones(len(d_in)), np.zeros(len(d_out))])

    try:
        auc = float(roc_auc_score(y, scores))
    except Exception:
        auc = float("nan")

    # advantage = max_t (TPR - FPR)
    # thresholds on score; brute over observed scores
    uniq = np.unique(scores)
    best = -1.0
    for t in uniq:
        pred = scores >= t
        tpr = float(np.mean(pred[y == 1])) if np.any(y == 1) else 0.0
        fpr = float(np.mean(pred[y == 0])) if np.any(y == 0) else 0.0
        best = max(best, tpr - fpr)
    return MembershipInferenceAudit(auc=auc, advantage=float(best if best >= 0 else float("nan")))

