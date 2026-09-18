"""Shared model-selection logic for the ensemble methods.

Single source of truth for:

- the baseline gate (which constituents are eligible for selection),
- ranking of eligible models (Borda count over PLR/FNR/DRP),
- selection strategies: top-k by rank, or greedy coverage (diversity-aware),
- vote weights for the selected models (Borda position -> weight),
- compilation of Decile Ranked Sets into the ``[id, sets, score]`` output.

All three ensemble methods (intersect, venn, ensemble) delegate here so
their selection behaviour cannot drift apart.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Any, Dict, List, Optional, Sequence

# Number of metrics (PLR, FNR, DRP) on which a model must beat the
# baseline to pass the gate. 1 reproduces the specified ELR behaviour:
# "outperform the baseline and/or have the best performance". Raise it
# (e.g. to 2) for a stricter gate; note that with a strong baseline a
# strict majority gate can reject every constituent model.
GATE_MIN_BETTER = 1

VALID_SELECTION_STRATEGIES = ["rank", "coverage"]


def gate_indices(
    constituent_results: List[Dict[str, Any]],
    baseline_results: Dict[str, Any],
    min_better: int = GATE_MIN_BETTER,
) -> List[int]:
    """Return indices of models that outperform the baseline.

    A model passes the gate when it is strictly better than the baseline
    on at least ``min_better`` of the three metrics (higher PLR, lower
    FNR, higher DRP). The default of 1 ("beats the baseline on any
    metric") reproduces the specified ELR selection semantics; a model
    that loses on all three metrics is never selected. ``inf`` baseline
    PLR blocks the PLR comparison (nothing is greater than infinity), as
    intended.

    Parameters
    ----------
    constituent_results : List[Dict[str, Any]]
        Metric dicts with keys 'plr', 'fnr', 'drp'.
    baseline_results : Dict[str, Any]
        Baseline metric dict with the same keys.
    min_better : int, default=GATE_MIN_BETTER
        How many of the three metrics a model must win.

    Returns
    -------
    List[int]
        Indices of gate-passing models (in input order).
    """
    gate = []
    for i, result in enumerate(constituent_results):
        better = (
            result["plr"] > baseline_results["plr"],
            result["fnr"] < baseline_results["fnr"],
            result["drp"] > baseline_results["drp"],
        )
        if sum(better) >= min_better:
            gate.append(i)
    return gate


def rank_models(
    constituent_results: List[Dict[str, Any]],
    baseline_results: Dict[str, Any],
    min_better: int = GATE_MIN_BETTER,
) -> List[int]:
    """Rank gate-passing models best-first using a Borda count.

    Each metric contributes the model's rank among the eligible models
    (higher PLR -> better rank, lower FNR -> better rank, higher DRP ->
    better rank; ties share the average rank). The final ordering
    minimizes the summed rank, which keeps the three metrics on equal
    footing regardless of their scales — unlike a raw weighted sum,
    which unbounded PLR values would dominate. ``inf`` PLR sorts as the
    best possible PLR. Ties in the Borda sum break by original model
    index, so ranking is deterministic.

    Parameters
    ----------
    constituent_results : List[Dict[str, Any]]
        Metric dicts with keys 'plr', 'fnr', 'drp'.
    baseline_results : Dict[str, Any]
        Baseline metric dict with the same keys.
    min_better : int, default=GATE_MIN_BETTER
        Baseline gate threshold (see :func:`gate_indices`).

    Returns
    -------
    List[int]
        Indices of gate-passing models, best first.
    """
    eligible = gate_indices(constituent_results, baseline_results, min_better)
    if not eligible:
        return []

    plr = np.array([constituent_results[i]["plr"] for i in eligible], dtype=float)
    fnr = np.array([constituent_results[i]["fnr"] for i in eligible], dtype=float)
    drp = np.array([constituent_results[i]["drp"] for i in eligible], dtype=float)

    # nan-safe: treat non-finite PLR other than +inf as worst.
    plr = np.where(np.isfinite(plr) | (plr == np.inf), plr, -np.inf)

    def avg_ranks(values, reverse):
        # rankdata 'average' handles ties; order controlled by sign
        signed = -values if reverse else values
        return rankdata(signed, method="average")

    borda = (
        avg_ranks(plr, reverse=True) + avg_ranks(fnr, reverse=False) + avg_ranks(drp, reverse=True)
    )

    # Sort by (borda sum ascending, original index ascending) for determinism
    order = sorted(range(len(eligible)), key=lambda k: (borda[k], eligible[k]))
    return [eligible[k] for k in order]


def coverage_select(
    constituent_results: List[Dict[str, Any]],
    ranked_indices: List[int],
    spread: int,
) -> List[int]:
    """Greedily select models that maximize marginal positive coverage.

    Starts from the best-ranked model, then repeatedly adds the candidate
    whose Decile Positive Set adds the most *new* true-positive IDs to the
    union already covered (ties broken by Borda rank, then index). This is
    greedy submodular maximization of union coverage: unlike top-k by
    rank, it will not spend selection slots on models whose positives are
    already captured — the union that intersect compiles is optimized
    directly.

    Parameters
    ----------
    constituent_results : List[Dict[str, Any]]
        Metric dicts with a 'dps' key (set of true-positive IDs).
    ranked_indices : List[int]
        Gate-passing model indices, best Borda rank first (candidates).
    spread : int
        Maximum number of models to select.

    Returns
    -------
    List[int]
        Selected model indices in greedy order (may be shorter than
        ``spread`` if candidates are exhausted or add no coverage).
    """
    if not ranked_indices or spread <= 0:
        return []

    selected = [ranked_indices[0]]
    covered = set(constituent_results[ranked_indices[0]]["dps"])
    remaining = list(ranked_indices[1:])

    while remaining and len(selected) < spread:
        best_idx = None
        best_gain = -1
        for pos, cand in enumerate(remaining):
            gain = len(constituent_results[cand]["dps"] - covered)
            # remaining is in Borda order, so strict > keeps the
            # better-ranked candidate on ties (first wins)
            if gain > best_gain:
                best_gain = gain
                best_idx = pos
        if best_idx is None or best_gain <= 0:
            break
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered |= constituent_results[chosen]["dps"]

    return selected


def weights_for_selection(
    selected_indices: Sequence[int],
    ranked_indices: Sequence[int],
) -> List[float]:
    """Vote weights for the selected models, by Borda position.

    The best-ranked selected model gets weight ``n_selected``, the next
    ``n_selected - 1``, ..., the last gets 1. Weights therefore express
    relative quality and yield a continuous score when summed over the
    models containing an ID (integer vote counts alone tie heavily and
    rank poorly at fine granularity).

    Parameters
    ----------
    selected_indices : Sequence[int]
        The selected model indices (any order).
    ranked_indices : Sequence[int]
        Full Borda-ordered gate-passing indices.

    Returns
    -------
    List[float]
        Weights aligned with ``selected_indices``.
    """
    position = {idx: pos for pos, idx in enumerate(ranked_indices)}
    n = len(selected_indices)
    weights = []
    for idx in selected_indices:
        pos = position.get(idx, len(ranked_indices))
        better_selected = sum(
            1 for other in selected_indices if position.get(other, len(ranked_indices)) < pos
        )
        weights.append(float(n - better_selected))
    return weights


def combine_drs(
    constituent_results: List[Dict[str, Any]],
    indices: List[int],
    weights: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Compile Decile Ranked Sets of the given models into ``[id, sets, score]``.

    Parameters
    ----------
    constituent_results : List[Dict[str, Any]]
        Metric dicts with a 'drs' key (set of IDs).
    indices : List[int]
        Indices of the models to combine.
    weights : Sequence[float], optional
        Per-model vote weights aligned with ``indices``. When omitted,
        every model weighs 1.0 and ``score`` equals ``sets``.

    Returns
    -------
    pd.DataFrame
        Deduplicated IDs with occurrence counts (``sets``) and weighted
        scores (``score``), sorted by sets desc, score desc, id asc.
    """
    if weights is None:
        weights = [1.0] * len(indices)
    id_counts: Dict[Any, int] = {}
    id_scores: Dict[Any, float] = {}
    for idx, weight in zip(indices, weights):
        for id_value in constituent_results[idx]["drs"]:
            id_counts[id_value] = id_counts.get(id_value, 0) + 1
            id_scores[id_value] = id_scores.get(id_value, 0.0) + weight
    return build_sets_dataframe(id_counts, id_scores)


def build_sets_dataframe(
    id_counts: Dict[Any, int],
    id_scores: Optional[Dict[Any, float]] = None,
) -> pd.DataFrame:
    """Build the sorted ``[id, sets, score]`` DataFrame from id->count/score maps."""
    if not id_counts:
        return pd.DataFrame(columns=["id", "sets", "score"])
    if id_scores is None:
        id_scores = {id_value: float(count) for id_value, count in id_counts.items()}
    df = pd.DataFrame(
        [
            {"id": id_value, "sets": count, "score": id_scores.get(id_value, float(count))}
            for id_value, count in id_counts.items()
        ]
    )
    df = df.sort_values(["sets", "score", "id"], ascending=[False, False, True]).reset_index(
        drop=True
    )
    return df
