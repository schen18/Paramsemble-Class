"""Intersect ensemble method for ELR."""

import pandas as pd
from typing import Dict, List, Any

from paramsemble_class.ensemble._common import (
    rank_models,
    combine_drs,
    coverage_select,
    weights_for_selection,
    VALID_SELECTION_STRATEGIES,
)


class IntersectMethod:
    """
    Intersect ensemble method for identifying high-confidence predictions.

    The intersect method selects top-performing constituent models and
    identifies IDs that appear in multiple models' Decile Ranked Sets (DRS).
    This approach finds predictions supported by diverse feature combinations.

    The method:
    1. Keeps models that beat the baseline on any of the 3 metrics
       (PLR, FNR, DRP) and ranks them by a Borda count across the metrics
    2. Selects the top ``spread`` models (``selection="rank"``), or greedily
       picks models that maximize marginal coverage of true-positive IDs
       (``selection="coverage"`` — diversity-aware, avoids spending slots on
       redundant models)
    3. Compiles weighted DRS from selected models: each model votes with a
       weight given by its Borda position (best = n_selected, ..., last = 1)

    Returns a DataFrame with deduplicated IDs, their occurrence counts
    (``sets``) and weighted scores (``score``).
    """

    @staticmethod
    def _rank_models(
        constituent_results: List[Dict[str, Any]], baseline_results: Dict[str, Any]
    ) -> List[int]:
        """
        Rank constituent models by performance metrics.

        Models must outperform the baseline on at least one of the metrics
        (higher PLR, lower FNR, higher DRP). Eligible models are ordered
        by a Borda count over the three metrics; see
        :func:`paramsemble_class.ensemble._common.rank_models`.

        Parameters
        ----------
        constituent_results : List[Dict[str, Any]]
            Metric dicts with keys 'plr', 'fnr', 'drp', 'drs', 'dps'.
        baseline_results : Dict[str, Any]
            Baseline metric dict with the same keys.

        Returns
        -------
        List[int]
            Indices of eligible models sorted by rank (best first).
        """
        return rank_models(constituent_results, baseline_results)

    @staticmethod
    def select(
        constituent_results: List[Dict[str, Any]],
        baseline_results: Dict[str, Any],
        spread: int,
        selection: str = "rank",
    ) -> List[int]:
        """
        Select models for the intersect combination.

        Parameters
        ----------
        constituent_results : List[Dict[str, Any]]
            Metric dicts with keys 'plr', 'fnr', 'drp', 'drs', 'dps'.
        baseline_results : Dict[str, Any]
            Baseline metric dict with the same keys.
        spread : int
            Number of top models to select.
        selection : str, default="rank"
            "rank": top ``spread`` by Borda count (classic behaviour).
            "coverage": greedy selection maximizing marginal coverage of
            true-positive IDs (diversity-aware).

        Returns
        -------
        List[int]
            Indices of the selected models (may be shorter than ``spread``
            if fewer models pass the baseline gate or coverage is exhausted).
        """
        if selection not in VALID_SELECTION_STRATEGIES:
            raise ValueError(
                f"selection must be one of {VALID_SELECTION_STRATEGIES}, got {selection!r}"
            )
        ranked = IntersectMethod._rank_models(constituent_results, baseline_results)
        if selection == "coverage":
            return coverage_select(constituent_results, ranked, spread)
        return ranked[: min(spread, len(ranked))]

    @staticmethod
    def select_and_combine(
        constituent_results: List[Dict[str, Any]],
        baseline_results: Dict[str, Any],
        spread: int,
        selection: str = "rank",
    ) -> pd.DataFrame:
        """
        Select models and combine their weighted Decile Ranked Sets.

        Parameters
        ----------
        constituent_results : List[Dict[str, Any]]
            Metric dicts with keys 'plr', 'fnr', 'drp', 'drs', 'dps'.
        baseline_results : Dict[str, Any]
            Baseline metric dict with the same keys.
        spread : int
            Number of top models to select.
        selection : str, default="rank"
            Selection strategy ("rank" or "coverage"); see :meth:`select`.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'id': Deduplicated IDs from selected model DRS
            - 'sets': Count of how many selected model DRS contain each ID
            - 'score': Weighted vote total (Borda-position weights)
        """
        ranked = IntersectMethod._rank_models(constituent_results, baseline_results)
        selected = IntersectMethod.select(constituent_results, baseline_results, spread, selection)
        weights = weights_for_selection(selected, ranked)
        return combine_drs(constituent_results, selected, weights)
