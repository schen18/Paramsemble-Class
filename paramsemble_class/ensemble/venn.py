"""Venn ensemble method for ELR."""

import pandas as pd
from typing import Dict, List, Any, Set

from paramsemble_class.ensemble._common import (
    rank_models,
    combine_drs,
    weights_for_selection,
)


class VennMethod:
    """
    Venn ensemble method for discovering unique predictions.

    The venn method identifies predictions not captured by the baseline model
    by selecting models that provide unique insights through alternative feature
    combinations.

    The method:
    1. Keeps models that beat the baseline on any of the 3 metrics
       (PLR, FNR, DRP) and ranks them by a Borda count across the metrics
    2. Initially selects the top 2 x spread models, then walks them in rank
       order discarding any model whose Decile Positive Set adds no new
       true-positive IDs beyond the baseline and previously kept models
       (``selection="rank"``); or greedily selects up to ``spread`` models
       by marginal positive coverage over the baseline DPS
       (``selection="coverage"``)
    3. Compiles weighted DRS from the kept models (Borda-position weights)

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
        (higher PLR, lower FNR, higher DRP); eligible models are ordered
        by a Borda count over the three metrics.

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
        Select models that contribute unique true-positive IDs.

        Parameters
        ----------
        constituent_results : List[Dict[str, Any]]
            Metric dicts with keys 'plr', 'fnr', 'drp', 'drs', 'dps'.
        baseline_results : Dict[str, Any]
            Baseline metric dict with the same keys.
        spread : int
            Number of top models to target.
        selection : str, default="rank"
            "rank": classic venn behaviour — pool of 2 x spread models,
            keep those whose DPS adds IDs beyond the baseline DPS and
            previously kept models.
            "coverage": greedy selection (up to ``spread``) maximizing
            marginal positive coverage starting from the baseline DPS.

        Returns
        -------
        List[int]
            Indices of kept models, in selection order.
        """
        ranked = VennMethod._rank_models(constituent_results, baseline_results)

        if selection == "coverage":
            # Seed coverage with the baseline's positives so selection
            # maximizes what the ensemble ADDS over the baseline.
            if not ranked:
                return []
            best = ranked[0]
            selected = [best]
            covered = set(baseline_results["dps"]) | set(constituent_results[best]["dps"])
            remaining = list(ranked[1:])
            while remaining and len(selected) < spread:
                gains = [len(constituent_results[c]["dps"] - covered) for c in remaining]
                best_gain = max(gains)
                if best_gain <= 0:
                    break
                pos = gains.index(best_gain)  # first = best-ranked on ties
                chosen = remaining.pop(pos)
                selected.append(chosen)
                covered |= constituent_results[chosen]["dps"]
            return selected

        # Classic behaviour: pool of 2 x spread, discard models with no
        # unique DPS IDs versus baseline + incremental set.
        initial = ranked[: min(2 * spread, len(ranked))]
        if not initial:
            return []

        baseline_dps: Set[Any] = set(baseline_results["dps"])
        incremental_id_set: Set[Any] = set(baseline_dps)
        undiscarded = []

        for idx in initial:
            unique_ids = constituent_results[idx]["dps"] - incremental_id_set
            if unique_ids:
                undiscarded.append(idx)
                incremental_id_set.update(unique_ids)

        return undiscarded

    @staticmethod
    def select_and_combine(
        constituent_results: List[Dict[str, Any]],
        baseline_results: Dict[str, Any],
        spread: int,
        selection: str = "rank",
    ) -> pd.DataFrame:
        """
        Select models with unique predictions and combine weighted Decile Ranked Sets.

        Parameters
        ----------
        constituent_results : List[Dict[str, Any]]
            Metric dicts with keys 'plr', 'fnr', 'drp', 'drs', 'dps'.
        baseline_results : Dict[str, Any]
            Baseline metric dict with the same keys.
        spread : int
            Number of top models to target.
        selection : str, default="rank"
            Selection strategy ("rank" or "coverage"); see :meth:`select`.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'id': Deduplicated IDs from kept model DRS
            - 'sets': Count of how many kept model DRS contain each ID
            - 'score': Weighted vote total (Borda-position weights)
        """
        ranked = VennMethod._rank_models(constituent_results, baseline_results)
        selected = VennMethod.select(constituent_results, baseline_results, spread, selection)
        weights = weights_for_selection(selected, ranked)
        return combine_drs(constituent_results, selected, weights)
