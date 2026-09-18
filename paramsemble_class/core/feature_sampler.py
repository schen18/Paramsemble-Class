"""Feature sampling module for generating diverse feature combinations."""

import logging
from itertools import combinations as iter_combinations
from itertools import combinations_with_replacement
from typing import List, Optional

import numpy as np
from scipy.special import comb

logger = logging.getLogger(__name__)

# Systematic enumeration is used instead of random sampling when the
# complete combination space is no larger than this.
SYSTEMATIC_LIMIT = 10000


class FeatureSampler:
    """
    Generates diverse feature combinations for ensemble model training.

    Supports two sampling methods:
    - "unique": Features cannot repeat within a featureset. The sampler
      draws uniformly from the C(n, f) combinations.
    - "replace": Features may repeat within a featureset. The sampler
      draws uniformly from the C(n + f - 1, f) multicombinations
      (feature sets where order does not matter), using the stars-and-bars
      bijection so every distinct multiset is equally likely.

    Parameters
    ----------
    n_features : int
        Total number of features available
    f : int
        Number of features per combination
    m : int
        Number of combinations to generate
    sample : str
        Sampling method: "unique" or "replace"
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self, n_features: int, f: int, m: int, sample: str, random_state: Optional[int] = None
    ):
        self.n_features = n_features
        self.f = f
        self.m = m
        self.sample = sample
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def generate_combinations(self) -> List[List[int]]:
        """
        Generate m feature combinations.

        Returns
        -------
        List[List[int]]
            List of feature index lists, where each inner list contains f
            feature indices. If ``m`` exceeds the number of distinct
            combinations possible, it is capped at that maximum. All
            generated featuresets are distinct.
        """
        max_combinations = self._calculate_max_combinations()

        # Override m if it exceeds maximum possible combinations
        actual_m = min(self.m, max_combinations)
        if actual_m < self.m:
            logger.warning(
                "Requested m=%d exceeds the maximum of %d distinct "
                "combinations for n_features=%d, f=%d, sample=%r; "
                "using the maximum.",
                self.m,
                max_combinations,
                self.n_features,
                self.f,
                self.sample,
            )

        if self.sample == "unique":
            combinations = self._generate_unique_combinations(actual_m)
        else:  # sample == "replace"
            combinations = self._generate_replace_combinations(actual_m)

        if len(combinations) < actual_m:
            logger.warning(
                "Only %d of %d requested feature combinations could be "
                "generated; results are based on the reduced set.",
                len(combinations),
                actual_m,
            )
        return combinations

    def _calculate_max_combinations(self) -> int:
        """
        Calculate maximum possible combinations.

        For unique sampling: C(n_features, f) = n! / (f! * (n-f)!)
        For replace sampling: C(n_features + f - 1, f) — the number of
        distinct multisets of size f drawn from n features (order
        irrelevant), matching the uniform distribution produced by
        :meth:`_generate_replace_combinations`.

        Returns
        -------
        int
            Maximum number of possible combinations
        """
        if self.sample == "unique":
            return int(comb(self.n_features, self.f, exact=True))
        else:  # sample == "replace"
            return int(comb(self.n_features + self.f - 1, self.f, exact=True))

    def _sample_unique_multiset(self) -> List[int]:
        """Draw one featureset without intra-set replacement, sorted."""
        return sorted(self.rng.choice(self.n_features, size=self.f, replace=False).tolist())

    def _sample_replace_multiset(self) -> List[int]:
        """Draw one uniformly random multiset of f features, sorted.

        Uses the stars-and-bars bijection: choosing f distinct positions
        from n + f - 1 slots maps one-to-one to a multiset of f features,
        and the uniform choice of positions makes every multiset equally
        likely (naive iid sampling would over-weight sets with repeats).
        """
        positions = np.sort(
            self.rng.choice(self.n_features + self.f - 1, size=self.f, replace=False)
        )
        return sorted((positions - np.arange(self.f)).tolist())

    def _generate_unique_combinations(self, actual_m: int) -> List[List[int]]:
        """
        Generate combinations without replacement (no duplicates within featureset).

        Parameters
        ----------
        actual_m : int
            Number of combinations to generate (already capped at maximum)

        Returns
        -------
        List[List[int]]
            List of distinct feature combinations
        """
        max_combinations = self._calculate_max_combinations()

        # If we need all possible combinations, generate them systematically
        if actual_m == max_combinations and max_combinations <= SYSTEMATIC_LIMIT:
            all_combos = list(iter_combinations(range(self.n_features), self.f))
            # Shuffle to maintain randomness
            self.rng.shuffle(all_combos)
            return [list(combo) for combo in all_combos[:actual_m]]

        return self._rejection_sample(actual_m, self._sample_unique_multiset)

    def _generate_replace_combinations(self, actual_m: int) -> List[List[int]]:
        """
        Generate combinations with replacement (duplicates allowed within featureset).

        Parameters
        ----------
        actual_m : int
            Number of combinations to generate (already capped at maximum)

        Returns
        -------
        List[List[int]]
            List of distinct feature combinations (may repeat features within a set)
        """
        max_combinations = self._calculate_max_combinations()

        # If we need all possible combinations, generate them systematically
        if actual_m == max_combinations and max_combinations <= SYSTEMATIC_LIMIT:
            all_combos = list(combinations_with_replacement(range(self.n_features), self.f))
            self.rng.shuffle(all_combos)
            return [list(combo) for combo in all_combos[:actual_m]]

        return self._rejection_sample(actual_m, self._sample_replace_multiset)

    def _rejection_sample(self, actual_m: int, sample_one) -> List[List[int]]:
        """Collect ``actual_m`` distinct featuresets by rejecting duplicates.

        Gives up after a generous number of attempts and returns what it
        has (the caller logs a warning about any shortfall).
        """
        combinations: List[List[int]] = []
        seen = set()

        # Coupon-collector style bound: enough attempts to collect m of
        # max items even when m is close to max.
        max_attempts = max(actual_m * 100, 10 * max(actual_m, 1) ** 2)

        attempts = 0
        while len(combinations) < actual_m and attempts < max_attempts:
            combo_tuple = tuple(sample_one())
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                combinations.append(list(combo_tuple))
            attempts += 1

        return combinations
