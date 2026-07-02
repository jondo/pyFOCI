"""
Feature Ordering by Conditional Independence (FOCI)
"""

# Authors: Robert Pollak <robert.pollak@jku.at>
# License: BSD 3 clause

from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, _fit_context
from sklearn.feature_selection import SelectorMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    InvalidParameterError,
    StrOptions,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data


def _rank_max(y):
    """Compute 1-based ranks with the ``max`` method for ties.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Values to rank.

    Returns
    -------
    ranks : ndarray of shape (n_samples,), dtype=float
        One-based ranks, assigning the maximum rank to tied values.
    """
    y = np.asarray(y)
    n = y.shape[0]
    idx = np.argsort(y, kind="mergesort")
    y_sorted = y[idx]
    ranks = np.empty(n, dtype=float)
    i = 0
    # Assign the maximum rank to ties
    while i < n:
        j = i
        while j + 1 < n and y_sorted[j + 1] == y_sorted[i]:
            j += 1
        ranks[idx[i : j + 1]] = j + 1
        i = j + 1
    return ranks


def _nn_radius_based(X_sub, random_state):
    """Radius-based NN selection with random tie breaking."""
    X_sub = np.asarray(X_sub)
    n = X_sub.shape[0]

    # Fit NN on X_sub
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
    nbrs.fit(X_sub)

    # Get min distances
    distances, _ = nbrs.kneighbors(n_neighbors=1)
    min_distance = distances[:, 0]

    eps = 1e-13  # to get all neighbors of min distance
    # For each i, collect all min-dist neighbors, remove self, pick one at random
    nbr_i = np.empty(n, dtype=int)
    for i in range(n):
        # Query neighbors in the tight radius around the nearest neighbor distance
        neighbors = nbrs.radius_neighbors(
            X_sub[i, :].reshape(1, -1), min_distance[i] + eps, return_distance=False
        )[0]
        # Remove self index if present
        neighbors = neighbors[neighbors != i]
        nbr_i[i] = random_state.choice(neighbors)

    return nbr_i


def _nn_grouping_based(X_sub, random_state):
    """Grouping-based NN selection with random tie-breaking.

    Precondition: n_samples >= 2.
    """
    X_sub = np.asarray(X_sub)
    n = X_sub.shape[0]

    # 1) Group exactly identical rows
    Xu, inv = np.unique(X_sub, axis=0, return_inverse=True)  # Xu: (m, p)
    m = Xu.shape[0]

    groups = [[] for _ in range(m)]
    for i, g in enumerate(inv):
        groups[g].append(i)
    groups = [np.asarray(g, dtype=int) for g in groups]

    nbr_i = np.empty(n, dtype=int)

    for i in range(n):
        gi = inv[i]
        members = groups[gi]

        # repeated data: choose another member of same group at random
        if members.size >= 2:
            choices = members[members != i]
            nbr_i[i] = int(random_state.choice(choices))
            continue

        # per-query brute-force distances to all unique rows
        diff = Xu - Xu[gi]  # (m, p)
        d2 = (diff * diff).sum(axis=1)  # rowwise dot products
        d2[gi] = np.inf  # exclude self

        # Choose among all original indices whose (unique) row is at minimal distance
        dmin = d2.min()
        tied = np.flatnonzero(d2 == dmin)  # tied unique rows
        candidates = np.concatenate([groups[u] for u in tied])
        nbr_i[i] = int(random_state.choice(candidates))

    return nbr_i


def _Tn(X_sub, y_rank, random_state, *, nn_strategy="grouping"):
    """Compute :math:`T_n` following Fuchs (2024).

    The implementation uses the expression for :math:`T_n` given in
    Section 4.2 after "straightforward calculation" in:

        Fuchs, Sebastian. "Quantifying directed dependence via dimension
        reduction." Journal of Multivariate Analysis 201 (2024): 105266.

    Parameters
    ----------
    X_sub : array-like of shape (n_samples, n_selected_features)
        Candidate subset of the input features used to compute nearest
        neighbors.
    y_rank : ndarray of shape (n_samples,)
        One-based ranks of the target values, typically computed with
        :func:`_rank_max`.
    random_state : numpy.random.RandomState
        Random number generator used to break nearest-neighbor ties.
    nn_strategy : {"grouping", "radius"}, default="grouping"
        Strategy used to select the nearest neighbor indices.

    Returns
    -------
    Tn : float
        Value of the :math:`T_n` statistic for ``X_sub`` and ``y_rank``.
    """
    X_sub = np.asarray(X_sub)
    n = X_sub.shape[0]

    if nn_strategy == "grouping":
        nbr_i = _nn_grouping_based(X_sub, random_state)
    else:
        assert nn_strategy == "radius"
        nbr_i = _nn_radius_based(X_sub, random_state)

    # Apply the formula (indices are 0-based; y_rank is 1-based)
    term1 = np.sum(np.abs(y_rank - y_rank[nbr_i]))
    term2 = np.sum(y_rank[nbr_i]) + np.sum(y_rank) - n * (n + 1)
    result = 1 - 3 / (n**2 - 1) * term1 + 3 / (n**2 - 1) * term2
    return float(result)


class FOCISelector(SelectorMixin, BaseEstimator):
    """
    Feature selector using hierarchical forward selection based on the
    nonlinear Azadkia–Chatterjee T_n coefficient and its Fuchs form (see references).

    At each step, among remaining features, we choose the feature that maximizes
    the cumulative T_n on the growing set S_k = S_{k-1} ∪ {j}.

    Parameters
    ----------
    max_features : int or None, default=None
        Maximum number of features to select. If None, no hard cap is applied
        and selection proceeds until early stopping (if `min_delta` is not None)
        or until all features are selected (if `min_delta` is None).

    min_delta : float or None, default=0
        Minimum required improvement in the cumulative T_n to continue selecting.
        Behavior:

          - First step:
            select a feature only if best_Tn > min_delta; otherwise, select none.
          - Subsequent steps:
            continue only if best_Tn > previous_best + min_delta; otherwise, stop.
          - None disables early stopping (select up to `max_features`).

        Notes:

          - min_delta can be negative to relax stopping,
            0 to reproduce standard early stopping,
            and positive to require stricter improvement.

        Compatibility with the reference implementation:

          - min_delta == 0 corresponds to stop=TRUE
          - min_delta is None corresponds to stop=FALSE

    standardize : {"normalize", None}, default="normalize"
        If "normalize", each column of X is standardized to zero mean and unit
        variance before computing nearest neighbors. If None, X is used as-is.
        Columns with zero variance are left unchanged.

    nn_strategy : {"grouping", "radius"}, default="grouping"
        Strategy used to select nearest neighbors for computing :math:`T_n`.

    random_state : int, RandomState instance or None, default=None
        Controls the random tie-breaking among nearest neighbors. Pass an int
        for reproducible results across multiple calls. If None, the global
        NumPy random state is used.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (``n_features_in_``,)
        Feature names seen during fit. Defined only when X has feature names.

    support_mask_ : ndarray of shape (``n_features_in_``,), dtype=bool
        Boolean mask of selected features determined during fit.

    Tn_path_ : ndarray of shape (n_selected,)
        Values of the cumulative T_n along the selection path.

    References
    ----------
    Mona Azadkia and Sourav Chatterjee. A simple measure of conditional dependence.
    The Annals of Statistics, 49(6):3070–3102, 2021. https://doi.org/10.1214/21-AOS2073

    R FOCI package (reference implementation): https://cran.r-project.org/package=FOCI

    Sebastian Fuchs. Quantifying directed dependence via dimension reduction.
    Journal of Multivariate Analysis 201 (2024): 105266. https://doi.org/10.1016/j.jmva.2023.105266
    """

    _parameter_constraints = {
        "max_features": [None, Interval(Integral, 1, None, closed="left")],
        "min_delta": [None, Interval(Real, None, None, closed="neither")],
        "standardize": [None, StrOptions({"normalize"})],
        "nn_strategy": [StrOptions({"grouping", "radius"})],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        max_features=None,
        min_delta=0,
        standardize="normalize",
        nn_strategy="grouping",
        random_state=None,
    ):
        self.max_features = max_features
        self.min_delta = min_delta
        self.standardize = standardize
        self.nn_strategy = nn_strategy
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the selector by hierarchical forward selection maximizing T_n
        over the growing feature set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        if y is None:
            raise ValueError("y must be provided for feature selection.")
        type_of_target(y, input_name="y", raise_unknown=True)

        X, y = validate_data(
            self, X, y, accept_sparse=False, y_numeric=True
        )  # asserts finite values

        # Standardization (if requested)
        if self.standardize == "normalize":
            X_mean = np.mean(X, axis=0, dtype=float)
            X_std = np.std(X, axis=0, dtype=float, ddof=0)
            # Avoid division by zero: leave zero-variance columns unchanged
            safe_std = X_std.copy()
            safe_std[safe_std == 0] = 1.0
            X = (X - X_mean) / safe_std

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise InvalidParameterError(
                "Just one sample provided. Need at least two for nearest neighbors."
            )

        y_rank = _rank_max(y)
        random_state = check_random_state(self.random_state)

        max_features = n_features if self.max_features is None else self.max_features

        selected = []  # S_k
        Tn_path = []
        remaining = list(range(n_features))
        Tn_prev = -np.inf

        # Forward selection up to max_features
        while remaining and (len(selected) < max_features):
            best_j = None
            best_Tn = -np.inf

            for j in remaining:
                sel_candidate = selected + [j]
                X_sub = X[:, sel_candidate]
                Tn_val = _Tn(X_sub, y_rank, random_state, nn_strategy=self.nn_strategy)
                if Tn_val > best_Tn:
                    best_Tn = Tn_val
                    best_j = j

            # Early stopping behavior controlled by self.min_delta
            if self.min_delta is not None:
                # First step: if best_Tn <= 0 + min_delta, select nothing and return
                if len(selected) == 0 and best_Tn <= 0 + self.min_delta:
                    self.selected_indices_ = np.asarray([], dtype=int)
                    self.Tn_path_ = np.asarray([], dtype=float)
                    mask = np.zeros(n_features, dtype=bool)
                    self.support_mask_ = mask
                    return self
                # Subsequent steps: stop if no sufficient improvement
                if len(selected) > 0 and best_Tn <= Tn_prev + self.min_delta:
                    break

            # Always add the best feature this round
            selected.append(best_j)
            Tn_path.append(best_Tn)
            remaining.remove(best_j)
            Tn_prev = best_Tn

        # Persist learned attributes
        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.Tn_path_ = np.asarray(Tn_path, dtype=float)

        # Build mask
        mask = np.zeros(n_features, dtype=bool)
        mask[self.selected_indices_] = True
        self.support_mask_ = mask

        return self

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected.
        """
        # SelectorMixin will call this during transform/get_support
        return self.support_mask_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        return tags
