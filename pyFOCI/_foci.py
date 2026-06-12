"""
Feature Ordering by Conditional Independence (FOCI)
"""

# Authors: Robert Pollak <robert.pollak@jku.at>
# License: BSD 3 clause

import numpy as np
import sklearn.neighbors
from sklearn.base import BaseEstimator, _fit_context
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Integral, Interval, InvalidParameterError
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


def _Tn(X_sub, y_rank, random_state):
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

    Returns
    -------
    Tn : float
        Value of the :math:`T_n` statistic for ``X_sub`` and ``y_rank``.
    """
    X_sub = np.asarray(X_sub)
    n = X_sub.shape[0]
    # Fit NN on X_sub
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
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

    # Apply the formula (indices are 0-based; y_rank is 1-based)
    term1 = np.sum(np.abs(y_rank - y_rank[nbr_i]))
    term2 = np.sum(y_rank[nbr_i]) + np.sum(y_rank) - n * (n + 1)
    result = 1 - 3 / (n**2 - 1) * term1 + 3 / (n**2 - 1) * term2
    return float(result)


class FOCISelector(SelectorMixin, BaseEstimator):
    """
    Feature selector using hierarchical forward selection based on the
    Azadkia-Chatterjee T_n coefficient.

    At each step, among remaining features, we choose the feature that maximizes
    the cumulative T_n on the growing set S_k = S_{k-1} ∪ {j}.

    Stopping behavior is controlled by the `stop` flag:
      - If stop=True, after the first scan, if the best T_n <= 0, no variable
        is selected. At subsequent steps, if the best T_n does not improve
        over the previous best, selection stops immediately.
      - If stop=False, early stopping is ignored and features are selected
        up to `max_features`, ordered by decreasing T_n at each step.

    Parameters
    ----------
    max_features : int or None, default=None
        Maximum number of features to select. If None, no hard cap is applied
        and selection proceeds until early stopping (if `stop=True`) or until
        all features are selected (if `stop=False`).

    stop : bool, default=True
        Whether to apply early stopping based on improvements in T_n.

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
    """

    _parameter_constraints = {
        "max_features": [None, Interval(Integral, 1, None, closed="left")],
        "stop": [bool],
        "random_state": ["random_state"],
    }

    def __init__(self, max_features=None, stop=True, random_state=None):
        self.max_features = max_features
        self.stop = stop
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
                Tn_val = _Tn(X_sub, y_rank, random_state)
                if Tn_val > best_Tn:
                    best_Tn = Tn_val
                    best_j = j

            # Early stopping behavior controlled by self.stop
            if self.stop:
                # First step: if best_Tn <= 0, select nothing and return
                if len(selected) == 0 and best_Tn <= 0:
                    self.selected_indices_ = np.asarray([], dtype=int)
                    self.Tn_path_ = np.asarray([], dtype=float)
                    mask = np.zeros(n_features, dtype=bool)
                    self.support_mask_ = mask
                    return self
                # Subsequent steps: stop if no improvement
                if best_Tn <= Tn_prev:
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
