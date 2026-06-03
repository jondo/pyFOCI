"""
Feature Ordering by Conditional Independence (stub selector)
"""

# Authors: Robert Pollak <robert.pollak@jku.at>
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, _fit_context
from sklearn.feature_selection import SelectorMixin
from sklearn.utils._param_validation import Integral, Interval
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data


class FOCISelector(SelectorMixin, BaseEstimator):
    """
    Stub feature selector following scikit-learn's SelectorMixin API.

    This stub selects up to `max_features` columns with the largest absolute
    Pearson correlation to the target y (a simple proxy for the real FOCI method).

    Parameters
    ----------
    max_features : int, default=4
        Maximum number of features to select.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (``n_features_in_``,)
        Feature names seen during fit. Defined only when X has feature names.

    support_mask_ : ndarray of shape (``n_features_in_``,), dtype=bool
        Boolean mask of selected features determined during fit.

    correlation_ : ndarray of shape (``n_features_in_``,)
        Absolute Pearson correlation between each feature and y computed at fit.
    """

    _parameter_constraints = {
        "max_features": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, max_features=4):
        self.max_features = max_features

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the selector by ranking features via absolute correlation with y.

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

        X = validate_data(self, X, accept_sparse=False)
        y = np.asarray(y).ravel()

        n_samples, n_features = X.shape

        # Compute absolute Pearson correlation for each feature with y.
        corr = np.empty(n_features, dtype=float)
        for j in range(n_features):
            xj = X[:, j]
            # Handle constant columns to avoid division by zero in correlation
            if np.std(xj) == 0 or np.std(y) == 0:
                corr[j] = 0.0
            else:
                corr[j] = abs(np.corrcoef(xj, y)[0, 1])

        self.correlation_ = corr

        k = min(self.max_features, n_features)
        top_idx = np.argsort(-corr)[:k]

        mask = np.zeros(n_features, dtype=bool)
        mask[top_idx] = True
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
