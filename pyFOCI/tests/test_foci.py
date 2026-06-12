"""Pytest for FOCI variable selection using a deterministic synthetic dataset."""

# Authors: Robert Pollak <robert.pollak@jku.at>
# License: BSD 3 clause

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils._testing import assert_allclose

from pyFOCI import FOCISelector


def make_demo_data(n: int = 100, p: int = 30, seed: int = 0):
    """
    Create a deterministic small dataset for feature selection tests,
    with n entries and p features per entry.
    """
    random_state = np.random.RandomState(seed)
    X = random_state.normal(size=(n, p))
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    y = (
        X_df.iloc[:, 0] * X_df.iloc[:, 1]
        + np.sin(X_df.iloc[:, 0] * X_df.iloc[:, 2])
        + X_df.iloc[:, 3] ** 2
    )
    return X_df, y.to_numpy()


def test_default_stopping_and_transform():
    """
    Default stopping at small n selects only a few variables, includes column index 3,
    and transform returns the selected columns in the same order.
    """
    X_df, y = make_demo_data(n=300, p=40, seed=0)

    selector = FOCISelector(random_state=0)
    selector.fit(X_df, y)

    names = selector.get_feature_names_out()

    # Expect only a few variables selected
    assert len(names) <= 4

    # Strong marginal effect should be present
    assert "x3" in names

    # Check transform output conforms to scikit-learn conventions
    X_trans = selector.transform(X_df)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == X_df.shape[0]
    assert X_trans.shape[1] == len(names)

    # Build expected output using reported names
    expected = X_df.loc[:, names].to_numpy()

    assert_allclose(X_trans, expected)


def test_stop_true_may_select_none_on_independent_data():
    """
    With stop=True (default), on data where y is independent of X,
    the selector can select no features if the best initial Tn <= 0.
    """
    # Small dataset with y independent of X
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(10, 1))
    y = random_state.normal(size=10)

    selector = FOCISelector(random_state=0, stop=True).fit(X, y)

    # With early stopping enabled, zero features may be selected
    assert selector.support_mask_.sum() == 0
    assert len(selector.Tn_path_) == 0


def test_stop_false_ignores_early_stopping_and_selects_up_to_max():
    """
    With stop=False, early stopping is ignored and features are selected
    up to max_features even on independent data.
    """
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(20, 5))
    y = random_state.normal(size=20)

    selector = FOCISelector(random_state=0, stop=False, max_features=3).fit(X, y)

    assert selector.support_mask_.sum() == 3
    assert len(selector.Tn_path_) == 3


def test_fit_raises_when_y_is_none():
    X = np.arange(10.0).reshape(-1, 1)
    sel = FOCISelector()
    with pytest.raises(ValueError, match="y must be provided"):
        sel.fit(X, y=None)


@pytest.mark.parametrize("max_features", [0, -1])
def test_fit_raises_when_max_features_invalid(max_features):
    n, p = 12, 5
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(n, p))
    y = random_state.normal(size=n)

    sel = FOCISelector(max_features=max_features)
    expected = "The 'max_features' parameter of FOCISelector must be"
    with pytest.raises(InvalidParameterError, match=re.escape(expected)):
        sel.fit(X, y)


def test_random_state_accepts_random_state_instance():
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(20, 3))
    y = random_state.normal(size=20)

    selector = FOCISelector(random_state=np.random.RandomState(0))
    selector.fit(X, y)

    assert selector.support_mask_.shape == (X.shape[1],)


def test_random_state_int_reproducible():
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(30, 5))
    y = random_state.normal(size=30)

    selector_1 = FOCISelector(random_state=0).fit(X, y)
    selector_2 = FOCISelector(random_state=0).fit(X, y)

    np.testing.assert_array_equal(
        selector_1.selected_indices_,
        selector_2.selected_indices_,
    )
    assert_allclose(
        selector_1.Tn_path_,
        selector_2.Tn_path_,
    )
