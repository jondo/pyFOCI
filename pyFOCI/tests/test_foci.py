"""Pytest for FOCI variable selection using a deterministic synthetic dataset."""

# Authors: Robert Pollak <robert.pollak@jku.at>
# License: BSD 3 clause

import re

import numpy as np
import pandas as pd
import pytest

from pyFOCI import FOCISelector


def make_demo_data(n: int = 200, p: int = 30, seed: int = 0):
    """
    Create a deterministic small dataset for feature selection tests,
    with n entries and p features per entry.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
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

    selector = FOCISelector()
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

    assert np.allclose(X_trans, expected)


def test_fit_raises_when_y_is_none():
    X = np.arange(10.0).reshape(-1, 1)
    sel = FOCISelector()
    with pytest.raises(ValueError, match="y must be provided"):
        sel.fit(X, y=None)


@pytest.mark.parametrize("max_features", [0, -1, None])
def test_fit_raises_when_max_features_invalid(max_features):
    n, p = 12, 5
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    sel = FOCISelector(max_features=max_features)
    expected = (
        "The 'max_features' parameter of FOCISelector must be an "
        "int in the range [1, inf)."
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        sel.fit(X, y)
