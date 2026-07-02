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
from pyFOCI._foci import _nn_grouping_based, _nn_radius_based


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
    Default early stopping (min_delta=0) at small n selects only a few variables,
    includes column index 3, and transform returns the selected columns in the
    same order.
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


def test_min_delta_zero_may_select_none_on_independent_data():
    """
    With min_delta=0 (default), on data where y is independent of X,
    the selector can select no features if the best initial Tn <= 0.
    """
    # Small dataset with y independent of X
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(10, 1))
    y = random_state.normal(size=10)

    selector = FOCISelector(random_state=0, min_delta=0).fit(X, y)

    # With early stopping enabled, zero features may be selected
    assert selector.support_mask_.sum() == 0
    assert len(selector.Tn_path_) == 0


def test_min_delta_none_ignores_early_stopping_and_selects_up_to_max():
    """
    With min_delta=None, early stopping is ignored and features are selected
    up to max_features even on independent data.
    """
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(20, 5))
    y = random_state.normal(size=20)

    selector = FOCISelector(random_state=0, min_delta=None, max_features=3).fit(X, y)

    assert selector.support_mask_.sum() == 3
    assert len(selector.Tn_path_) == 3


def test_min_delta_enforces_gap():
    """
    With a positive min_delta, consecutive cumulative Tn values must improve
    by more than min_delta; the first selected Tn must exceed min_delta.
    """
    X_df, y = make_demo_data(n=200, p=10, seed=0)
    min_delta = 0.03

    selector = FOCISelector(random_state=0, min_delta=min_delta).fit(X_df, y)
    tn = selector.Tn_path_
    assert tn.size > 1  # precondition for testing a delta

    assert tn[0] > min_delta

    diffs = np.diff(tn)
    assert np.all(diffs > min_delta)


def test_standardize_none():
    """
    standardize=None is also accepted.
    """
    X_df, y = make_demo_data(n=100, p=10, seed=0)

    FOCISelector(random_state=0, standardize=None).fit(X_df, y)


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


def test_nn_strategy_grouping_and_radius_are_accepted_and_reproducible():
    X_df, y = make_demo_data(n=200, p=10, seed=0)

    for strategy in ("grouping", "radius"):
        selector_1 = FOCISelector(
            random_state=0, nn_strategy=strategy, min_delta=None, max_features=3
        ).fit(X_df, y)
        selector_2 = FOCISelector(
            random_state=0, nn_strategy=strategy, min_delta=None, max_features=3
        ).fit(X_df, y)

        # Non-trivial selection (avoid early stopping selecting none)
        assert selector_1.support_mask_.sum() > 2

        np.testing.assert_array_equal(
            selector_1.selected_indices_,
            selector_2.selected_indices_,
        )
        assert_allclose(
            selector_1.Tn_path_,
            selector_2.Tn_path_,
        )


def test_nn_strategy_invalid_raises():
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(20, 3))
    y = random_state.normal(size=20)

    sel = FOCISelector(nn_strategy="invalid")
    expected = "The 'nn_strategy' parameter of FOCISelector must be"
    with pytest.raises(InvalidParameterError, match=re.escape(expected)):
        sel.fit(X, y)


def test_nn_grouping_based_no_ties():
    """
    Test _nn_grouping_based on a simple dataset where all pairwise distances
    are distinct (no ties and no identical rows).
    """
    # 1D array where nearest neighbors are unique and unambiguous
    # idx 0 (val 0) -> closest is 10 (idx 1)
    # idx 1 (val 10) -> closest is 12 (idx 2)
    # idx 2 (val 12) -> closest is 10 (idx 1)
    # idx 3 (val 30) -> closest is 12 (idx 2)
    X = np.array([[0], [10], [12], [30]])
    random_state = np.random.RandomState(0)
    nbr_i = _nn_grouping_based(X, random_state)

    expected = np.array([1, 2, 1, 2])
    np.testing.assert_array_equal(nbr_i, expected)

    # Also verify that no point selects itself as neighbor
    assert not np.any(nbr_i == np.arange(len(X)))


def test_nn_grouping_based_identical_rows_pair():
    """
    Test _nn_grouping_based when there is an exact pair of identical rows.
    Identical rows have distance 0 between each other and must mutually select
    each other.
    """
    # Indices 0 and 2 are identical [1.0, 2.0]
    X = np.array(
        [
            [1.0, 2.0],  # idx 0
            [5.0, 5.0],  # idx 1
            [1.0, 2.0],  # idx 2
            [10.0, 10.0],  # idx 3
        ]
    )
    random_state = np.random.RandomState(0)
    nbr_i = _nn_grouping_based(X, random_state)

    # Since idx 0 and idx 2 are identical, distance is 0 -> select each other
    assert nbr_i[0] == 2
    assert nbr_i[2] == 0
    # For idx 3 [10, 10], closest unique row is [5, 5] (idx 1)
    assert nbr_i[3] == 1


def test_nn_grouping_based_identical_rows_multiple():
    """
    Test _nn_grouping_based when there are 3 or more identical rows.
    Ties within the identical group must be broken randomly without any
    sample selecting itself.
    """
    # 4 identical rows [7], 1 distinct row [100]
    X = np.array([[7], [7], [7], [7], [100]])
    random_state = np.random.RandomState(0)
    nbr_i = _nn_grouping_based(X, random_state)

    # For rows 0..3, nearest neighbors must be within {0, 1, 2, 3} \ {i}
    for i in range(4):
        assert nbr_i[i] in {0, 1, 2, 3}
        assert nbr_i[i] != i

    # Verify tie-breaking selects varied members across the identical group
    assert len(set(nbr_i[:4])) > 1

    # For row 4 (value 100), closest unique row has indices {0, 1, 2, 3}
    assert nbr_i[4] in {0, 1, 2, 3}


def test_nn_grouping_based_distance_ties():
    """
    Test _nn_grouping_based when there are distance ties between distinct
    unique rows.
    """
    # In 1D: row 0 (val 0) is equidistant to row 1 (val -3) and row 2 (val 3)
    X_1d = np.array([[0], [-3], [3]])
    random_state = np.random.RandomState(0)
    nbr_1d = _nn_grouping_based(X_1d, random_state)

    assert nbr_1d[0] in {1, 2}
    assert nbr_1d[0] != 0

    # In 2D: row 0 is at (0, 0), surrounded by 4 points at Euclidean distance 1;
    # row 5 is at (10, 10), surrounded by 4 points at Euclidean distance 1.
    X_2d = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [10, 10],
            [11, 10],
            [10, 11],
            [9, 10],
            [10, 9],
        ]
    )
    random_state = np.random.RandomState(0)
    nbr_2d = _nn_grouping_based(X_2d, random_state)

    assert nbr_2d[0] in {1, 2, 3, 4}
    assert nbr_2d[5] in {6, 7, 8, 9}
    # Verify tie-breaking selects distinct relative neighbors across query points
    assert (nbr_2d[0] - 0) != (nbr_2d[5] - 5)


def test_nn_grouping_based_combined_ties_and_identical_rows():
    """
    Test _nn_grouping_based when there is a distance tie between two different
    unique rows, where one of the unique rows has multiple identical copies.
    """
    # Row 0 is [0]. Minimal distance is 2, achieved by [2] (indices 1 and 2)
    # and [-2] (index 3).
    # All indices belonging to tied unique rows should be pooled as candidates.
    X = np.array(
        [
            [0],  # idx 0
            [2],  # idx 1
            [2],  # idx 2
            [-2],  # idx 3
        ]
    )
    random_state = np.random.RandomState(0)
    nbr_i = _nn_grouping_based(X, random_state)

    # Candidate indices for row 0 are {1, 2, 3}
    assert nbr_i[0] in {1, 2, 3}
    # Since idx 1 and 2 are identical, they must select each other
    assert nbr_i[1] == 2
    assert nbr_i[2] == 1
    # For idx 3 [-2], closest is [0] (idx 0)
    assert nbr_i[3] == 0


def test_nn_grouping_based_reproducibility():
    """
    Test that _nn_grouping_based yields reproducible neighbor assignments
    for the same seed, even with distance ties and identical rows.
    """
    X = np.array(
        [
            [0, 0],
            [1, 1],
            [1, 1],
            [1, 1],
            [-1, -1],
            [0, 0],
            [2, 2],
            [-2, -2],
            [-1, -1],
        ]
    )

    nbr_i_1 = _nn_grouping_based(X, random_state=np.random.RandomState(0))
    nbr_i_2 = _nn_grouping_based(X, random_state=np.random.RandomState(0))

    np.testing.assert_array_equal(nbr_i_1, nbr_i_2)
    assert not np.any(nbr_i_1 == np.arange(len(X)))


def test_nn_grouping_based_all_identical_rows():
    """
    Test behavior when all samples in X are identical.
    Each sample should pick a random neighbor among all other samples.
    """
    n = 10
    X = np.ones((n, 3))
    random_state = np.random.RandomState(0)
    nbr_i = _nn_grouping_based(X, random_state)

    assert nbr_i.shape == (n,)
    assert not np.any(nbr_i == np.arange(n))
    assert np.all((nbr_i >= 0) & (nbr_i < n))
    assert len(set(nbr_i)) > 1


def test_nn_grouping_based_matches_radius_based_no_ties():
    """
    On random continuous data without ties or identical rows, grouping-based
    and radius-based NN selection should yield identical assignments.
    """
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(50, 5))

    nbr_grouping = _nn_grouping_based(X, random_state=np.random.RandomState(0))
    nbr_radius = _nn_radius_based(X, random_state=np.random.RandomState(0))

    np.testing.assert_array_equal(nbr_grouping, nbr_radius)


def test_nn_grouping_based_two_samples():
    """
    Test minimal sample size (n_samples == 2).
    With only 2 samples, index 0 must select 1 and index 1 must select 0,
    regardless of whether they are distinct or identical.
    """
    X_distinct = np.array([[10], [20]])
    X_identical = np.array([[5], [5]])

    random_state = np.random.RandomState(0)
    np.testing.assert_array_equal(_nn_grouping_based(X_distinct, random_state), [1, 0])
    np.testing.assert_array_equal(_nn_grouping_based(X_identical, random_state), [1, 0])
