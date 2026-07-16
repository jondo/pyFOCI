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
from pyFOCI._foci import _nn_grouping_based, _nn_radius_based, _Tn


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


def collect_nn_ties(nn_func, X):
    """
    Collect nearest-neighbor tie sets produced by an NN traversal helper.

    The NN helper API aggregates each tie set immediately and returns the
    aggregated values. These tests still need to inspect the tie sets directly,
    so this helper uses a test-only aggregator that records each tie set and
    returns a dummy scalar aggregation value.
    """
    nn_ties_by_sample = []

    def record_ties(nn_ties):
        nn_ties_by_sample.append(np.asarray(nn_ties, dtype=int).copy())
        return 0.0

    aggregated = nn_func(X, record_ties)

    assert isinstance(aggregated, np.ndarray)
    assert aggregated.shape == (np.asarray(X).shape[0],)
    assert aggregated.dtype.kind == "f"
    assert len(nn_ties_by_sample) == np.asarray(X).shape[0]

    return nn_ties_by_sample


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
    nn_ties = collect_nn_ties(_nn_grouping_based, X)

    expected = [np.array([1]), np.array([2]), np.array([1]), np.array([2])]
    assert len(nn_ties) == len(expected)
    for got, exp in zip(nn_ties, expected):
        np.testing.assert_array_equal(got, exp)

    # Also verify that no point includes itself as a tied neighbor
    for i, ties in enumerate(nn_ties):
        assert i not in set(ties.tolist())


def test_nn_grouping_based_identical_rows_pair():
    """
    Test _nn_grouping_based when there is an exact pair of identical rows.
    Identical rows have distance 0 between each other and must list each other
    as the (only) member of their tie sets.
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
    nn_ties = collect_nn_ties(_nn_grouping_based, X)

    np.testing.assert_array_equal(nn_ties[0], np.array([2]))
    np.testing.assert_array_equal(nn_ties[2], np.array([0]))
    # For idx 3 [10, 10], closest unique row is [5, 5] (idx 1)
    np.testing.assert_array_equal(nn_ties[3], np.array([1]))


def test_nn_grouping_based_identical_rows_multiple():
    """
    Test _nn_grouping_based when there are 3 or more identical rows.

    For samples in the identical group, the tie set is all other members of that
    group. No random tie-breaking happens in _nn_grouping_based.
    """
    # 4 identical rows [7], 1 distinct row [100]
    X = np.array([[7], [7], [7], [7], [100]])
    nn_ties = collect_nn_ties(_nn_grouping_based, X)

    expected = [
        np.array([1, 2, 3]),
        np.array([0, 2, 3]),
        np.array([0, 1, 3]),
        np.array([0, 1, 2]),
        np.array([0, 1, 2, 3]),
    ]
    assert len(nn_ties) == len(expected)
    for got, exp in zip(nn_ties, expected):
        np.testing.assert_array_equal(got, exp)


def test_nn_grouping_based_distance_ties():
    """
    Test _nn_grouping_based when there are distance ties between distinct
    unique rows.
    """
    # In 1D: row 0 (val 0) is equidistant to row 1 (val -3) and row 2 (val 3)
    X_1d = np.array([[0], [-3], [3]])
    nn_1d = collect_nn_ties(_nn_grouping_based, X_1d)

    np.testing.assert_array_equal(np.sort(nn_1d[0]), np.array([1, 2]))
    assert 0 not in set(nn_1d[0].tolist())

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
    nn_2d = collect_nn_ties(_nn_grouping_based, X_2d)

    np.testing.assert_array_equal(np.sort(nn_2d[0]), np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(np.sort(nn_2d[5]), np.array([6, 7, 8, 9]))


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
    nn_ties = collect_nn_ties(_nn_grouping_based, X)

    # Candidate indices for row 0 are {1, 2, 3}
    np.testing.assert_array_equal(np.sort(nn_ties[0]), np.array([1, 2, 3]))

    # Since idx 1 and 2 are identical, their tie sets are each other
    np.testing.assert_array_equal(nn_ties[1], np.array([2]))
    np.testing.assert_array_equal(nn_ties[2], np.array([1]))

    # For idx 3 [-2], closest is [0] (idx 0)
    np.testing.assert_array_equal(nn_ties[3], np.array([0]))


def test_nn_grouping_based_deterministic():
    """
    Test that _nn_grouping_based is deterministic and does not depend on a RNG.
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

    nn_ties_1 = collect_nn_ties(_nn_grouping_based, X)
    nn_ties_2 = collect_nn_ties(_nn_grouping_based, X)

    assert len(nn_ties_1) == len(nn_ties_2)
    for t1, t2 in zip(nn_ties_1, nn_ties_2):
        np.testing.assert_array_equal(np.sort(t1), np.sort(t2))

    for i, ties in enumerate(nn_ties_1):
        assert i not in set(ties.tolist())


def test_nn_grouping_based_all_identical_rows():
    """
    Test behavior when all samples in X are identical.
    Each sample should list all other samples as its tie set.
    """
    n = 10
    X = np.ones((n, 3))
    nn_ties = collect_nn_ties(_nn_grouping_based, X)

    assert len(nn_ties) == n
    for i in range(n):
        np.testing.assert_array_equal(
            nn_ties[i], np.array([j for j in range(n) if j != i])
        )


def test_nn_grouping_based_matches_radius_based_no_ties():
    """
    On random continuous data without ties or identical rows, grouping-based
    and radius-based NN tie sets should match and be singletons.
    """
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(50, 5))

    nn_grouping = collect_nn_ties(_nn_grouping_based, X)
    nn_radius = collect_nn_ties(_nn_radius_based, X)

    assert len(nn_grouping) == len(nn_radius)
    for g, r in zip(nn_grouping, nn_radius):
        np.testing.assert_array_equal(g, r)
        assert g.shape == (1,)


def test_nn_grouping_based_two_samples():
    """
    Test minimal sample size (n_samples == 2).
    With only 2 samples, index 0's tie set must be [1] and index 1's tie set must
    be [0], regardless of whether they are distinct or identical.
    """
    X_distinct = np.array([[10], [20]])
    X_identical = np.array([[5], [5]])

    nn_distinct = collect_nn_ties(_nn_grouping_based, X_distinct)
    nn_identical = collect_nn_ties(_nn_grouping_based, X_identical)

    np.testing.assert_array_equal(nn_distinct[0], np.array([1]))
    np.testing.assert_array_equal(nn_distinct[1], np.array([0]))
    np.testing.assert_array_equal(nn_identical[0], np.array([1]))
    np.testing.assert_array_equal(nn_identical[1], np.array([0]))


@pytest.mark.parametrize("nn_func", [_nn_grouping_based, _nn_radius_based])
def test_nn_helpers_return_aggregated_values(nn_func):
    """
    NN helpers should return one scalar aggregation result per sample.

    The aggregator receives each sample's nearest-neighbor tie set and returns
    the number of tied nearest neighbors. The helper should collect these
    returned scalar values in an ndarray.
    """
    X = np.array([[0.0], [-1.0], [1.0]])

    aggregated = nn_func(X, lambda nn_ties: float(len(nn_ties)))

    assert aggregated.shape == (X.shape[0],)
    assert aggregated.dtype.kind == "f"

    # sample 0 has two equidistant nearest neighbors; samples 1 and 2 each have
    # one nearest neighbor.
    np.testing.assert_array_equal(aggregated, np.array([2.0, 1.0, 1.0]))


def test_nn_grouping_based_returns_aggregated_values_for_identical_rows():
    """
    _nn_grouping_based should aggregate identical-row tie sets immediately.

    In this dataset:
      - samples 0, 1, and 2 are identical, so each has two tied neighbors
      - sample 3 has the identical-row group as nearest neighbors, so it has
        three tied neighbors
    """
    X = np.array([[1.0], [1.0], [1.0], [5.0]])

    aggregated = _nn_grouping_based(X, lambda nn_ties: float(len(nn_ties)))

    np.testing.assert_array_equal(aggregated, np.array([2.0, 2.0, 2.0, 3.0]))


def test_Tn_invalid_tie_breaking_raises():
    X = np.array([[0.0], [1.0], [2.0]])
    y_rank = np.array([1.0, 2.0, 3.0])
    random_state = np.random.RandomState(0)

    with pytest.raises(ValueError, match=re.escape("nn_tie_breaking must be one of")):
        _Tn(
            X,
            y_rank,
            random_state,
            nn_strategy="grouping",
            nn_tie_breaking="invalid",
        )


def test_Tn_mean_tie_breaking_grouping():
    """
    Cover nn_tie_breaking="mean" on a dataset with a true distance tie.

    For X = [[0], [-1], [1]]:
      - sample 0 has two nearest neighbors at equal distance: indices {1, 2}
      - using mean tie-breaking sets neighbor rank for sample 0 to mean([2, 3]) = 2.5
    """
    X = np.array([[0.0], [-1.0], [1.0]])
    y_rank = np.array([1.0, 2.0, 3.0])
    random_state = np.random.RandomState(0)

    tn = _Tn(
        X,
        y_rank,
        random_state,
        nn_strategy="grouping",
        nn_tie_breaking="mean",
    )
    assert np.isfinite(tn)


def test_Tn_mean_tie_breaking_radius():
    """
    Cover nn_strategy="radius" together with nn_tie_breaking="mean".
    """
    X = np.array([[0.0], [-1.0], [1.0]])
    y_rank = np.array([1.0, 2.0, 3.0])
    random_state = np.random.RandomState(0)

    tn = _Tn(
        X,
        y_rank,
        random_state,
        nn_strategy="radius",
        nn_tie_breaking="mean",
    )
    assert np.isfinite(tn)
