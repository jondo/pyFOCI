"""This file will just show how to write tests for the template classes."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose

from pyFOCI import TemplateTransformer

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_template_transformer(data):
    """Check the internals and behaviour of `TemplateTransformer`."""
    X, y = data
    trans = TemplateTransformer()
    assert trans.demo_param == "demo"

    trans.fit(X)
    assert trans.n_features_in_ == X.shape[1]

    X_trans = trans.transform(X)
    assert_allclose(X_trans, np.sqrt(X))

    X_trans = trans.fit_transform(X)
    assert_allclose(X_trans, np.sqrt(X))
