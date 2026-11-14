"""Tests that run offline on small synthetic datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability import (
    AverageDepth,
    Cases,
    Conditions,
    Dataset,
    Exceptions,
    GlobalLoad,
    InterpretabilityMetric,
    LocalLoad,
    NumberOfLeaves,
    SimulationEffort,
    default_metrics,
    evaluate,
    get_models,
)


class _ConstantMetric(InterpretabilityMetric):
    name = "constant"

    def score(self, model, dataset):
        return 1.0


def _dataset(n_classes: int = 2) -> Dataset:
    X, y = make_classification(n_samples=200, n_features=6, n_informative=4, n_classes=n_classes, random_state=0)
    return Dataset(name="synthetic", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))


@pytest.fixture
def dataset() -> Dataset:
    return _dataset()


def test_evaluate_returns_one_row_per_dataset_and_model(dataset):
    models = get_models()
    results = evaluate([dataset], models, [_ConstantMetric()])
    assert len(results) == len(models)
    assert set(results.columns) == {"dataset", "model", "accuracy", "constant"}


def test_stump_has_known_values(dataset):
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(dataset.X, dataset.y)
    assert Conditions().score(stump, dataset) == 1
    assert Exceptions().score(stump, dataset) == 0  # one split can't contradict itself
    assert 1 < Cases().score(stump, dataset) <= 2  # two leaves
    assert AverageDepth().score(stump, dataset) == 1
    assert NumberOfLeaves().score(stump, dataset) == 2


def test_cases_equal_two_for_balanced_stump():
    X = pd.DataFrame({"x": np.arange(100, dtype=float)})
    y = pd.Series((X["x"] >= 50).astype(int))
    data = Dataset("line", X, y)
    stump = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert Cases().score(stump, data) == pytest.approx(2.0)


def test_global_load_is_cases_times_local_load(dataset):
    model = DecisionTreeClassifier(max_depth=4, random_state=0).fit(dataset.X, dataset.y)
    expected = Cases().score(model, dataset) * LocalLoad().score(model, dataset)
    assert GlobalLoad().score(model, dataset) == pytest.approx(expected)


def test_global_load_grows_with_depth_and_ensemble_size(dataset):
    loads = [
        GlobalLoad().score(DecisionTreeClassifier(max_depth=d, random_state=0).fit(dataset.X, dataset.y), dataset)
        for d in [1, 2, 4]
    ]
    assert loads == sorted(loads)
    small = RandomForestClassifier(3, max_depth=3, random_state=0).fit(dataset.X, dataset.y)
    large = RandomForestClassifier(30, max_depth=3, random_state=0).fit(dataset.X, dataset.y)
    assert GlobalLoad().score(small, dataset) < GlobalLoad().score(large, dataset)


def test_contradicting_split_is_counted():
    # Class 1 for x in [0, 1) and [2, 3): a larger x first helps, then hurts, then helps again.
    X = pd.DataFrame({"x": np.linspace(0, 3, 300, endpoint=False)})
    y = pd.Series(((X["x"] < 1) | (X["x"] >= 2)).astype(int))
    data = Dataset("zigzag", X, y)
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert Exceptions().score(model, data) > 0


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeClassifier(max_depth=3, random_state=0),
        RandomForestClassifier(5, max_depth=3, random_state=0),
        GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=0),
    ],
)
def test_all_metrics_work_for_all_model_types(model, n_classes):
    data = _dataset(n_classes)
    model.fit(data.X, data.y)
    for metric in default_metrics():
        value = metric.score(model, data)
        assert np.isfinite(value) and value >= 0, metric.name


def test_simulation_effort_counts_tests_and_switches(dataset):
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(dataset.X, dataset.y)
    assert SimulationEffort().score(stump, dataset) == 1  # one test, no switch


def test_unsupported_model_raises(dataset):
    model = LogisticRegression().fit(dataset.X, dataset.y)
    with pytest.raises(TypeError):
        GlobalLoad().score(model, dataset)
