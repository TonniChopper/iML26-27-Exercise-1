"""Interpretability metrics.

Our main metric is the **Interpretability Load**: how many conditions a person has to deal with to
understand a tree-based model. It has three parts (see README.md for the full explanation):

* :class:`Cases` – effective number of typical cases (leaf combinations) the model uses on the data,
* :class:`Conditions` – average number of distinct features tested to reach one prediction,
* :class:`Exceptions` – average number of splits on the way that contradict their feature's usual direction.

They are combined into

* :class:`LocalLoad` = Conditions + lambda * Exceptions  (effort to trace one prediction, "with pen and paper"),
* :class:`GlobalLoad` = Cases * LocalLoad                 (effort to understand the whole model "in your head").

For all load metrics, **lower means more interpretable**. :class:`AverageDepth`, :class:`NumberOfLeaves`
and :class:`SimulationEffort` are simple baselines for comparison.

All metrics support decision trees, random forests and gradient-boosted trees, binary and multiclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import ClassifierMixin

from iml_interpretability.data import Dataset
from iml_interpretability.trees import feature_matrix, get_trees, leaf_ids, node_values, path_matrix

#: Changes of a node output smaller than this are treated as "no direction".
EPS = 1e-3
#: Default weight of an exception compared to a normal condition.
DEFAULT_LAMBDA = 2.0


class InterpretabilityMetric(ABC):
    """Base class for a metric that quantifies how interpretable a fitted model is."""

    #: Column name under which the score is reported.
    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score of a fitted model.

        :param model: A model fitted on ``dataset``.
        :param dataset: The dataset the model was fitted on.
        :return: The interpretability score. Document whether higher means more or less interpretable.
        :rtype: float
        """


def _as_array(dataset: Dataset) -> np.ndarray:
    # scikit-learn trees work internally with C-contiguous float32 arrays, so we pass the same values.
    return np.ascontiguousarray(dataset.X, dtype=np.float32)


# --------------------------------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------------------------------


def effective_number_of_cases(model: ClassifierMixin, X: np.ndarray) -> float:
    """exp(entropy) of how samples are spread over the leaf combinations of all trees.

    Every sample lands in one leaf per tree; the tuple of these leaves is the sample's "case".
    If all samples share 3 cases equally, the result is 3. Cases that almost no sample reaches
    count very little.
    """
    leaves = np.column_stack([leaf_ids(view.tree, X) for view in get_trees(model)])
    _, counts = np.unique(leaves, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def conditions_per_prediction(model: ClassifierMixin, X: np.ndarray) -> float:
    """Average number of distinct features tested on a sample's paths, summed over all trees.

    A feature that is tested several times on one path (e.g. ``age < 50`` and then ``age < 30``)
    counts once, because a person reads this as one condition.
    """
    total = np.zeros(len(X))
    for view in get_trees(model):
        used = path_matrix(view.tree, X) @ feature_matrix(view.tree, X.shape[1])  # (samples, features)
        total += (used > 0).sum(axis=1).A1
    return float(total.mean())


def _dominant_directions(model: ClassifierMixin) -> dict[tuple[int, int], float]:
    """Sample-weighted vote over all splits: does a larger feature value push towards a class?"""
    vote: dict[tuple[int, int], float] = {}
    for view in get_trees(model):
        t = view.tree.tree_
        values = node_values(view.tree)
        for node in np.where(t.children_left != -1)[0]:
            delta = values[t.children_right[node]] - values[t.children_left[node]]
            for k, d in zip(view.outputs, delta, strict=True):
                key = (t.feature[node], k)
                vote[key] = vote.get(key, 0.0) + t.weighted_n_node_samples[node] * d
    return vote


def exceptions_per_prediction(model: ClassifierMixin, X: np.ndarray) -> float:
    """Average number of splits on a sample's paths that go against their feature's usual direction.

    The usual direction is defined per (feature, class). Every split gets a contradiction score in
    [0, 1]: the share of its effect (summed over classes) that goes the "wrong" way. For binary
    problems this score is exactly 0 or 1.
    """
    vote = _dominant_directions(model)
    total = np.zeros(len(X))
    for view in get_trees(model):
        t = view.tree.tree_
        values = node_values(view.tree)
        contradiction = np.zeros(t.node_count)
        for node in np.where(t.children_left != -1)[0]:
            delta = values[t.children_right[node]] - values[t.children_left[node]]
            size = np.abs(delta).sum()
            if size <= EPS:
                continue
            wrong = sum(
                abs(d)
                for k, d in zip(view.outputs, delta, strict=True)
                if abs(d) > EPS and np.sign(d) != np.sign(vote[(t.feature[node], k)])
            )
            contradiction[node] = wrong / size
        total += path_matrix(view.tree, X) @ contradiction
    return float(total.mean())


# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------


class Cases(InterpretabilityMetric):
    """Effective number of cases the model uses on the data. Lower = more interpretable."""

    name = "cases"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        return effective_number_of_cases(model, _as_array(dataset))


class Conditions(InterpretabilityMetric):
    """Average number of distinct features tested to reach one prediction. Lower = more interpretable."""

    name = "conditions"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        return conditions_per_prediction(model, _as_array(dataset))


class Exceptions(InterpretabilityMetric):
    """Average number of contradicting splits met on the way to one prediction. Lower = more interpretable."""

    name = "exceptions"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        return exceptions_per_prediction(model, _as_array(dataset))


class LocalLoad(InterpretabilityMetric):
    """Conditions + lambda * Exceptions: effort to trace one prediction. Lower = more interpretable."""

    def __init__(self, lam: float = DEFAULT_LAMBDA):
        self.lam = lam
        self.name = "local_load" if lam == DEFAULT_LAMBDA else f"local_load_lam{lam:g}"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        X = _as_array(dataset)
        return conditions_per_prediction(model, X) + self.lam * exceptions_per_prediction(model, X)


class GlobalLoad(InterpretabilityMetric):
    """Cases * LocalLoad: effort to understand the whole model. Lower = more interpretable.

    This is our main metric. We consider a model interpretable if its global load is at most
    :data:`INTERPRETABILITY_THRESHOLD` (motivated by the "7 +- 2" capacity of working memory).
    """

    def __init__(self, lam: float = DEFAULT_LAMBDA):
        self.lam = lam
        self.name = "global_load" if lam == DEFAULT_LAMBDA else f"global_load_lam{lam:g}"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        X = _as_array(dataset)
        local = conditions_per_prediction(model, X) + self.lam * exceptions_per_prediction(model, X)
        return effective_number_of_cases(model, X) * local


#: Global load up to which we call a model interpretable.
INTERPRETABILITY_THRESHOLD = 10.0


# --------------------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------------------


class NumberOfLeaves(InterpretabilityMetric):
    """Total number of leaves over all trees (simple size baseline). Lower = more interpretable."""

    name = "n_leaves"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        return float(sum(view.tree.get_n_leaves() for view in get_trees(model)))


class AverageDepth(InterpretabilityMetric):
    """Average number of tests on a sample's paths, summed over trees (= weighted average depth).

    Baseline from the literature (weighted average depth / operation count). Lower = more interpretable.
    """

    name = "avg_depth"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        X = _as_array(dataset)
        total = np.zeros(len(X))
        for view in get_trees(model):
            is_split = (view.tree.tree_.children_left != -1).astype(float)
            total += path_matrix(view.tree, X) @ is_split
        return float(total.mean())


class SimulationEffort(InterpretabilityMetric):
    """Tests + feature switches on a sample's paths (the "CSE" idea from our group). Lower = more interpretable."""

    name = "simulation_effort"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        X = _as_array(dataset)
        total = np.zeros(len(X))
        for view in get_trees(model):
            paths = path_matrix(view.tree, X)
            feature = view.tree.tree_.feature
            for i in range(len(X)):
                features = [feature[n] for n in paths.indices[paths.indptr[i] : paths.indptr[i + 1]] if feature[n] >= 0]
                total[i] += len(features) + sum(a != b for a, b in zip(features, features[1:], strict=False))
        return float(total.mean())


def default_metrics() -> list[InterpretabilityMetric]:
    """All metrics reported in our experiments."""
    return [
        Cases(),
        Conditions(),
        Exceptions(),
        LocalLoad(),
        GlobalLoad(),
        NumberOfLeaves(),
        AverageDepth(),
        SimulationEffort(),
    ]
