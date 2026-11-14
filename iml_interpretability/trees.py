"""Helpers that look inside fitted scikit-learn tree models.

All metrics in :mod:`iml_interpretability.metric` work on the same view of a model: a list of single
trees, where every tree knows which class each of its node outputs belongs to. This way decision trees,
random forests and gradient-boosted trees (binary and multiclass) are handled by the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier

SUPPORTED_MODELS = (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)


@dataclass(frozen=True)
class TreeView:
    """One tree of a model.

    :param tree: A fitted single tree (classifier or, inside gradient boosting, regressor).
    :param outputs: Class index of every column of the tree's node values.
    """

    tree: BaseDecisionTree
    outputs: list[int]


def get_trees(model: ClassifierMixin) -> list[TreeView]:
    """Split a fitted model into its single trees.

    :param model: A fitted decision tree, random forest or gradient-boosting classifier.
    :return: One :class:`TreeView` per tree.
    :rtype: list[TreeView]
    """
    if isinstance(model, DecisionTreeClassifier):
        return [TreeView(model, list(range(model.n_classes_)))]
    if isinstance(model, RandomForestClassifier):
        return [TreeView(tree, list(range(model.n_classes_))) for tree in model.estimators_]
    if isinstance(model, GradientBoostingClassifier):
        # One regression tree per boosting round and class (a single column for binary problems).
        return [TreeView(tree, [k]) for row in model.estimators_ for k, tree in enumerate(row)]
    raise TypeError(f"Unsupported model {type(model).__name__}, expected one of {SUPPORTED_MODELS}")


def node_values(tree: BaseDecisionTree) -> np.ndarray:
    """Output of every node, shape ``(n_nodes, n_outputs)``.

    For classification trees these are class fractions, for the regression trees inside gradient
    boosting these are the boosting scores. In both cases "larger" means "more towards that class".
    """
    values = tree.tree_.value[:, 0, :]
    if values.shape[1] == 1:
        return values
    return values / values.sum(axis=1, keepdims=True)


def path_matrix(tree: BaseDecisionTree, X: np.ndarray) -> sparse.csr_matrix:
    """Sparse 0/1 matrix ``(n_samples, n_nodes)``: which nodes each sample passes through.

    ``X`` must be a C-contiguous float32 array, as scikit-learn uses internally.
    """
    return tree.tree_.decision_path(X).tocsr()


def leaf_ids(tree: BaseDecisionTree, X: np.ndarray) -> np.ndarray:
    """Index of the leaf every sample ends up in."""
    return tree.tree_.apply(X)


def feature_matrix(tree: BaseDecisionTree, n_features: int) -> sparse.csr_matrix:
    """Sparse 0/1 matrix ``(n_nodes, n_features)``: which feature each internal node tests."""
    feature = tree.tree_.feature
    internal = np.where(feature >= 0)[0]
    data = np.ones(len(internal))
    return sparse.csr_matrix((data, (internal, feature[internal])), shape=(tree.tree_.node_count, n_features))
