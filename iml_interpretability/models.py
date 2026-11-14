"""Tree-based models we compare.

All models are from one model class (trees and tree ensembles). We vary their complexity step by step
to find the point where a model stops being interpretable:

* single decision trees with depth 1 to 10 and unlimited depth,
* random forests with 1 to 100 trees of depth 3,
* gradient boosting with 1 to 100 rounds of depth-2 trees.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

TREE_DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, None]
FOREST_SIZES = [1, 3, 10, 30, 100]
BOOSTING_ROUNDS = [1, 3, 10, 30, 100]


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return unfitted models, keyed by a unique name used in result tables.

    Names look like ``dt_d3`` (tree of depth 3), ``rf_n10`` (forest with 10 trees) or ``gb_n30``
    (boosting with 30 rounds).

    :param seed: Random seed passed to every model.
    :return: Mapping from model name to unfitted scikit-learn classifier.
    :rtype: dict[str, ClassifierMixin]
    """
    models: dict[str, ClassifierMixin] = {}
    for depth in TREE_DEPTHS:
        name = f"dt_d{depth}" if depth is not None else "dt_full"
        models[name] = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    for n in FOREST_SIZES:
        models[f"rf_n{n}"] = RandomForestClassifier(n_estimators=n, max_depth=3, random_state=seed)
    for n in BOOSTING_ROUNDS:
        models[f"gb_n{n}"] = GradientBoostingClassifier(n_estimators=n, max_depth=2, random_state=seed)
    return models


def model_family(name: str) -> str:
    """``"dt"``, ``"rf"`` or ``"gb"`` for a model name from :func:`get_models`."""
    return name.split("_")[0]
