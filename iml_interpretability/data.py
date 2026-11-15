"""Dataset loading.

Every dataset is returned as a :class:`Dataset` with numeric, imputed features so that all models in
:mod:`iml_interpretability.models` can be fit on it directly.

We use three small, well-known datasets that ship with scikit-learn (no download needed) and two synthetic
datasets with label noise. Titanic from OpenML is available as an option but not used by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn import datasets as skdata
from sklearn.datasets import fetch_openml

# OpenML downloads are cached here, relative to the current working directory.
DATA_HOME = Path("data")


@dataclass(frozen=True)
class Dataset:
    """A preprocessed tabular classification dataset.

    :param name: Human-readable dataset name, used in result tables.
    :param X: Numeric feature matrix without missing values.
    :param y: Target vector.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


# Columns that leak the target (boat = lifeboat, body = body identification number) or are
# free-text identifiers without a meaningful encoding.
_TITANIC_DROP_COLUMNS = ["boat", "body", "home.dest", "name", "ticket", "cabin"]


def load_titanic(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the Titanic dataset from OpenML (data id 40945), binary.

    The download is cached in ``data_home``, so network access is only needed on the first call.
    """
    # The default pandas ARFF parser crashes on Titanic's integer-valued categorical columns.
    bunch = fetch_openml("titanic", version=1, as_frame=True, parser="liac-arff", data_home=data_home)
    X = bunch.data.drop(columns=_TITANIC_DROP_COLUMNS)
    y = bunch.target.astype(int).rename("survived")

    numeric_columns = X.select_dtypes(include="number").columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    return Dataset(name="titanic", X=X, y=y)


def _from_sklearn(name: str, loader) -> Dataset:
    bunch = loader(as_frame=True)
    return Dataset(name=name, X=bunch.data, y=bunch.target.rename("target"))


def load_breast_cancer() -> Dataset:
    """Breast cancer Wisconsin: 569 samples, 30 features, 2 classes."""
    return _from_sklearn("breast_cancer", skdata.load_breast_cancer)


def load_wine() -> Dataset:
    """Wine: 178 samples, 13 features, 3 classes."""
    return _from_sklearn("wine", skdata.load_wine)


def load_iris() -> Dataset:
    """Iris: 150 samples, 4 features, 3 classes."""
    return _from_sklearn("iris", skdata.load_iris)


def make_noisy(n_classes: int = 2, seed: int = 0) -> Dataset:
    """Synthetic data with 10 % flipped labels, to see what happens when models start to fit noise.

    2000 samples, 10 features (6 informative).
    """
    X, y = skdata.make_classification(
        n_samples=2000, n_features=10, n_informative=6, n_classes=n_classes, flip_y=0.1, random_state=seed
    )
    X = pd.DataFrame(X).add_prefix("x")
    return Dataset(name=f"noisy_{n_classes}class", X=X, y=pd.Series(y, name="target"))


def get_datasets(include_openml: bool = False) -> list[Dataset]:
    """All datasets used in our experiments.

    :param include_openml: Also load Titanic from OpenML (not used in our reported results). If the
        download fails (e.g. no internet), Titanic is skipped with a message.
    """
    datasets = [load_iris(), load_wine(), load_breast_cancer(), make_noisy(2), make_noisy(4)]
    if include_openml:
        try:
            datasets.append(load_titanic())
        except Exception as error:  # noqa: BLE001 - any download problem just means "skip Titanic"
            print(f"Skipping Titanic, could not load it from OpenML: {error}")
    return datasets
