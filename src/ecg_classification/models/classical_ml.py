from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_random_forest(random_state: int = 42) -> Pipeline:
    try:
        from imblearn.ensemble import BalancedRandomForestClassifier

        model = BalancedRandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            replacement=True,
            sampling_strategy="all",
        )
    except ModuleNotFoundError:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
        )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def fit_classifier(features: pd.DataFrame, labels: pd.Series, random_state: int = 42) -> Pipeline:
    classifier = build_random_forest(random_state=random_state)
    classifier.fit(features, labels)
    return classifier


def predict_classifier(classifier: Pipeline, features: pd.DataFrame) -> pd.Series:
    return pd.Series(classifier.predict(features))
