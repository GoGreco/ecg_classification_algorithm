"""Treina e compara LSTM contínua e simbólica com coarse-graining ``tau=5``.

As duas entradas têm 50 posições e usam exatamente a mesma divisão por
registro. A tarefa é binária: ``N`` contra ``não-N``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.config import ExperimentConfig, ProjectPaths
from lstm_training import montar_dataset
from symbolic_lstm_training import escolher_particao_por_entropia, simbolizar_janelas


SCALE = 5
WINDOW_SIZE = 250
COARSE_SIZE = WINDOW_SIZE // SCALE
EPOCHS = 30
BATCH_SIZE = 256


def coarse_grain(X: np.ndarray) -> np.ndarray:
    """Normaliza cada janela e calcula a média de blocos não sobrepostos."""
    minimo = X.min(axis=1, keepdims=True)
    maximo = X.max(axis=1, keepdims=True)
    amplitude = maximo - minimo
    amplitude[amplitude == 0] = 1.0
    normalized = (X - minimo) / amplitude
    return normalized.reshape(len(X), COARSE_SIZE, SCALE).mean(axis=2).astype("float32")


def split_by_record(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Escolhe uma divisão fixa por registro, com ambas as classes em cada parte."""
    from sklearn.model_selection import StratifiedGroupKFold

    _, numeric = np.unique(y, return_inverse=True)
    global_distribution = np.bincount(numeric, minlength=2) / len(y)
    candidates = []
    for attempt in range(64):
        outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed + attempt)
        train_val, test = next(outer.split(X, y, groups))
        inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + attempt + 1)
        relative_train, validation = next(
            inner.split(X[train_val], y[train_val], groups[train_val])
        )
        train = train_val[relative_train]
        validation = train_val[validation]
        parts = (train, validation, test)
        counts = [np.bincount(y[index], minlength=2) for index in parts]
        if not all(np.all(count > 0) for count in counts):
            continue
        error = sum(np.abs(count / count.sum() - global_distribution).sum() for count in counts)
        candidates.append((error, parts))
    if not candidates:
        raise ValueError("Não foi encontrada divisão por registro com as duas classes em todos os conjuntos.")
    return min(candidates, key=lambda item: item[0])[1]


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    best = (float("-inf"), 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
        prediction = (probability >= threshold).astype("int32")
        tp = ((prediction == 1) & (y_true == 1)).sum()
        fp = ((prediction == 1) & (y_true == 0)).sum()
        fn = ((prediction == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
        if f2 > best[0]:
            best = (float(f2), float(threshold))
    return best[1]


def build_model(kind: str, n_symbols: int = 7):
    import tensorflow as tf

    if kind == "continuous":
        inputs = tf.keras.Input(shape=(COARSE_SIZE, 1), dtype="float32")
        x = tf.keras.layers.LSTM(32, return_sequences=True)(inputs)
    else:
        inputs = tf.keras.Input(shape=(COARSE_SIZE,), dtype="int32")
        x = tf.keras.layers.CategoryEncoding(num_tokens=n_symbols, output_mode="one_hot")(inputs)
        x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(16)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_one(
    kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    seed: int,
    n_symbols: int = 7,
) -> tuple[pd.DataFrame, float]:
    import tensorflow as tf
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        classification_report,
        roc_auc_score,
    )
    from sklearn.utils.class_weight import compute_class_weight

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(kind, n_symbols=n_symbols)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=2,
    )
    validation_probability = model.predict(X_validation, verbose=0).ravel()
    threshold = choose_threshold(y_validation, validation_probability)
    test_probability = model.predict(X_test, verbose=0).ravel()
    prediction = (test_probability >= threshold).astype("int32")
    report = pd.DataFrame(
        classification_report(
            y_test,
            prediction,
            labels=[0, 1],
            target_names=["N", "nao_N"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report.index.name = "classe"
    report.insert(0, "modelo", kind)
    report["threshold"] = threshold
    report.loc["balanced_accuracy", ["precision", "recall", "f1-score", "support"]] = (
        balanced_accuracy_score(y_test, prediction),
        balanced_accuracy_score(y_test, prediction),
        balanced_accuracy_score(y_test, prediction),
        len(y_test),
    )
    report.loc["roc_auc", "precision"] = roc_auc_score(y_test, test_probability)
    report.loc["average_precision", "precision"] = average_precision_score(y_test, test_probability)
    report["modelo"] = kind
    from sklearn.metrics import confusion_matrix
    pd.DataFrame(
        confusion_matrix(y_test, prediction, labels=[0, 1]),
        index=["N", "nao_N"],
        columns=["N", "nao_N"],
    ).to_csv(output_dir / f"{kind}_confusion_matrix.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / f"{kind}_lstm.keras")
    pd.DataFrame({"real": y_test, "probability_nao_N": test_probability, "prediction": prediction}).to_csv(output_dir / f"{kind}_predictions.csv", index=False)
    pd.DataFrame([["threshold_f2_validation", threshold]], columns=["metric", "value"]).to_csv(output_dir / f"{kind}_calibration.csv", index=False)
    return report, threshold


def main() -> None:
    import tensorflow as tf

    paths = ProjectPaths()
    config = ExperimentConfig()
    output_dir = paths.reports / "symbolic_lstm_scale5_binary_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, labels, groups = montar_dataset(Path("data/filtered"))
    y = (labels != "N").astype("int32")
    indices_train, indices_validation, indices_test = split_by_record(X, y, groups, config.random_seed)
    split_metadata = {
        "train": sorted(np.unique(groups[indices_train]).tolist()),
        "validation": sorted(np.unique(groups[indices_validation]).tolist()),
        "test": sorted(np.unique(groups[indices_test]).tolist()),
        "scale": SCALE,
        "window_size": WINDOW_SIZE,
        "coarse_size": COARSE_SIZE,
    }
    (output_dir / "record_split.json").write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")

    X_coarse = coarse_grain(X)
    X_train, X_validation, X_test = (X_coarse[index] for index in (indices_train, indices_validation, indices_test))
    y_train, y_validation, y_test = (y[index] for index in (indices_train, indices_validation, indices_test))

    n_symbols, limits, history = escolher_particao_por_entropia(X_train)
    history.to_csv(output_dir / "entropy_partition_history.csv", index=False)
    pd.DataFrame({"symbol": np.arange(n_symbols), "lower_limit": limits[:-1], "upper_limit": limits[1:]}).to_csv(output_dir / "symbol_partition_limits.csv", index=False)
    X_train_symbolic = simbolizar_janelas(X_train, limits)
    X_validation_symbolic = simbolizar_janelas(X_validation, limits)
    X_test_symbolic = simbolizar_janelas(X_test, limits)

    continuous_report, continuous_threshold = train_one(
        "continuous", X_train[..., None], y_train, X_validation[..., None], y_validation, X_test[..., None], y_test, output_dir, config.random_seed, n_symbols
    )
    symbolic_report, symbolic_threshold = train_one(
        "symbolic", X_train_symbolic, y_train, X_validation_symbolic, y_validation, X_test_symbolic, y_test, output_dir, config.random_seed, n_symbols
    )
    comparison = pd.concat([continuous_report, symbolic_report])
    comparison.to_csv(output_dir / "comparison_scores.csv")
    pd.DataFrame({"model": ["continuous", "symbolic"], "threshold": [continuous_threshold, symbolic_threshold], "n_symbols": [n_symbols, n_symbols]}).to_csv(output_dir / "comparison_calibration.csv", index=False)
    print(f"Resultados salvos em {output_dir}")
    print(comparison.loc[["N", "nao_N", "balanced_accuracy", "roc_auc", "average_precision"], ["modelo", "precision", "recall", "f1-score", "threshold"]])


if __name__ == "__main__":
    main()
