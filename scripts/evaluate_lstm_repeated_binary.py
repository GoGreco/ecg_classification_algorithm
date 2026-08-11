"""Avalia LSTM contínua e simbólica em divisões repetidas por registro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.config import ExperimentConfig, ProjectPaths
from lstm_training import montar_dataset
from symbolic_lstm_training import escolher_particao_por_entropia, simbolizar_janelas


def coarse_grain(X: np.ndarray, scale: int) -> np.ndarray:
    minimum = X.min(axis=1, keepdims=True)
    maximum = X.max(axis=1, keepdims=True)
    amplitude = maximum - minimum
    amplitude[amplitude == 0] = 1.0
    normalized = (X - minimum) / amplitude
    usable_length = (normalized.shape[1] // scale) * scale
    if usable_length == 0:
        raise ValueError("A escala é maior que o comprimento da janela.")
    truncated = normalized[:, :usable_length]
    return truncated.reshape(len(X), usable_length // scale, scale).mean(axis=2).astype("float32")


def split_by_record(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold

    global_distribution = np.bincount(y, minlength=2) / len(y)
    candidates = []
    for attempt in range(64):
        outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed + attempt)
        train_validation, test = next(outer.split(X, y, groups))
        inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + attempt + 1)
        relative_train, relative_validation = next(
            inner.split(X[train_validation], y[train_validation], groups[train_validation])
        )
        train = train_validation[relative_train]
        validation = train_validation[relative_validation]
        parts = (train, validation, test)
        counts = [np.bincount(y[index], minlength=2) for index in parts]
        if not all(np.all(count > 0) for count in counts):
            continue
        error = sum(np.abs(count / count.sum() - global_distribution).sum() for count in counts)
        candidates.append((error, parts))
    if not candidates:
        raise ValueError("Não foi possível criar uma divisão com as duas classes em todos os conjuntos.")
    return min(candidates, key=lambda candidate: candidate[0])[1]


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


def build_model(kind: str, sequence_length: int, n_symbols: int):
    import tensorflow as tf

    if kind == "continuous":
        inputs = tf.keras.Input(shape=(sequence_length, 1), dtype="float32")
        x = tf.keras.layers.LSTM(32, return_sequences=True)(inputs)
    else:
        inputs = tf.keras.Input(shape=(sequence_length,), dtype="int32")
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


def train_and_measure(kind, data, seed, epochs, batch_size, n_symbols=7):
    import tensorflow as tf
    from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
    from sklearn.utils.class_weight import compute_class_weight

    X_train, y_train, X_validation, y_validation, X_test, y_test = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(kind, X_train.shape[1], n_symbols)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=dict(zip(classes, class_weights)),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )
    validation_probability = model.predict(X_validation, verbose=0).ravel()
    threshold = choose_threshold(y_validation, validation_probability)
    probability = model.predict(X_test, verbose=0).ravel()
    prediction = (probability >= threshold).astype("int32")
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "model": kind,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity_non_N": sensitivity,
        "false_negative_rate": 1 - sensitivity,
        "specificity_N": specificity,
        "false_positive_rate": 1 - specificity,
        "precision_non_N": precision,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "roc_auc": roc_auc_score(y_test, probability),
        "average_precision": average_precision_score(y_test, probability),
        "test_size": len(y_test),
        "test_non_N": int(y_test.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-start", type=int, default=2)
    parser.add_argument("--scale-end", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.scale_start < 2 or args.scale_end < args.scale_start:
        raise ValueError("Informe um intervalo de escalas válido.")

    config = ExperimentConfig()
    output_dir = ProjectPaths().reports / "symbolic_lstm_scale_sweep_binary_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    X, labels, groups = montar_dataset(Path("data/filtered"))
    y = (labels != "N").astype("int32")
    all_rows: list[dict[str, object]] = []
    split_rows: dict[str, object] = {}

    fixed_splits = {}
    for repetition in range(args.repetitions):
        seed = config.random_seed + repetition
        train_index, validation_index, test_index = split_by_record(X, y, groups, seed)
        fixed_splits[repetition] = (train_index, validation_index, test_index)
        split_rows[str(repetition)] = {
            "seed": seed,
            "train": sorted(np.unique(groups[train_index]).tolist()),
            "validation": sorted(np.unique(groups[validation_index]).tolist()),
            "test": sorted(np.unique(groups[test_index]).tolist()),
        }

    for scale in range(args.scale_start, args.scale_end + 1):
        X_coarse = coarse_grain(X, scale)
        sequence_length = X_coarse.shape[1]
        partition_dir = output_dir / "partitions" / f"scale_{scale}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        for repetition in range(args.repetitions):
            seed = config.random_seed + repetition
            train_index, validation_index, test_index = fixed_splits[repetition]
            X_train = X_coarse[train_index]
            X_validation = X_coarse[validation_index]
            X_test = X_coarse[test_index]
            y_train, y_validation, y_test = y[train_index], y[validation_index], y[test_index]
            n_symbols, limits, history = escolher_particao_por_entropia(X_train)
            history.to_csv(partition_dir / f"entropy_partition_history_rep{repetition}.csv", index=False)
            pd.DataFrame({"symbol": np.arange(n_symbols), "lower_limit": limits[:-1], "upper_limit": limits[1:]}).to_csv(partition_dir / f"symbol_partition_limits_rep{repetition}.csv", index=False)
            symbolic_train = simbolizar_janelas(X_train, limits)
            symbolic_validation = simbolizar_janelas(X_validation, limits)
            symbolic_test = simbolizar_janelas(X_test, limits)
            datasets = {
                "continuous": (X_train[..., None], y_train, X_validation[..., None], y_validation, X_test[..., None], y_test),
                "symbolic": (symbolic_train, y_train, symbolic_validation, y_validation, symbolic_test, y_test),
            }
            for kind, data in datasets.items():
                row = train_and_measure(kind, data, seed, args.epochs, args.batch_size, n_symbols)
                row.update({"repetition": repetition, "seed": seed, "scale": scale, "sequence_length": sequence_length, "n_symbols": n_symbols})
                all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(output_dir / "repeated_fold_metrics_partial.csv", index=False)
            print(f"Escala {scale}, repetição {repetition + 1}/{args.repetitions} concluída")

    (output_dir / "record_splits.json").write_text(json.dumps(split_rows, indent=2), encoding="utf-8")
    results = pd.DataFrame(all_rows)
    results.to_csv(output_dir / "repeated_fold_metrics.csv", index=False)
    summary = results.groupby("model").agg({column: ["mean", "std"] for column in ("fp", "fn", "sensitivity_non_N", "false_negative_rate", "specificity_N", "false_positive_rate", "balanced_accuracy", "roc_auc", "average_precision")})
    summary.to_csv(output_dir / "repeated_summary_metrics.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
