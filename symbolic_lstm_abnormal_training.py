"""Detector simbólico de batimentos N versus não-N."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()

from symbolic_lstm_training import (
    BATCH_SIZE,
    EPOCHS,
    escolher_particao_por_entropia,
    construir_modelo,
    normalizar_janelas_minmax,
    simbolizar_janelas,
)


def escolher_limiar_f2(y_true: np.ndarray, probabilidade_nao_n: np.ndarray) -> float:
    """Escolhe o limiar que maximiza F2 na validação."""
    melhor = (float("-inf"), 0.5)
    for limiar in np.linspace(0.05, 0.95, 181):
        pred = (probabilidade_nao_n >= limiar).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precisao = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f2 = 5 * precisao * recall / (4 * precisao + recall) if precisao + recall else 0.0
        if f2 > melhor[0]:
            melhor = (f2, float(limiar))
    return melhor[1]


def main() -> None:
    import tensorflow as tf
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
    )
    from sklearn.utils.class_weight import compute_class_weight

    from ecg_classification.config import ExperimentConfig, ProjectPaths
    from lstm_training import dividir_por_registro, montar_dataset

    config = ExperimentConfig()
    output_dir = ProjectPaths().reports / "symbolic_lstm_binary"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y_aami, grupos = montar_dataset(Path("./data/filtered"))
    y_binario = (y_aami != "N").astype("int32")

    (
        X_treino,
        y_treino,
        X_validacao,
        y_validacao,
        X_teste,
        y_teste,
    ) = dividir_por_registro(X, y_binario, grupos, config.random_seed)

    X_treino = normalizar_janelas_minmax(X_treino)
    X_validacao = normalizar_janelas_minmax(X_validacao)
    X_teste = normalizar_janelas_minmax(X_teste)

    n_simbolos, limites, historico = escolher_particao_por_entropia(X_treino)
    X_treino = simbolizar_janelas(X_treino, limites)
    X_validacao = simbolizar_janelas(X_validacao, limites)
    X_teste = simbolizar_janelas(X_teste, limites)

    historico.to_csv(output_dir / "entropy_partition_history.csv", index=False)
    pd.DataFrame(
        {
            "symbol": np.arange(n_simbolos),
            "lower_limit": limites[:-1],
            "upper_limit": limites[1:],
        }
    ).to_csv(output_dir / "symbol_partition_limits.csv", index=False)

    classes = np.unique(y_treino)
    pesos = compute_class_weight(class_weight="balanced", classes=classes, y=y_treino)
    class_weight = dict(zip(classes, pesos))

    tf.keras.utils.set_random_seed(config.random_seed)
    modelo = construir_modelo(tamanho_janela=250, n_simbolos=n_simbolos, n_classes=2)
    modelo.fit(
        X_treino,
        y_treino,
        validation_data=(X_validacao, y_validacao),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
        verbose=2,
    )

    prob_val = modelo.predict(X_validacao, verbose=0)[:, 1]
    limiar = escolher_limiar_f2(y_validacao, prob_val)
    prob_teste = modelo.predict(X_teste, verbose=0)[:, 1]
    pred_teste = (prob_teste >= limiar).astype("int32")

    nomes = ["N", "nao_N"]
    scores = pd.DataFrame(
        classification_report(
            y_teste,
            pred_teste,
            labels=[0, 1],
            target_names=nomes,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    scores.to_csv(output_dir / "scores.csv")
    pd.DataFrame(
        confusion_matrix(y_teste, pred_teste, labels=[0, 1]),
        index=nomes,
        columns=nomes,
    ).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(
        {
            "metric": ["threshold_f2_validation", "n_symbols"],
            "value": [limiar, n_simbolos],
        }
    ).to_csv(output_dir / "calibration.csv", index=False)
    modelo.save(output_dir / "symbolic_lstm_binary.keras")

    print(f"\nAlfabeto: {n_simbolos} símbolos")
    print(f"Limiar escolhido na validação (F2): {limiar:.3f}")
    print("Matriz de confusão [N, não-N]:")
    print(confusion_matrix(y_teste, pred_teste, labels=[0, 1]))
    print(scores)
    print(f"Resultados salvos em {output_dir}")


if __name__ == "__main__":
    main()
