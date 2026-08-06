"""Treinamento LSTM com simbolização morfológica dos batimentos.

Cada exemplo é uma janela de 250 amostras em torno de um batimento. A janela é
normalizada, convertida em uma sequência simbólica e classificada pela classe
AAMI do batimento central.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.features.symbolic_dynamics import (
    probability_partition_limits,
    symbol_entropy,
    symbolize_with_partitions,
)

MARGEM_ESQUERDA_AMOSTRAS = 100
MARGEM_DIREITA_AMOSTRAS = 150
TAMANHO_JANELA = MARGEM_ESQUERDA_AMOSTRAS + MARGEM_DIREITA_AMOSTRAS
MIN_SIMBOLOS = 2
MAX_SIMBOLOS = 16
ENTROPIA_EPSILON = 0.2
EPOCHS = 15
BATCH_SIZE = 256


def normalizar_janelas_minmax(X: np.ndarray) -> np.ndarray:
    """Coloca cada janela em [0, 1], preservando sua morfologia relativa."""
    minimo = X.min(axis=1, keepdims=True)
    maximo = X.max(axis=1, keepdims=True)
    amplitude = maximo - minimo
    amplitude[amplitude == 0] = 1.0
    return (X - minimo) / amplitude


def escolher_particao_por_entropia(
    X_treino: np.ndarray,
    min_simbolos: int = MIN_SIMBOLOS,
    max_simbolos: int = MAX_SIMBOLOS,
    epsilon: float = ENTROPIA_EPSILON,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    """Escolhe o alfabeto pela primeira saturação de ΔH.

    Os limites são calculados exclusivamente com as janelas de treinamento.
    Todas as curvas são mantidas no relatório, inclusive após a escolha, para
    permitir inspeção da decisão.
    """
    referencia = X_treino.reshape(-1)
    historico: list[dict[str, float | int]] = []
    entropia_anterior: float | None = None
    escolhido = max_simbolos
    limites_escolhidos: np.ndarray | None = None

    for numero_simbolos in range(min_simbolos, max_simbolos + 1):
        limites = probability_partition_limits(referencia, numero_simbolos)
        simbolos = symbolize_with_partitions(referencia, limites)
        entropia = symbol_entropy(simbolos)
        delta = 0.0 if entropia_anterior is None else entropia - entropia_anterior
        historico.append(
            {
                "num_symbols": numero_simbolos,
                "entropy": float(entropia),
                "entropy_delta": float(delta),
            }
        )

        if numero_simbolos == min_simbolos:
            entropia_anterior = entropia
            continue
        if delta < epsilon:
            escolhido = numero_simbolos - 1
            break
        entropia_anterior = entropia

    limites_escolhidos = probability_partition_limits(referencia, escolhido)
    historico_df = pd.DataFrame(historico)
    return escolhido, limites_escolhidos, historico_df


def simbolizar_janelas(
    X: np.ndarray,
    limites: np.ndarray,
) -> np.ndarray:
    """Converte cada amostra contínua em um índice inteiro do alfabeto."""
    return np.asarray(
        [symbolize_with_partitions(janela, limites) for janela in X],
        dtype=np.int32,
    )


def construir_modelo(tamanho_janela: int, n_simbolos: int, n_classes: int):
    import tensorflow as tf

    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(tamanho_janela,), dtype="int32"),
            # A codificação one-hot mantém a natureza simbólica e evita a
            # operação Embedding, que neste ambiente caiu para execução CPU.
            tf.keras.layers.CategoryEncoding(
                num_tokens=n_simbolos,
                output_mode="one_hot",
            ),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def main() -> None:
    import tensorflow as tf
    from sklearn.metrics import balanced_accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight

    from ecg_classification.config import ExperimentConfig, ProjectPaths
    from lstm_training import dividir_por_registro, montar_dataset

    config = ExperimentConfig()
    paths = ProjectPaths()
    output_dir = paths.reports / "symbolic_lstm"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, grupos = montar_dataset(Path("./data/filtered"))
    codificador = LabelEncoder()
    y_codificado = codificador.fit_transform(y)

    (
        X_treino,
        y_treino,
        X_validacao,
        y_validacao,
        X_teste,
        y_teste,
    ) = dividir_por_registro(X, y_codificado, grupos, config.random_seed)

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

    print(f"\nAlfabeto escolhido pela entropia: {n_simbolos} símbolo(s)")
    print(f"Histórico salvo em {output_dir / 'entropy_partition_history.csv'}")
    print(f"Limites salvos em {output_dir / 'symbol_partition_limits.csv'}")
    print("Distribuição simbólica no treino:", dict(zip(*np.unique(X_treino, return_counts=True))))

    pesos = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_treino),
        y=y_treino,
    )
    class_weight = dict(zip(np.unique(y_treino), pesos))

    tf.keras.utils.set_random_seed(config.random_seed)
    modelo = construir_modelo(TAMANHO_JANELA, n_simbolos, len(codificador.classes_))
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

    y_pred = np.argmax(modelo.predict(X_teste, verbose=0), axis=1)
    balanced = balanced_accuracy_score(y_teste, y_pred)
    print(f"\nAcurácia balanceada no teste: {balanced:.4f}")
    scores = pd.DataFrame(
        classification_report(
            y_teste,
            y_pred,
            labels=np.arange(len(codificador.classes_)),
            target_names=codificador.classes_,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    scores.to_csv(output_dir / "scores.csv")
    modelo.save(output_dir / "symbolic_lstm.keras")
    print(f"Scores salvos em {output_dir / 'scores.csv'}")


if __name__ == "__main__":
    main()
