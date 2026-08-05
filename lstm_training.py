"""LSTM (TensorFlow/Keras) 
   - Classificar[] o tipo de cada batimento cardíaco a partir dos sinais filtrados em `exportar_sinais_filtrados.py`;
   - Salva uma tabela com o score de acurácia;

Pipeline:
    1. Lê os sinais em:  `./data/filtered/{paciente}_filtered.csv` (sinal filtrado + anotação de cada batimento, uma linha por amostra);
    2. Para cada amostra anotada, extrai uma janela fixa em torno do batimento — mesmas margens usadas em `segmentation.beat_windows.build_beat_windows` (100 amostras antes, 150 depois do pico) — e usa o símbolo da anotação (N, V, A, ...) como rótulo daquele batimento;
    3. Junta os batimentos de TODOS os pacientes num único dataset e faz um split aleatório por batimento: 70% treino / 30% teste (estratificado por classe, pra manter a proporção de cada tipo de batimento nos dois conjuntos);
    4. Treina uma LSTM multiclasse (uma classe por símbolo de anotação);
    5. Avalia no conjunto de teste e salva uma tabela de métricas (accuracy geral + precisão/recall/f1-score por classe) em `<reports>/lstm_beat_classification_scores.csv`.

Requer `tensorflow` e `scikit-learn` instalados no ambiente
(`pip install tensorflow scikit-learn --break-system-packages`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()

MARGEM_ESQUERDA_AMOSTRAS = 100
MARGEM_DIREITA_AMOSTRAS = 150
TAMANHO_JANELA = MARGEM_ESQUERDA_AMOSTRAS + MARGEM_DIREITA_AMOSTRAS

FRACAO_TESTE = 0.30
EPOCHS = 30
BATCH_SIZE = 128


def listar_pacientes_filtrados(filtered_dir: Path) -> list[str]:
    """Retorna os IDs de paciente que possuem `{id}_filtered.csv` em `filtered_dir`."""
    if not filtered_dir.exists():
        raise FileNotFoundError(
            f"Pasta de sinais filtrados não encontrada: {filtered_dir}"
        )
    return [
        path.stem.removesuffix("_filtered")
        for path in sorted(filtered_dir.glob("*_filtered.csv"))
    ]


def extrair_batimentos(
    tabela_filtrada: pd.DataFrame,
    margem_esquerda: int,
    margem_direita: int,
) -> tuple[list[np.ndarray], list[str]]:
    amplitude = tabela_filtrada["amplitude_filtered"].to_numpy()
    mascara_anotada = tabela_filtrada["annotation"].notna() & (
        tabela_filtrada["annotation"] != ""
    )
    indices_anotados = tabela_filtrada.index[mascara_anotada].to_numpy()
    rotulos_anotados = tabela_filtrada.loc[mascara_anotada, "annotation"].to_numpy()

    janelas: list[np.ndarray] = []
    rotulos: list[str] = []
    for idx, rotulo in zip(indices_anotados, rotulos_anotados):
        inicio = idx - margem_esquerda
        fim = idx + margem_direita
        if inicio < 0 or fim > len(amplitude):
            continue
        janelas.append(amplitude[inicio:fim])
        rotulos.append(str(rotulo))

    return janelas, rotulos


def montar_dataset(filtered_dir: Path) -> tuple[np.ndarray, np.ndarray]:

    pacientes = listar_pacientes_filtrados(filtered_dir)
    print(f"{len(pacientes)} paciente(s) filtrado(s) encontrados em {filtered_dir}.")

    todas_janelas: list[np.ndarray] = []
    todos_rotulos: list[str] = []
    for paciente_id in pacientes:
        tabela_filtrada = pd.read_csv(filtered_dir / f"{paciente_id}_filtered.csv")
        janelas, rotulos = extrair_batimentos(
            tabela_filtrada,
            margem_esquerda=MARGEM_ESQUERDA_AMOSTRAS,
            margem_direita=MARGEM_DIREITA_AMOSTRAS,
        )
        todas_janelas.extend(janelas)
        todos_rotulos.extend(rotulos)
        print(f"[{paciente_id}] {len(janelas)} batimento(s) extraído(s).")

    X = np.stack(todas_janelas).astype("float32")
    y = np.array(todos_rotulos)
    print(f"\nTotal de batimentos no dataset: {len(y)}")
    return X, y


def normalizar_janelas(X: np.ndarray) -> np.ndarray:
    media = X.mean(axis=1, keepdims=True)
    desvio = X.std(axis=1, keepdims=True)
    desvio[desvio == 0] = 1.0  # evita divisão por zero em janelas constantes
    return (X - media) / desvio


def construir_modelo(tamanho_janela: int, n_classes: int):
    import tensorflow as tf

    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(tamanho_janela, 1)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
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


def treinar_e_avaliar(X: np.ndarray, y: np.ndarray, random_seed: int) -> pd.DataFrame:
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight

    codificador = LabelEncoder()
    y_codificado = codificador.fit_transform(y)

    X_normalizado = normalizar_janelas(X)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_normalizado,
        y_codificado,
        test_size=FRACAO_TESTE,
        random_state=random_seed,
        stratify=y_codificado,
    )
    print(
        f"\nSplit por batimento: {len(y_treino)} treino "
        f"({100 * (1 - FRACAO_TESTE):.0f}%) / {len(y_teste)} teste "
        f"({100 * FRACAO_TESTE:.0f}%)."
    )

    X_treino = X_treino[..., np.newaxis]
    X_teste = X_teste[..., np.newaxis]

    pesos_classe = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_treino), y=y_treino
    )
    class_weight = dict(zip(np.unique(y_treino), pesos_classe))

    tf.random.set_seed(random_seed)
    modelo = construir_modelo(TAMANHO_JANELA, n_classes=len(codificador.classes_))
    modelo.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    modelo.fit(
        X_treino,
        y_treino,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    y_pred_prob = modelo.predict(X_teste, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acuracia_geral = accuracy_score(y_teste, y_pred)
    print(f"\nAcurácia no conjunto de teste: {acuracia_geral:.4f}")

    relatorio = classification_report(
        y_teste,
        y_pred,
        target_names=codificador.classes_,
        output_dict=True,
        zero_division=0,
    )
    tabela_scores = pd.DataFrame(relatorio).transpose()
    tabela_scores.index.name = "classe"
    tabela_scores.loc["accuracy_geral", "f1-score"] = acuracia_geral

    return tabela_scores


def main() -> None:
    from ecg_classification.config import ExperimentConfig, ProjectPaths

    paths = ProjectPaths()
    config = ExperimentConfig()

    filtered_dir = Path("./data/filtered")
    output_dir = paths.reports
    output_path = output_dir / "lstm_beat_classification_scores.csv"

    X, y = montar_dataset(filtered_dir)
    tabela_scores = treinar_e_avaliar(X, y, random_seed=config.random_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    tabela_scores.to_csv(output_path)
    print(f"\nTabela de scores salva em {output_path}")
    print(tabela_scores)


if __name__ == "__main__":
    main()