"""LSTM (TensorFlow/Keras) 
   - Classificar o tipo de cada batimento cardíaco a partir dos sinais filtrados;
   - Salva uma tabela com o score de acurácia;

Pipeline:
    1. Lê os sinais em:  `./data/filtered/{paciente}_filtered.csv` (sinal filtrado + anotação de cada batimento, uma linha por amostra);
    2. Mantém somente anotações de batimentos, mapeadas para as cinco classes AAMI;
    3. Extrai uma janela fixa em torno do batimento (100 amostras antes e 150 depois do pico);
    4. Divide os registros, e não os batimentos, em treino, validação e teste;
    5. Treina uma LSTM multiclasse;
    6. Avalia no conjunto de teste e salva métricas por classe e métricas robustas ao
       desbalanceamento em `<reports>/lstm_beat_classification_scores.csv`.

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

    from ecg_classification.features.labels import is_beat_label, map_to_aami

    janelas: list[np.ndarray] = []
    rotulos: list[str] = []
    for idx, rotulo in zip(indices_anotados, rotulos_anotados):
        rotulo_original = str(rotulo)
        if not is_beat_label(rotulo_original):
            continue
        rotulo_aami = map_to_aami(rotulo_original)
        if rotulo_aami is None:
            continue
        inicio = idx - margem_esquerda
        fim = idx + margem_direita
        if inicio < 0 or fim > len(amplitude):
            continue
        janelas.append(amplitude[inicio:fim])
        rotulos.append(rotulo_aami)

    return janelas, rotulos


def montar_dataset(filtered_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    pacientes = listar_pacientes_filtrados(filtered_dir)
    print(f"{len(pacientes)} paciente(s) filtrado(s) encontrados em {filtered_dir}.")

    todas_janelas: list[np.ndarray] = []
    todos_rotulos: list[str] = []
    todos_registros: list[str] = []
    for paciente_id in pacientes:
        tabela_filtrada = pd.read_csv(filtered_dir / f"{paciente_id}_filtered.csv")
        janelas, rotulos = extrair_batimentos(
            tabela_filtrada,
            margem_esquerda=MARGEM_ESQUERDA_AMOSTRAS,
            margem_direita=MARGEM_DIREITA_AMOSTRAS,
        )
        todas_janelas.extend(janelas)
        todos_rotulos.extend(rotulos)
        todos_registros.extend([paciente_id] * len(janelas))
        print(f"[{paciente_id}] {len(janelas)} batimento(s) extraído(s).")

    X = np.stack(todas_janelas).astype("float32")
    y = np.array(todos_rotulos)
    grupos = np.array(todos_registros)
    print(f"\nTotal de batimentos no dataset: {len(y)}")
    print(f"Classes AAMI: {dict(pd.Series(y).value_counts().sort_index())}")
    return X, y, grupos


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


def dividir_por_registro(
    X: np.ndarray,
    y: np.ndarray,
    grupos: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide por registro usando StratifiedGroupKFold.

    Sete folds produzem aproximadamente 71/14/14% para treino/validação/teste,
    sem permitir que batimentos do mesmo registro atravessem as partições.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    # Uma única chamada a ``next`` pode colocar todos os registros de uma
    # classe rara em validação/treino. Procuramos algumas divisões candidatas
    # e escolhemos a mais próxima da distribuição global, sem quebrar grupos.
    # Isso não transforma o split em aleatório por batimento (o que causaria
    # vazamento entre batimentos do mesmo registro).
    _, y_numerico = np.unique(y, return_inverse=True)
    distribuicao_global = np.bincount(y_numerico, minlength=len(np.unique(y)))
    distribuicao_global = distribuicao_global / distribuicao_global.sum()
    candidatos = []

    for tentativa in range(32):
        seed = random_seed + tentativa
        outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
        indices_treino_validacao, indices_teste = next(outer.split(X, y, grupos))

        inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + 1)
        relativos_treino, relativos_validacao = next(
            inner.split(
                X[indices_treino_validacao],
                y[indices_treino_validacao],
                grupos[indices_treino_validacao],
            )
        )
        indices_treino = indices_treino_validacao[relativos_treino]
        indices_validacao = indices_treino_validacao[relativos_validacao]

        contagens = []
        for indices in (indices_treino, indices_validacao, indices_teste):
            contagem = np.bincount(
                y_numerico[indices], minlength=len(distribuicao_global)
            )
            contagens.append(contagem)

        # O treino e o teste precisam conter todas as classes para que o
        # treinamento e o relatório final sejam comparáveis entre execuções.
        if not np.all(contagens[0] > 0) or not np.all(contagens[2] > 0):
            continue

        erro_distribuicao = sum(
            np.abs((contagem / contagem.sum()) - distribuicao_global).sum()
            for contagem in contagens
        )
        candidatos.append((erro_distribuicao, indices_treino, indices_validacao, indices_teste))

    if not candidatos:
        raise ValueError(
            "Não foi possível encontrar uma divisão por registro com todas as "
            "classes no treino e no teste. Verifique a distribuição por registro."
        )

    _, indices_treino, indices_validacao, indices_teste = min(
        candidatos, key=lambda candidato: candidato[0]
    )

    for nome, indices in (
        ("treino", indices_treino),
        ("validação", indices_validacao),
        ("teste", indices_teste),
    ):
        print(
            f"{nome.capitalize()}: {len(indices)} batimentos, "
            f"{len(np.unique(grupos[indices]))} registro(s): "
            f"{sorted(np.unique(grupos[indices]).tolist())}"
        )

    return (
        X[indices_treino], y[indices_treino],
        X[indices_validacao], y[indices_validacao],
        X[indices_teste], y[indices_teste],
    )


def treinar_e_avaliar(
    X: np.ndarray, y: np.ndarray, grupos: np.ndarray, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight

    codificador = LabelEncoder()
    y_codificado = codificador.fit_transform(y)

    X_normalizado = normalizar_janelas(X)
    (
        X_treino, y_treino,
        X_validacao, y_validacao,
        X_teste, y_teste,
    ) = dividir_por_registro(X_normalizado, y_codificado, grupos, random_seed)

    X_treino = X_treino[..., np.newaxis]
    X_validacao = X_validacao[..., np.newaxis]
    X_teste = X_teste[..., np.newaxis]

    pesos_classe = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_treino), y=y_treino
    )
    class_weight = dict(zip(np.unique(y_treino), pesos_classe))
    print(
        "Pesos das classes: "
        + str(
            {
                codificador.inverse_transform([classe])[0]: round(float(peso), 4)
                for classe, peso in class_weight.items()
            }
        )
    )

    tf.keras.utils.set_random_seed(random_seed)
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
        validation_data=(X_validacao, y_validacao),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    y_pred_prob = modelo.predict(X_teste, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acuracia_geral = accuracy_score(y_teste, y_pred)
    acuracia_balanceada = balanced_accuracy_score(y_teste, y_pred)
    print(f"\nAcurácia no conjunto de teste: {acuracia_geral:.4f}")
    print(f"Acurácia balanceada no conjunto de teste: {acuracia_balanceada:.4f}")

    relatorio = classification_report(
        y_teste,
        y_pred,
        target_names=codificador.classes_,
        labels=np.arange(len(codificador.classes_)),
        output_dict=True,
        zero_division=0,
    )
    tabela_scores = pd.DataFrame(relatorio).transpose()
    tabela_scores.index.name = "classe"
    tabela_scores.loc["accuracy_geral", ["precision", "recall", "f1-score", "support"]] = (
        acuracia_geral,
        acuracia_geral,
        acuracia_geral,
        len(y_teste),
    )
    tabela_scores.loc[
        "balanced_accuracy", ["precision", "recall", "f1-score", "support"]
    ] = (
        acuracia_balanceada,
        acuracia_balanceada,
        acuracia_balanceada,
        len(y_teste),
    )

    matriz_confusao = pd.DataFrame(
        confusion_matrix(
            y_teste,
            y_pred,
            labels=np.arange(len(codificador.classes_)),
        ),
        index=codificador.classes_,
        columns=codificador.classes_,
    )
    matriz_confusao.index.name = "classe_real"
    matriz_confusao.columns.name = "classe_predita"

    distribuicao = pd.DataFrame(
        {
            "classe": codificador.classes_,
            "real": np.bincount(
                y_teste, minlength=len(codificador.classes_)
            ),
            "predita": np.bincount(
                y_pred, minlength=len(codificador.classes_)
            ),
        }
    )
    distribuicao["real_percentual"] = (
        distribuicao["real"] / len(y_teste) * 100
    )
    distribuicao["predita_percentual"] = (
        distribuicao["predita"] / len(y_teste) * 100
    )

    return tabela_scores, matriz_confusao, distribuicao


def salvar_diagnosticos(
    matriz_confusao: pd.DataFrame,
    distribuicao: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Salva tabelas e gráficos para diagnosticar os erros por classe."""
    import matplotlib.pyplot as plt

    tabelas_dir = output_dir / "tables"
    figuras_dir = output_dir / "figures"
    tabelas_dir.mkdir(parents=True, exist_ok=True)
    figuras_dir.mkdir(parents=True, exist_ok=True)

    matriz_confusao.to_csv(tabelas_dir / "lstm_confusion_matrix.csv")
    distribuicao.to_csv(
        tabelas_dir / "lstm_predicted_class_distribution.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    imagem = ax.imshow(matriz_confusao.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(matriz_confusao.columns)), matriz_confusao.columns)
    ax.set_yticks(range(len(matriz_confusao.index)), matriz_confusao.index)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")
    ax.set_title("Matriz de confusão — LSTM no teste")
    fig.colorbar(imagem, ax=ax, label="Quantidade de batimentos")
    limite = matriz_confusao.to_numpy().max() / 2
    for linha in range(matriz_confusao.shape[0]):
        for coluna in range(matriz_confusao.shape[1]):
            valor = matriz_confusao.iloc[linha, coluna]
            ax.text(
                coluna,
                linha,
                str(valor),
                ha="center",
                va="center",
                color="white" if valor > limite else "black",
            )
    fig.tight_layout()
    fig.savefig(figuras_dir / "lstm_confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    indices = np.arange(len(distribuicao))
    largura = 0.38
    ax.bar(
        indices - largura / 2,
        distribuicao["real"],
        largura,
        label="Real",
    )
    ax.bar(
        indices + largura / 2,
        distribuicao["predita"],
        largura,
        label="Predita",
    )
    ax.set_xticks(indices, distribuicao["classe"])
    ax.set_xlabel("Classe AAMI")
    ax.set_ylabel("Quantidade de batimentos")
    ax.set_title("Distribuição real e predita — teste")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figuras_dir / "lstm_predicted_class_distribution.png", dpi=150)
    plt.close(fig)

    print(f"Matriz de confusão salva em {tabelas_dir / 'lstm_confusion_matrix.csv'}")
    print(
        "Distribuição real/predita salva em "
        f"{tabelas_dir / 'lstm_predicted_class_distribution.csv'}"
    )
    print(f"Gráficos salvos em {figuras_dir}")


def main() -> None:
    from ecg_classification.config import ExperimentConfig, ProjectPaths

    paths = ProjectPaths()
    config = ExperimentConfig()

    filtered_dir = Path("./data/filtered")
    output_dir = paths.reports
    output_path = output_dir / "lstm_beat_classification_scores.csv"

    X, y, grupos = montar_dataset(filtered_dir)
    tabela_scores, matriz_confusao, distribuicao = treinar_e_avaliar(
        X, y, grupos, random_seed=config.random_seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tabela_scores.to_csv(output_path)
    print(f"\nTabela de scores salva em {output_path}")
    print(tabela_scores)
    print("\nMatriz de confusão:")
    print(matriz_confusao)
    print("\nDistribuição real e predita:")
    print(distribuicao)
    salvar_diagnosticos(matriz_confusao, distribuicao, output_dir)


if __name__ == "__main__":
    main()
