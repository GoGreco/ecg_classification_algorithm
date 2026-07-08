"""Compara, para cada paciente já processado, o sinal ECG original (bruto)
com o sinal filtrado, um gráfico em cima do outro, destacando no sinal
filtrado a anotação de cada batimento acima do pico R.

Entradas:
    - Sinal original: `data/interim/signal_tables/{paciente}_record.csv`
      (coluna MLII), lido via `ProjectPaths().data_interim`.
    - Sinal filtrado + anotações: `./data/filtered/{paciente}_filtered.csv`
      (gerado por `exportar_sinais_filtrados.py`), com as colunas
      `sample_index`, `time_s`, `amplitude_filtered`, `annotation`.

Saída:
    `./data/images/filtered_raw/{paciente}_filtered_raw.png`
    Figura com 2 subplots empilhados (compartilhando o eixo do tempo):
        - topo: sinal original (bruto)
        - baixo: sinal filtrado, com o símbolo da anotação exibido acima
          de cada pico R anotado

Só são processados os pacientes que possuem um arquivo `*_filtered.csv`
em `./data/filtered` (ou seja, os que tinham o canal MLII e já passaram
pelo pipeline de filtragem).
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend não-interativo: evita depender de GTK/Qt/etc,
# já que este script apenas salva imagens em disco e nunca chama plt.show()

import matplotlib.pyplot as plt
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()

N_BATIMENTOS_JANELA = 5
MARGEM_ESQUERDA_AMOSTRAS = 100  # amostras extras antes do 1º batimento da janela
MARGEM_DIREITA_AMOSTRAS = 150  # amostras extras depois do último batimento da janela


def listar_pacientes_filtrados(filtered_dir: Path) -> list[str]:
    """Retorna os IDs de paciente que possuem `{id}_filtered.csv` em `filtered_dir`."""
    if not filtered_dir.exists():
        raise FileNotFoundError(
            f"Pasta de sinais filtrados não encontrada: {filtered_dir}"
        )

    ids = []
    for path in sorted(filtered_dir.glob("*_filtered.csv")):
        paciente_id = path.stem.removesuffix("_filtered")
        ids.append(paciente_id)
    return ids


def carregar_sinal_original(data_interim: Path, paciente_id: str, default_lead: str) -> pd.DataFrame | None:
    record_csv = data_interim / f"{paciente_id}_record.csv"
    if not record_csv.exists():
        return None

    frame = pd.read_csv(record_csv)
    if default_lead not in frame.columns:
        return None

    return frame


def selecionar_janela_de_batimentos(
    tabela_filtrada: pd.DataFrame,
    n_batimentos: int,
    margem_esquerda: int,
    margem_direita: int,
) -> tuple[int, int] | None:
    """Escolhe aleatoriamente `n_batimentos` anotados consecutivos e retorna
    o intervalo de amostras [inicio, fim) que cobre esses batimentos, com uma
    margem antes do primeiro e depois do último (pra mostrar a onda completa).

    Retorna None se o paciente tiver menos de `n_batimentos` anotados.
    """
    indices_anotados = tabela_filtrada.index[
        tabela_filtrada["annotation"].notna() & (tabela_filtrada["annotation"] != "")
    ].to_numpy()

    if len(indices_anotados) < n_batimentos:
        return None

    inicio_possiveis = len(indices_anotados) - n_batimentos
    inicio = random.randint(0, inicio_possiveis)  # posição aleatória do 1º batimento
    batimentos_selecionados = indices_anotados[inicio : inicio + n_batimentos]

    primeiro_sample = int(tabela_filtrada.loc[batimentos_selecionados[0], "sample_index"])
    ultimo_sample = int(tabela_filtrada.loc[batimentos_selecionados[-1], "sample_index"])

    inicio_janela = max(0, primeiro_sample - margem_esquerda)
    fim_janela = min(len(tabela_filtrada), ultimo_sample + margem_direita)

    return inicio_janela, fim_janela


def montar_figura_comparativa(
    paciente_id: str,
    sinal_original,
    tempo_original,
    tabela_filtrada: pd.DataFrame,
) -> plt.Figure:
    fig, (ax_original, ax_filtrado) = plt.subplots(
        2, 1, figsize=(20, 10), sharex=True
    )

    # --- Sinal original (bruto) ---
    ax_original.plot(tempo_original, sinal_original, color="black", alpha=0.7, label="Sinal Original")
    ax_original.set_title(f"Paciente {paciente_id} - Sinal Original")
    ax_original.set_ylabel("Amplitude")
    ax_original.grid(True, alpha=0.3)
    ax_original.legend(loc="upper right")

    # --- Sinal filtrado + anotações ---
    tempo_filtrado = tabela_filtrada["time_s"].to_numpy()
    amplitude_filtrada = tabela_filtrada["amplitude_filtered"].to_numpy()

    ax_filtrado.plot(tempo_filtrado, amplitude_filtrada, color="darkblue", alpha=0.8, label="Sinal Filtrado")
    ax_filtrado.set_title(f"Paciente {paciente_id} - Sinal Filtrado (com anotações dos batimentos)")
    ax_filtrado.set_xlabel("Tempo (s)")
    ax_filtrado.set_ylabel("Amplitude")
    ax_filtrado.grid(True, alpha=0.3)

    # Filtro vetorizado (mais rápido que .astype(str).str.strip() em colunas
    # com centenas de milhares de linhas).
    mascara_anotada = tabela_filtrada["annotation"].notna() & (tabela_filtrada["annotation"] != "")
    anotados = tabela_filtrada.loc[mascara_anotada, ["time_s", "amplitude_filtered", "annotation"]]

    xs = anotados["time_s"].to_numpy()
    ys = anotados["amplitude_filtered"].to_numpy()
    labels = anotados["annotation"].to_numpy()

    # Deslocamento acima do pico proporcional à amplitude do sinal, calculado
    # uma única vez (em vez de usar textcoords="offset points" por anotação).
    amplitude_range = amplitude_filtrada.max() - amplitude_filtrada.min()
    offset_y = 0.05 * amplitude_range if amplitude_range > 0 else 0.05

    # Um único scatter para todos os pontos anotados (muito mais rápido que
    # desenhar um bbox circular por anotação, que era o principal gargalo de
    # performance com milhares de batimentos por paciente).
    ax_filtrado.scatter(xs, ys, color="blue", s=25, zorder=3, label="Batimento anotado")
    for x, y, label in zip(xs, ys, labels):
        ax_filtrado.text(
            x,
            y + offset_y,
            str(label),
            ha="center",
            va="bottom",
            fontsize=9,
            color="blue",
            fontweight="bold",
        )

    ax_filtrado.legend(loc="upper right")
    fig.tight_layout()
    return fig


def processar_paciente(
    paciente_id: str,
    data_interim: Path,
    filtered_dir: Path,
    output_dir: Path,
    sampling_rate: int,
    default_lead: str,
) -> bool:
    frame_original = carregar_sinal_original(data_interim, paciente_id, default_lead)
    if frame_original is None:
        print(f"[{paciente_id}] Sinal original não encontrado ou sem canal {default_lead}, pulando.")
        return False

    tabela_filtrada = pd.read_csv(filtered_dir / f"{paciente_id}_filtered.csv")

    sinal_original = frame_original[default_lead].to_numpy()
    tempo_original = pd.RangeIndex(len(sinal_original)).to_numpy() / sampling_rate

    janela = selecionar_janela_de_batimentos(
        tabela_filtrada,
        n_batimentos=N_BATIMENTOS_JANELA,
        margem_esquerda=MARGEM_ESQUERDA_AMOSTRAS,
        margem_direita=MARGEM_DIREITA_AMOSTRAS,
    )
    if janela is None:
        print(
            f"[{paciente_id}] Menos de {N_BATIMENTOS_JANELA} batimentos anotados, pulando."
        )
        return False

    inicio_janela, fim_janela = janela
    fim_janela = min(fim_janela, len(sinal_original))  # segurança caso os tamanhos difiram

    tabela_filtrada = tabela_filtrada.iloc[inicio_janela:fim_janela].reset_index(drop=True)
    sinal_original = sinal_original[inicio_janela:fim_janela]
    tempo_original = tempo_original[inicio_janela:fim_janela]

    fig = montar_figura_comparativa(
        paciente_id=paciente_id,
        sinal_original=sinal_original,
        tempo_original=tempo_original,
        tabela_filtrada=tabela_filtrada,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{paciente_id}_filtered_raw.png"
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    print(f"[{paciente_id}] Imagem salva em {output_path}")
    return True


def main() -> None:
    from ecg_classification.config import ExperimentConfig, ProjectPaths

    paths = ProjectPaths()
    config = ExperimentConfig()

    filtered_dir = Path("./data/filtered")
    output_dir = Path("./data/images/filtered_raw")

    pacientes = listar_pacientes_filtrados(filtered_dir)
    print(f"{len(pacientes)} paciente(s) filtrado(s) encontrados em {filtered_dir}.")

    processados = 0
    pulados = 0
    for paciente_id in pacientes:
        sucesso = processar_paciente(
            paciente_id=paciente_id,
            data_interim=paths.data_interim,
            filtered_dir=filtered_dir,
            output_dir=output_dir,
            sampling_rate=config.sampling_rate,
            default_lead=config.default_lead,
        )
        if sucesso:
            processados += 1
        else:
            pulados += 1

    print(f"\nConcluído: {processados} imagem(ns) gerada(s), {pulados} pulado(s).")


if __name__ == "__main__":
    main()