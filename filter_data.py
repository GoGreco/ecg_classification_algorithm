"""Filtra os sinais de ECG (canal MLII) de todos os pacientes da MIT-BIH e
exporta, para cada paciente, um CSV com o sinal filtrado e a anotação de
cada batimento.

Usa exatamente o mesmo pipeline de filtragem do
`filtro_nk2_picos_manuais_alteracoes.py`:
    ecg_cleaned  = clean_ecg(sinal_bruto, sampling_rate=config.sampling_rate)
    ecg_filtrado = smooth_signal(ecg_cleaned, window=20)

As anotações de cada batimento (coluna `Symbol` do arquivo `.atr`, já
convertido para `{id}_annotation.csv`) são alinhadas diretamente pelo índice
de amostra (`Sample`), sem precisar redetectar picos-R: o `clean_ecg`
(filtfilt) e o `smooth_signal` (convolução "same") preservam o tamanho e o
alinhamento de índice do sinal original, então o `Sample` anotado no `.atr`
continua apontando para a amostra correta no sinal filtrado.

Saída: um arquivo `./data/filtered/{paciente}_filtered.csv` por paciente,
com as colunas:
    sample_index       -> índice da amostra no sinal
    time_s             -> tempo em segundos (sample_index / sampling_rate)
    amplitude_filtered -> amplitude do sinal já filtrado
    annotation          -> símbolo da anotação do batimento (vazio se não
                            houver anotação naquela amostra)

Pacientes sem o canal MLII são pulados (com um aviso no console).
Pacientes sem os CSVs interinos correspondentes (`{id}_record.csv` /
`{id}_annotation.csv`) também são pulados.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()


def ler_lista_de_pacientes(records_path: Path) -> list[str]:
    """Lê o arquivo `RECORDS` da MIT-BIH e retorna a lista de IDs de pacientes."""
    if not records_path.exists():
        raise FileNotFoundError(
            f"Arquivo de lista de pacientes não encontrado: {records_path}"
        )

    pacientes: list[str] = []
    for linha in records_path.read_text().splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        pacientes.append(linha)
    return pacientes


def filtrar_sinal(sinal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Aplica o mesmo filtro usado em `filtro_nk2_picos_manuais_alteracoes.py`."""
    from ecg_classification.preprocessing.filtering import clean_ecg, smooth_signal

    ecg_cleaned = clean_ecg(sinal, sampling_rate=sampling_rate)
    return smooth_signal(ecg_cleaned, window=20)


def montar_tabela_exportacao(
    sinal_filtrado: np.ndarray,
    annotation_frame: pd.DataFrame,
    sampling_rate: int,
) -> pd.DataFrame:
    """Monta a tabela final: sinal filtrado + anotação de cada batimento."""
    from ecg_classification.segmentation.beat_windows import annotations_to_dict

    annotations = annotations_to_dict(annotation_frame)

    n_amostras = len(sinal_filtrado)
    coluna_anotacao = np.full(n_amostras, "", dtype=object)

    fora_do_alcance = 0
    for sample_idx, symbol in annotations.items():
        if 0 <= sample_idx < n_amostras:
            coluna_anotacao[sample_idx] = symbol
        else:
            fora_do_alcance += 1

    if fora_do_alcance:
        print(
            f"  Aviso: {fora_do_alcance} anotação(ões) fora do alcance do sinal "
            "e foram ignoradas."
        )

    return pd.DataFrame(
        {
            "sample_index": np.arange(n_amostras),
            "time_s": np.arange(n_amostras) / sampling_rate,
            "amplitude_filtered": sinal_filtrado,
            "annotation": coluna_anotacao,
        }
    )


def processar_paciente(
    paciente_id: str,
    data_interim: Path,
    output_dir: Path,
    sampling_rate: int,
    default_lead: str,
) -> bool:
    """Processa um único paciente. Retorna True se um CSV foi gerado."""
    record_csv = data_interim / f"{paciente_id}_record.csv"
    annotation_csv = data_interim / f"{paciente_id}_annotation.csv"

    if not record_csv.exists() or not annotation_csv.exists():
        print(f"[{paciente_id}] CSV interino não encontrado, pulando.")
        return False

    frame = pd.read_csv(record_csv)
    if default_lead not in frame.columns:
        print(f"[{paciente_id}] Não possui o canal {default_lead}, pulando.")
        return False

    annotation_frame = pd.read_csv(annotation_csv)

    sinal_bruto = frame[default_lead].to_numpy()
    sinal_filtrado = filtrar_sinal(sinal_bruto, sampling_rate=sampling_rate)

    tabela = montar_tabela_exportacao(
        sinal_filtrado=sinal_filtrado,
        annotation_frame=annotation_frame,
        sampling_rate=sampling_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{paciente_id}_filtered.csv"
    tabela.to_csv(output_path, index=False)
    print(f"[{paciente_id}] Salvo em {output_path} ({len(tabela)} amostras).")
    return True


def main() -> None:
    from ecg_classification.config import ExperimentConfig, ProjectPaths

    paths = ProjectPaths()
    config = ExperimentConfig()

    records_path = paths.data_raw / "RECORDS"
    output_dir = Path("./data/filtered")

    pacientes = ler_lista_de_pacientes(records_path)
    print(f"{len(pacientes)} paciente(s) encontrados em {records_path}.")

    processados = 0
    pulados = 0
    for paciente_id in pacientes:
        sucesso = processar_paciente(
            paciente_id=paciente_id,
            data_interim=paths.data_interim,
            output_dir=output_dir,
            sampling_rate=config.sampling_rate,
            default_lead=config.default_lead,
        )
        if sucesso:
            processados += 1
        else:
            pulados += 1

    print(f"\nConcluído: {processados} paciente(s) exportado(s), {pulados} pulado(s).")


if __name__ == "__main__":
    main()