"""
=============================================================================
ECG - Filtragem e Visualização Seletiva com NeuroKit2
=============================================================================
Descrição:
    Processa sinais de ECG da base MIT-BIH diretamente dos arquivos originais.
    Para cada paciente:
      - Lê o sinal WFDB (.dat/.hea) de ./database/{paciente}
      - Lê as anotações originais de ./database/{paciente}_.atr
      - Filtra o sinal com nk.ecg_clean() (método "neurokit", 360 Hz)
      - Suaviza com média móvel para visualização.
      - Seleciona uma janela de 5 batimentos a partir de um índice fixo (ex: 50).
      - Delimita P-Onset (linha verde --) e T-Offset (linha roxa :).
      - Anota o símbolo de classificação em uma bolinha azul sobre o pico R.
      - Marca picos locais com 'P' (vermelho) e vales locais com 'V' (azul).
      - Salva a imagem em ./results/filtered_signals/{paciente}_filtered.png.
=============================================================================
"""

import os
import math
import wfdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend para salvar imagens sem abrir interface gráfica
import matplotlib.pyplot as plt
import neurokit2 as nk

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

FS            = 360     # Frequência de amostragem MIT-BIH (Hz)
BEAT_START    = 50      # Número de indexação inicial (índice fixo do batimento)
NUM_BEATS     = 5       # Quantidade de batimentos a exibir na janela
SMOOTH_WINDOW = 20      # Janela da média móvel para suavização (amostras)
ANNOT_TOL     = 15      # Tolerância de amostras para o mapeamento da anotação no pico R

# Caminhos
DATABASE_PATH = "./database"
RESULTS_PATH  = "./results/filtered_signals"

# =============================================================================
# 1. CARGA DE DADOS DIRETAMENTE DO WFDB
# =============================================================================

def load_record_names(database_path: str) -> list:
    """Lê o arquivo RECORDS e retorna a lista de pacientes."""
    records_file = os.path.join(database_path, "RECORDS")
    if not os.path.exists(records_file):
        raise FileNotFoundError(f"Arquivo RECORDS não encontrado em: {records_file}")
    
    with open(records_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names

def load_ecg_signal(patient_name: str) -> np.ndarray:
    """
    Carrega o sinal de ECG diretamente dos arquivos WFDB (.dat/.hea) em ./database/.
    Busca pelas colunas 'MLII' ou 'V5'. Caso não existam, usa a primeira coluna.
    """
    record_path = os.path.join(DATABASE_PATH, patient_name)
    try:
        # wfdb.rdsamp lê os arquivos .dat e .hea automaticamente
        signal, fields = wfdb.rdsamp(record_path)
    except Exception as e:
        raise FileNotFoundError(f"Erro ao ler os arquivos do sinal {patient_name} em {DATABASE_PATH}: {e}")

    df = pd.DataFrame(signal, columns=fields['sig_name'])
    # Remove eventuais espaços nos nomes das colunas
    df.columns = df.columns.str.strip()

    # Prioriza MLII, se não achar, busca V5, ou pega a primeira disponível
    for col in ("MLII", "V5"):
        if col in df.columns:
            return df[col].values
            
    if len(df.columns) > 1:
        # Pega a segunda coluna se MLII/V5 não estiver com nome padrão
        return df.iloc[:, 1].values
        
    return df.iloc[:, 0].values

def load_annotations_atr(patient_name: str) -> dict:
    """
    Carrega anotações do arquivo wfdb. O script busca em ./database/{paciente}_.atr
    (e usa ./database/{paciente}.atr como fallback).
    Retorna um dicionário { indice_da_amostra: 'Símbolo' }.
    """
    ann_prefix_under = os.path.join(DATABASE_PATH, f"{patient_name}_")
    ann_prefix_plain = os.path.join(DATABASE_PATH, patient_name)
    
    annotations = None
    for prefix in (ann_prefix_under, ann_prefix_plain):
        try:
            annotations = wfdb.rdann(prefix, "atr")
            break
        except FileNotFoundError:
            continue
            
    if annotations is None:
        print(f"  [AVISO] Arquivo de anotação (.atr) não encontrado para {patient_name}.")
        return {}

    # Cria um mapeamento de Amostra -> Símbolo
    return dict(zip(annotations.sample.tolist(), annotations.symbol))

# =============================================================================
# 2. PROCESSAMENTO E DELIMITAÇÃO (NeuroKit2)
# =============================================================================

def filter_and_smooth(ecg_signal: np.ndarray):
    """
    Limpa o sinal com nk.ecg_clean() e suaviza com média móvel.
    """
    # 1. Limpeza (método 'neurokit' preserva bem a morfologia)
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=FS, method="neurokit")

    # 2. Suavização para o Gráfico (estilo do exemplo)
    series = pd.Series(ecg_cleaned)
    ecg_smoothed = (
        series.rolling(window=SMOOTH_WINDOW, center=True)
              .mean()
              .bfill()
              .ffill()
              .values
    )
    return ecg_cleaned, ecg_smoothed

def delineate_beats(ecg_cleaned: np.ndarray):
    """
    Detecta picos R e delimita P-Onsets e T-Offsets.
    """
    # Detecta Picos R
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=FS)
    r_indices = rpeaks["ECG_R_Peaks"]

    # Delimita ondas (DWT)
    _, waves = nk.ecg_delineate(ecg_cleaned, r_indices, sampling_rate=FS, method="dwt")

    # Filtra NaNs dos Onsets e Offsets
    p_onsets = [int(x) for x in waves["ECG_P_Onsets"] if not math.isnan(x)]
    t_offsets = [int(x) for x in waves["ECG_T_Offsets"] if not math.isnan(x)]

    return r_indices, p_onsets, t_offsets

# =============================================================================
# 3. DETECÇÃO MANUAL DE EXTREMOS
# =============================================================================

def detectar_extremos_manualmente(sinal: np.ndarray):
    """
    Detecta picos e vales locais (pontos maiores ou menores que os vizinhos).
    """
    picos, vales = [], []
    for i in range(1, len(sinal) - 1):
        if sinal[i] > sinal[i - 1] and sinal[i] > sinal[i + 1]:
            picos.append(i)
        elif sinal[i] < sinal[i - 1] and sinal[i] < sinal[i + 1]:
            vales.append(i)
    return picos, vales

def get_closest_annotation(peak_idx: int, annot_map: dict) -> str:
    """Busca a anotação WFDB mais próxima ao pico R detectado, com tolerância."""
    for i in range(peak_idx - ANNOT_TOL, peak_idx + ANNOT_TOL + 1):
        if i in annot_map:
            return annot_map[i]
    return None

# =============================================================================
# 4. PLOTAGEM E VISUALIZAÇÃO
# =============================================================================

def plot_and_save_window(patient_name: str, ecg_smoothed: np.ndarray, r_indices: list, 
                         p_onsets: list, t_offsets: list, annot_map: dict):
    """
    Seleciona a janela específica de batimentos e gera o gráfico visual.
    """
    # Verifica se há batimentos suficientes
    max_start = max(0, len(r_indices) - NUM_BEATS)
    start_idx = min(BEAT_START, max_start)
    end_idx = start_idx + NUM_BEATS

    if len(r_indices) == 0:
        print(f"  [AVISO] Nenhum batimento detectado para {patient_name}.")
        return

    # Batimentos que compõem a janela
    r_win = r_indices[start_idx:end_idx]
    
    # Define início e fim do recorte (com margem de tempo)
    sample_start = max(0, int(r_win[0]) - int(0.30 * FS))
    sample_end = min(len(ecg_smoothed), int(r_win[-1]) + int(0.60 * FS))

    # Recorta o sinal e converte índice para o eixo de tempo (segundos)
    seg = ecg_smoothed[sample_start:sample_end]
    t_axis = np.arange(len(seg)) / FS
    offset = sample_start # Valor para subtrair dos índices globais

    # Prepara a figura
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_facecolor("#fcfcfc")
    fig.patch.set_facecolor("#fcfcfc")

    # 1. Plotar o Sinal Principal
    ax.plot(t_axis, seg, color="#2c3e50", linewidth=1.5, zorder=3, label="ECG Filtrado")

    # 2. Plotar Delimitações (P-Onset e T-Offset)
    for p in p_onsets:
        if sample_start <= p < sample_end:
            loc = p - offset
            ax.axvline(t_axis[loc], color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)

    for t in t_offsets:
        if sample_start <= t < sample_end:
            loc = t - offset
            ax.axvline(t_axis[loc], color="#8e44ad", linestyle=":", linewidth=1.5, alpha=0.8, zorder=1)
            
    # Linhas representativas para a legenda
    ax.axvline(-1, color="#27ae60", linestyle="--", label="P-Onset")
    ax.axvline(-1, color="#8e44ad", linestyle=":", label="T-Offset")

    # 3. Anotar Picos R com Classificação (Bolinha Azul)
    for r in r_win:
        loc = int(r) - offset
        if 0 <= loc < len(seg):
            symbol = get_closest_annotation(int(r), annot_map)
            # Se não houver anotação, não coloca letra
            if not symbol: continue
                
            ax.annotate(
                str(symbol),
                xy=(t_axis[loc], seg[loc]),
                xytext=(0, 25), # Distância vertical da bolinha
                textcoords="offset points",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="circle,pad=0.3", fc="#2980b9", ec="#1f618d", lw=1.5, alpha=0.9),
                arrowprops=dict(arrowstyle="-", color="#2980b9", lw=1),
                zorder=5
            )

    # 4. Anotar Picos (P) e Vales (V)
    picos_loc, vales_loc = detectar_extremos_manualmente(seg)
    amp_range = seg.max() - seg.min() if seg.max() != seg.min() else 1.0
    margin = amp_range * 0.05

    for i in picos_loc:
        ax.text(t_axis[i], seg[i] + margin, "P", color="#c0392b", fontsize=9, fontweight="bold", ha="center", va="bottom", zorder=4)

    for i in vales_loc:
        ax.text(t_axis[i], seg[i] - margin, "V", color="#3498db", fontsize=9, fontweight="bold", ha="center", va="top", zorder=4)

    # Configurações do Gráfico
    ax.set_xlim(0, t_axis[-1])
    ax.set_title(f"Paciente: {patient_name} | Sequência: Batimentos {start_idx} a {end_idx - 1}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tempo (segundos)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()

    # Salvar
    save_path = os.path.join(RESULTS_PATH, f"{patient_name}_filtered.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# =============================================================================
# 5. LOOP PRINCIPAL
# =============================================================================

def process_patient(patient_name: str):
    """Pipeline de processamento para um único paciente."""
    print(f"Processando paciente: {patient_name}...")
    
    # Carga
    ecg_signal = load_ecg_signal(patient_name)
    annot_map = load_annotations_atr(patient_name)
    
    # Processamento
    ecg_cleaned, ecg_smoothed = filter_and_smooth(ecg_signal)
    r_indices, p_onsets, t_offsets = delineate_beats(ecg_cleaned)
    
    # Visualização
    plot_and_save_window(patient_name, ecg_smoothed, r_indices, p_onsets, t_offsets, annot_map)
    print(f"  -> Concluído com sucesso. Gráfico salvo.")

def main():
    print("Iniciando Pipeline de Processamento ECG...")
    
    # Garante que o diretório de resultados existe
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    if not os.path.exists(DATABASE_PATH):
        print(f"ERRO FATAL: Diretório base '{DATABASE_PATH}' não encontrado.")
        return

    try:
        patient_names = load_record_names(DATABASE_PATH)
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        return

    sucessos = 0
    falhas = []

    for name in patient_names:
        try:
            process_patient(name)
            sucessos += 1
        except FileNotFoundError as fnf_err:
            print(f"  -> ERRO (Arquivos faltantes): {fnf_err}")
            falhas.append(name)
        except Exception as e:
            print(f"  -> ERRO (Processamento): {e}")
            falhas.append(name)

    print("-" * 50)
    print(f"Resumo Final: {sucessos} pacientes processados com sucesso, {len(falhas)} com falhas.")
    if falhas:
        print(f"Falharam: {', '.join(falhas)}")

if __name__ == "__main__":
    main()