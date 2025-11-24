import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
import math
import os

# =============================================================================
# 1. CONFIGURAÇÃO DE CAMINHOS (MÉTODO ROBUSTO)
# =============================================================================

fs = 360

# Obtém o diretório onde ESTE script (.py) está salvo
base_path = os.path.dirname(os.path.abspath(__file__))

# Constrói os caminhos absolutos para os arquivos CSV
# Ajuste "signal_tables" se o arquivo de anotação também estiver lá dentro
record_path_abs = os.path.join(base_path, "signal_tables", "100_record.csv")
annotation_path_abs = os.path.join(base_path, "signal_tables", "100_annotation.csv") # Assumindo que está na mesma pasta do script

print(f"Diretório base do script: {base_path}")
print(f"Procurando registro em: {record_path_abs}")
print(f"Procurando anotações em: {annotation_path_abs}")

# =============================================================================
# 2. CARREGAMENTO DO SINAL
# =============================================================================

try:
    if os.path.exists(record_path_abs):
        df = pd.read_csv(record_path_abs)
        # Verifica se a coluna V5 existe, senão tenta a segunda coluna
        if 'V5' in df.columns:
            ecg_signal = df['V5'].values
        elif len(df.columns) > 1:
            print(f"Coluna 'V5' não encontrada. Usando a coluna '{df.columns[1]}'.")
            ecg_signal = df.iloc[:, 1].values
        else:
            ecg_signal = df.iloc[:, 0].values
    else:
        raise FileNotFoundError(f"Arquivo não encontrado no caminho: {record_path_abs}")

except Exception as e:
    print(f"ERRO CRÍTICO AO CARREGAR REGISTRO: {e}")
    print("Usando sinal SIMULADO para não parar a execução.")
    ecg_signal = nk.ecg_simulate(duration=10, sampling_rate=fs, heart_rate=75, noise=0.05)

# Limpeza e Suavização
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="neurokit")
s_cleaned = pd.Series(ecg_cleaned)
ecg_smoothed = s_cleaned.rolling(window=20, center=True).mean().bfill().ffill().values

# =============================================================================
# 3. DELIMITAÇÃO DE BATIMENTOS
# =============================================================================

# Detectar Picos R
_, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
r_indices = rpeaks['ECG_R_Peaks']

# Delimitar Início/Fim
_, waves_peak = nk.ecg_delineate(ecg_cleaned, r_indices, sampling_rate=fs, method="dwt")
starts = [int(x) for x in waves_peak['ECG_P_Onsets'] if not math.isnan(x)]
ends = [int(x) for x in waves_peak['ECG_T_Offsets'] if not math.isnan(x)]

# =============================================================================
# 4. CARREGAMENTO DAS ANOTAÇÕES
# =============================================================================

annotation_map = {}

if os.path.exists(annotation_path_abs):
    try:
        annot_df = pd.read_csv(annotation_path_abs)
        annot_df.columns = annot_df.columns.str.strip() # Remove espaços extras
        
        if 'Sample' in annot_df.columns and 'Symbol' in annot_df.columns:
            for _, row in annot_df.iterrows():
                annotation_map[row['Sample']] = row['Symbol']
            print(f"Sucesso: {len(annotation_map)} anotações carregadas.")
        else:
            print(f"Erro: Colunas esperadas ('Sample', 'Symbol') não encontradas em {annotation_path_abs}")
    except Exception as e:
        print(f"Erro ao ler anotações: {e}")
else:
    print(f"Aviso: Arquivo de anotações não encontrado em: {annotation_path_abs}")

# Função para encontrar a anotação mais próxima
def get_label_with_tolerance(peak_idx, annotations, tolerance=15):
    for i in range(peak_idx - tolerance, peak_idx + tolerance + 1):
        if i in annotations:
            return annotations[i]
    return None

# =============================================================================
# 5. VISUALIZAÇÃO
# =============================================================================

time = np.arange(len(ecg_smoothed)) / fs
plt.figure(figsize=(20, 8))

# Plot Sinal
plt.plot(time, ecg_smoothed, label='Sinal Suavizado', color='black', alpha=0.7)

# Plot Linhas Verticais (Início/Fim)
label_added_start = False
for idx in starts:
    if idx < len(time):
        plt.axvline(x=time[idx], color='green', linestyle='--', alpha=0.5, 
                   label='Início (P-On)' if not label_added_start else "")
        label_added_start = True

label_added_end = False
for idx in ends:
    if idx < len(time):
        plt.axvline(x=time[idx], color='purple', linestyle=':', alpha=0.5, 
                   label='Fim (T-Off)' if not label_added_end else "")
        label_added_end = True

# --- Adicionar Classificação sobre o Pico R ---
detected_labels_count = 0
for r_idx in r_indices:
    if r_idx < len(time):
        label = get_label_with_tolerance(r_idx, annotation_map)
        if label:
            detected_labels_count += 1
            plt.annotate(
                f"{label}",
                xy=(time[r_idx], ecg_smoothed[r_idx]),
                xytext=(0, 25),
                textcoords='offset points',
                ha='center',
                fontsize=12,
                color='white',
                fontweight='bold',
                bbox=dict(boxstyle="circle,pad=0.3", fc="blue", ec="darkblue", alpha=0.8)
            )

# Detectar picos/vales manuais e adicionar P/V
def detectar_extremos_manualmente(sinal):
    picos, vales = [], []
    for i in range(1, len(sinal) - 1):
        if sinal[i] > sinal[i-1] and sinal[i] > sinal[i+1]: picos.append(i)
        elif sinal[i] < sinal[i-1] and sinal[i] < sinal[i+1]: vales.append(i)
    return picos, vales

picos_manuais, vales_manuais = detectar_extremos_manualmente(ecg_smoothed)

for idx in picos_manuais:
    plt.text(time[idx], ecg_smoothed[idx] + 0.02, 'P', color='red', fontsize=9, ha='center', va='bottom', fontweight='bold')
for idx in vales_manuais:
    plt.text(time[idx], ecg_smoothed[idx] - 0.02, 'V', color='blue', fontsize=9, ha='center', va='top', fontweight='bold')

plt.title(f'ECG: Delimitação e Classificação ({detected_labels_count} anotações correspondidas)')
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 