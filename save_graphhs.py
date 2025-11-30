import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
import math
import os
import glob

# =============================================================================
# 1. CONFIGURAÇÕES GERAIS E DIRETÓRIOS
# =============================================================================

fs = 360
base_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(base_path, "signal_tables")

# Configura pasta de saída
results_folder = os.path.join(base_path, "results", "graphs")
os.makedirs(results_folder, exist_ok=True)

print(f"Diretório base: {base_path}")
print(f"Lendo dados de: {data_folder}")
print(f"Salvando gráficos em: {results_folder}")

# =============================================================================
# 2. FUNÇÃO AUXILIAR (Do seu código original)
# =============================================================================

def detectar_extremos_manualmente(sinal):
    picos, vales = [], []
    for i in range(1, len(sinal) - 1):
        if sinal[i] > sinal[i-1] and sinal[i] > sinal[i+1]: picos.append(i)
        elif sinal[i] < sinal[i-1] and sinal[i] < sinal[i+1]: vales.append(i)
    return picos, vales

def get_label_with_tolerance(peak_idx, annotations, tolerance=15):
    for i in range(peak_idx - tolerance, peak_idx + tolerance + 1):
        if i in annotations:
            return annotations[i]
    return None

# =============================================================================
# 3. FUNÇÃO DE PROCESSAMENTO POR PACIENTE
# =============================================================================

def processar_paciente(paciente_id, record_path_abs, annotation_path_abs):
    print(f"\n--- Processando Paciente: {paciente_id} ---")
    
    # --- A. Carregamento do Sinal ---
    try:
        if os.path.exists(record_path_abs):
            df = pd.read_csv(record_path_abs)
            # Verifica colunas conforme sua lógica original
            if 'V5' in df.columns:
                ecg_signal = df['V5'].values
            elif len(df.columns) > 1:
                # print(f"Coluna 'V5' não encontrada. Usando a coluna '{df.columns[1]}'.")
                ecg_signal = df.iloc[:, 1].values
            else:
                ecg_signal = df.iloc[:, 0].values
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {record_path_abs}")

    except Exception as e:
        print(f"ERRO CRÍTICO AO CARREGAR REGISTRO {paciente_id}: {e}")
        # Fallback para simulação apenas se falhar o carregamento real
        ecg_signal = nk.ecg_simulate(duration=10, sampling_rate=fs, heart_rate=75, noise=0.05)

    # --- B. Processamento (Limpeza e Suavização) ---
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="neurokit")
    s_cleaned = pd.Series(ecg_cleaned)
    ecg_smoothed = s_cleaned.rolling(window=20, center=True).mean().bfill().ffill().values

    # --- C. Delimitação e Picos ---
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
        r_indices = rpeaks['ECG_R_Peaks']
        
        if len(r_indices) == 0:
            print(f"Nenhum pico R detectado para {paciente_id}. Pulando.")
            return

        # Delimitar Início/Fim
        _, waves_peak = nk.ecg_delineate(ecg_cleaned, r_indices, sampling_rate=fs, method="dwt", show=False)
        starts = [int(x) for x in waves_peak['ECG_P_Onsets'] if not math.isnan(x)]
        ends = [int(x) for x in waves_peak['ECG_T_Offsets'] if not math.isnan(x)]
    except Exception as e:
        print(f"Erro no processamento de sinal do paciente {paciente_id}: {e}")
        return

    # --- D. Carregamento das Anotações ---
    annotation_map = {}
    if os.path.exists(annotation_path_abs):
        try:
            annot_df = pd.read_csv(annotation_path_abs)
            annot_df.columns = annot_df.columns.str.strip()
            
            if 'Sample' in annot_df.columns and 'Symbol' in annot_df.columns:
                for _, row in annot_df.iterrows():
                    annotation_map[row['Sample']] = row['Symbol']
                # print(f"Sucesso: {len(annotation_map)} anotações carregadas.")
            else:
                print(f"Erro: Colunas esperadas não encontradas em {annotation_path_abs}")
        except Exception as e:
            print(f"Erro ao ler anotações: {e}")
    else:
        print(f"Aviso: Anotações não encontradas para {paciente_id}")

    # --- E. Definição do Limite de Visualização (5 Batimentos) ---
    num_beats_to_show = 5
    limit_samples = len(ecg_smoothed) # Default: sinal inteiro
    
    if len(r_indices) >= num_beats_to_show:
        # Pega o 5º pico R e adiciona uma margem de 0.8s (aprox 288 samples)
        last_r_idx = r_indices[num_beats_to_show - 1]
        limit_samples = int(last_r_idx + (0.8 * fs))
        if limit_samples > len(ecg_smoothed):
            limit_samples = len(ecg_smoothed)
            
    # --- F. Visualização ---
    time = np.arange(len(ecg_smoothed)) / fs
    
    plt.figure(figsize=(20, 8))

    # Plot Sinal
    plt.plot(time, ecg_smoothed, label=f'Sinal Suavizado (Pac. {paciente_id})', color='black', alpha=0.7)

    # Plot Linhas Verticais (Início/Fim) - Filtrando para performance
    label_added_start = False
    for idx in starts:
        if idx < limit_samples:
            plt.axvline(x=time[idx], color='green', linestyle='--', alpha=0.5, 
                       label='Início (P-On)' if not label_added_start else "")
            label_added_start = True

    label_added_end = False
    for idx in ends:
        if idx < limit_samples:
            plt.axvline(x=time[idx], color='purple', linestyle=':', alpha=0.5, 
                       label='Fim (T-Off)' if not label_added_end else "")
            label_added_end = True

    # Adicionar Classificação sobre o Pico R
    detected_labels_count = 0
    for r_idx in r_indices:
        if r_idx < limit_samples:
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

    # Detectar picos/vales manuais
    picos_manuais, vales_manuais = detectar_extremos_manualmente(ecg_smoothed)

    for idx in picos_manuais:
        if idx < limit_samples:
            plt.text(time[idx], ecg_smoothed[idx] + 0.02, 'P', color='red', fontsize=9, ha='center', va='bottom', fontweight='bold')
    for idx in vales_manuais:
        if idx < limit_samples:
            plt.text(time[idx], ecg_smoothed[idx] - 0.02, 'V', color='blue', fontsize=9, ha='center', va='top', fontweight='bold')

    # Configurações do Gráfico
    plt.title(f'Paciente {paciente_id}: Classificação e Delimitação (5 Primeiros Batimentos)')
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Aplica o Zoom (Limite de X)
    plt.xlim(0, time[limit_samples-1])
    
    plt.tight_layout()
    
    # Salvar e Fechar
    filename = f"{paciente_id}_5_beats.png"
    save_path = os.path.join(results_folder, filename)
    plt.savefig(save_path)
    print(f"Gráfico salvo: {save_path}")
    plt.close()

# =============================================================================
# 4. EXECUÇÃO EM LOTE
# =============================================================================

if __name__ == "__main__":
    if os.path.exists(data_folder):
        # Procura por todos os arquivos *_record.csv
        record_files = glob.glob(os.path.join(data_folder, "*_record.csv"))
        record_files.sort()
        
        if record_files:
            print(f"Iniciando processamento de {len(record_files)} registros...")
            for rec_path in record_files:
                # Extrai ID (ex: 100_record.csv -> 100)
                filename = os.path.basename(rec_path)
                paciente_id = filename.split('_')[0]
                
                # Define caminho da anotação correspondente
                annot_path = os.path.join(data_folder, f"{paciente_id}_annotation.csv")
                
                processar_paciente(paciente_id, rec_path, annot_path)
            
            print("\nProcessamento concluído.")
        else:
            print("Nenhum arquivo '_record.csv' encontrado na pasta signal_tables.")
            # Opcional: Executar simulação para teste se não houver arquivos
            # processar_paciente("Simulado_100", "", "") 
    else:
        print(f"Pasta de dados não encontrada: {data_folder}")