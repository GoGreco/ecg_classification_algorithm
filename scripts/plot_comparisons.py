import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

PREFERRED_LEAD = "MLII"


def select_plot_lead(frame, preferred_lead=PREFERRED_LEAD):
    if preferred_lead in frame.columns:
        return preferred_lead
    return frame.columns[0]


def select_processed_signal_column(frame):
    if "baseline_corrected" in frame.columns:
        return "baseline_corrected"
    if "cleaned" in frame.columns:
        return "cleaned"
    return frame.columns[0]


def generate_comparison_plots():
    interim_dir = os.path.join("data", "interim", "signal_tables")
    processed_dir = os.path.join("data", "processed")
    output_dir = os.path.join("data", "images")

    os.makedirs(output_dir, exist_ok=True)

    interim_files = glob.glob(os.path.join(interim_dir, "*_record.csv"))

    if not interim_files:
        print(f"Nenhum arquivo de registro encontrado em {interim_dir}.")
        return

    for interim_path in interim_files:
        filename = os.path.basename(interim_path)
        
        patient_num = filename.split('_')[0]
        
        processed_filename = f"{patient_num}_record_processed.csv"
        processed_path = os.path.join(processed_dir, processed_filename)

        if not os.path.exists(processed_path):
            print(f"Aviso: Arquivo processado não encontrado para o paciente {patient_num} ({processed_filename}). Pulando...")
            continue

        df_unprocessed = pd.read_csv(interim_path)
        df_processed = pd.read_csv(processed_path)

        # Limita a quantidade de amostras (1800 amostras = 5 segundos a 360 Hz)
        # Remova ou ajuste o slice '[:1800]' se quiser ver o sinal inteiro
        LIMITE_AMOSTRAS = 1800
        df_unproc_slice = df_unprocessed.iloc[:LIMITE_AMOSTRAS]
        df_proc_slice = df_processed.iloc[:LIMITE_AMOSTRAS]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Sinal do Paciente: {patient_num}", fontsize=16, fontweight='bold')

        col_sinal_unproc = select_plot_lead(df_unprocessed)
        col_sinal_proc = select_processed_signal_column(df_processed)

        ax1.plot(df_unproc_slice[col_sinal_unproc], color='tab:blue', linewidth=1.0)
        ax1.set_title(f"Sinal Não Processado ({filename}, derivação {col_sinal_unproc})", fontsize=12)
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.plot(df_proc_slice[col_sinal_proc], color='tab:orange', linewidth=1.0)
        ax2.set_title(f"Sinal Processado ({processed_filename}, {col_sinal_proc})", fontsize=12)
        ax2.set_ylabel("Amplitude")
        ax2.set_xlabel("Amostras (Índice)")
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        output_filepath = os.path.join(output_dir, f"comparacao_{patient_num}.png")
        plt.savefig(output_filepath, dpi=150, bbox_inches='tight')
        plt.close(fig) 

        print(f"Imagem gerada com sucesso: {output_filepath}")

if __name__ == "__main__":
    generate_comparison_plots()
