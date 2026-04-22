import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

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

        col_sinal_unproc = df_unprocessed.columns[1] if len(df_unprocessed.columns) > 1 else df_unprocessed.columns[0]
        col_sinal_proc = df_processed.columns[1] if len(df_processed.columns) > 1 else df_processed.columns[0]

        ax1.plot(df_unproc_slice[col_sinal_unproc], color='tab:blue', linewidth=1.0)
        ax1.set_title(f"Sinal Não Processado ({filename})", fontsize=12)
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.plot(df_proc_slice[col_sinal_proc], color='tab:orange', linewidth=1.0)
        ax2.set_title(f"Sinal Processado ({processed_filename})", fontsize=12)
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