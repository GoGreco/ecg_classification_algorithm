import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# =============================================================================
# 1. CONFIGURAÇÕES E DIRETÓRIOS
# =============================================================================

# Diretório onde o seu script anterior salvou os arquivos CSV
data_path = './signal_tables'

# Dicionário de referência para os símbolos mais comuns da MIT-BIH (Opcional, para legenda)
mit_bih_classes = {
    'N': 'Normal',
    'L': 'Bloqueio de Ramo Esquerdo',
    'R': 'Bloqueio de Ramo Direito',
    'V': 'Contração Ventricular Prematura',
    '/': 'Batimento Paced',
    'A': 'Contração Atrial Prematura',
    'f': 'Fusão Paced/Normal',
    'F': 'Fusão Ventricular/Normal',
    'j': 'Escape Nodal',
    'a': 'Batimento Atrial Prematuro Aberrante',
    'E': 'Escape Ventricular',
    'J': 'Batimento Prematuro Nodal',
    'e': 'Escape Atrial',
    'Q': 'Batimento Desconhecido',
    'S': 'Batimento Supraventricular Prematuro',
    '~': 'Mudança na Qualidade do Sinal',
    '+': 'Anotação de Ritmo',
    '|': 'Batimento Isolado'
}

# =============================================================================
# 2. EXTRAÇÃO E CONTAGEM DOS DADOS
# =============================================================================

def coletar_distribuicao(path):
    print(f"Buscando arquivos de anotação em: {path}...")
    archivos_annot = glob.glob(os.path.join(path, '*_annotation.csv'))
    
    if not archivos_annot:
        print("Nenhum arquivo de anotação encontrado. Verifique o caminho.")
        return None

    contador_global = Counter()

    for arquivo in archivos_annot:
        try:
            df = pd.read_csv(arquivo)
            # O script original salva a coluna com um espaço no final: 'Symbol '
            # Usamos strip() para limpar os nomes das colunas e evitar erros
            df.columns = df.columns.str.strip()
            
            if 'Symbol' in df.columns:
                # Conta a frequência dos símbolos neste paciente e soma ao contador global
                contador_global.update(df['Symbol'].tolist())
            else:
                print(f"Aviso: Coluna 'Symbol' não encontrada no arquivo {arquivo}")
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")

    return contador_global

# =============================================================================
# 3. GERAÇÃO DO GRÁFICO LOGARÍTMICO
# =============================================================================

def plotar_grafico_logaritmico(contador):
    if not contador:
        return

    # Ordena os símbolos do mais frequente para o menos frequente
    simbolos_ordenados = contador.most_common()
    
    # Separa em duas listas para o eixo X e Y
    labels = [item[0] for item in simbolos_ordenados]
    valores = [item[1] for item in simbolos_ordenados]

    # Cria a figura
    plt.figure(figsize=(14, 7))
    
    # Plota as barras
    barras = plt.bar(labels, valores, color='royalblue', edgecolor='black')

    # Configura o eixo Y para escala logarítmica
    plt.yscale('log')

    # Adiciona os valores exatos em cima de cada barra para clareza
    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2, yval * 1.1, 
                 int(yval), ha='center', va='bottom', fontsize=9, rotation=45)

    # Configurações de texto do gráfico
    plt.title('Distribuição de Patologias na Base MIT-BIH (Escala Logarítmica)', fontsize=16, fontweight='bold')
    plt.xlabel('Símbolo da Anotação (Patologia / Evento)', fontsize=12)
    plt.ylabel('Frequência (Escala Logarítmica)', fontsize=12)
    
    # Ajusta o limite do eixo Y para os números não cortarem no topo
    plt.ylim(bottom=1, top=max(valores) * 5)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Cria pasta de resultados se não existir
    results_path = './results'
    os.makedirs(results_path, exist_ok=True)
    
    # Salva e mostra o gráfico
    save_path = os.path.join(results_path, 'distribuicao_logaritmica.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nGráfico salvo com sucesso em: {save_path}")
    
    plt.show()

# =============================================================================
# 4. EXECUÇÃO
# =============================================================================

if __name__ == '__main__':
    distribuicao = coletar_distribuicao(data_path)
    
    if distribuicao:
        print("\nContagem total de símbolos:")
        for simbolo, contagem in distribuicao.most_common():
            desc = mit_bih_classes.get(simbolo, 'Outro')
            print(f"[{simbolo}] {desc}: {contagem}")
            
        plotar_grafico_logaritmico(distribuicao)