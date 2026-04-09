# Classificação de Sinais de ECG com MIT-BIH, Dinâmica Simbólica e Modelos Temporais

## Visão geral

Este repositório organiza um pipeline de pesquisa para **classificação de sinais de ECG** com foco em duas frentes complementares:

1. **dinâmica simbólica e dinâmica não linear**, inspiradas na tese de doutorado de Tiago Leite Pereira;
2. **modelos temporais orientados por dados**, com ênfase em **LSTM** e possíveis extensões como **CNN/LSTM**.

O objetivo é manter um fluxo reprodutível que parte do sinal bruto da **MIT-BIH Arrhythmia Database**, passa por ingestão, pré-processamento, segmentação e extração de atributos, e evolui para comparação sistemática entre abordagens de classificação.

## Estado atual do repositório

O repositório já foi reorganizado para a estrutura proposta no plano metodológico. Hoje ele contém:

- dados brutos da MIT-BIH em [`data/raw/mit_bih`](data/raw/mit_bih);
- CSVs intermediários em [`data/interim/signal_tables`](data/interim/signal_tables);
- artefatos gerados em [`data/processed`](data/processed);
- pacote principal em [`src/ecg_classification`](src/ecg_classification);
- scripts de entrada em [`scripts`](scripts);
- scripts originais preservados em [`scripts/legacy`](scripts/legacy);
- testes automatizados em [`tests`](tests).

Também existe compatibilidade com os nomes antigos de script na raiz do projeto:

- [`data_load.py`](data_load.py)
- [`filter_data.py`](filter_data.py)
- [`baseline_wander.py`](baseline_wander.py)
- [`filtro_nk2_picos_manuais_alteracoes.py`](filtro_nk2_picos_manuais_alteracoes.py)

Esses arquivos agora funcionam como **wrappers finos** para o pipeline novo.

## Estrutura atual

```text
.
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── data/
│   ├── raw/
│   │   └── mit_bih/
│   ├── interim/
│   │   └── signal_tables/
│   ├── processed/
│   └── external/
├── docs/
│   ├── README.md
│   ├── projeto_pibic/
│   ├── referencias/
│   └── figuras/
├── notebooks/
│   ├── 01_exploracao_mitbih.ipynb
│   ├── 02_preprocessamento.ipynb
│   └── 03_modelagem.ipynb
├── src/
│   └── ecg_classification/
│       ├── config.py
│       ├── io/
│       ├── preprocessing/
│       ├── segmentation/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       └── utils/
├── scripts/
│   ├── make_dataset.py
│   ├── preprocess_mitbih.py
│   ├── extract_symbolic_features.py
│   ├── train_lstm.py
│   ├── evaluate_models.py
│   ├── run_experiment.py
│   └── legacy/
├── experiments/
│   ├── exp_001_baseline_filtering/
│   ├── exp_002_symbolic_dynamics/
│   └── exp_003_lstm/
├── reports/
│   ├── legacy_results/
│   ├── figures/
│   ├── tables/
│   └── manuscript/
└── tests/
```

## Mapeamento entre estrutura antiga e nova

- `database/` foi movido para `data/raw/mit_bih/`
- `signal_tables/` foi movido para `data/interim/signal_tables/`
- `results/` foi movido para `reports/legacy_results/`
- scripts exploratórios antigos foram preservados em `scripts/legacy/`

## Módulos principais

### Ingestão e exportação

- [`src/ecg_classification/io/load_wfdb.py`](src/ecg_classification/io/load_wfdb.py): leitura de sinais, anotações e metadados da MIT-BIH.
- [`src/ecg_classification/io/export_csv.py`](src/ecg_classification/io/export_csv.py): exportação de registros WFDB para CSV.

### Pré-processamento

- [`src/ecg_classification/preprocessing/filtering.py`](src/ecg_classification/preprocessing/filtering.py): limpeza e filtragem.
- [`src/ecg_classification/preprocessing/baseline.py`](src/ecg_classification/preprocessing/baseline.py): contaminação sintética e remoção de baseline wander.
- [`src/ecg_classification/preprocessing/normalization.py`](src/ecg_classification/preprocessing/normalization.py): normalização.
- [`src/ecg_classification/preprocessing/quality.py`](src/ecg_classification/preprocessing/quality.py): métricas simples de qualidade.

### Segmentação

- [`src/ecg_classification/segmentation/rpeaks.py`](src/ecg_classification/segmentation/rpeaks.py): detecção de picos R.
- [`src/ecg_classification/segmentation/delineation.py`](src/ecg_classification/segmentation/delineation.py): delineação de ondas.
- [`src/ecg_classification/segmentation/beat_windows.py`](src/ecg_classification/segmentation/beat_windows.py): janelas por batimento e alinhamento com anotações.

### Features e modelagem

- [`src/ecg_classification/features/symbolic_dynamics.py`](src/ecg_classification/features/symbolic_dynamics.py): baseline inicial de dinâmica simbólica.
- [`src/ecg_classification/features/rr_features.py`](src/ecg_classification/features/rr_features.py): atributos RR.
- [`src/ecg_classification/features/morphology.py`](src/ecg_classification/features/morphology.py): atributos morfológicos simples.
- [`src/ecg_classification/models/classical_ml.py`](src/ecg_classification/models/classical_ml.py): baseline clássico com Random Forest.
- [`src/ecg_classification/models/lstm.py`](src/ecg_classification/models/lstm.py): scaffold para baseline LSTM.
- [`src/ecg_classification/models/cnn_lstm.py`](src/ecg_classification/models/cnn_lstm.py): scaffold para CNN/LSTM.

### Avaliação

- [`src/ecg_classification/evaluation/metrics.py`](src/ecg_classification/evaluation/metrics.py): métricas de classificação e matriz de confusão.
- [`src/ecg_classification/evaluation/validation.py`](src/ecg_classification/evaluation/validation.py): split por registro.
- [`src/ecg_classification/evaluation/statistical_tests.py`](src/ecg_classification/evaluation/statistical_tests.py): teste estatístico pareado básico.

## Como preparar o ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

O arquivo [`requirements.txt`](requirements.txt) já foi convertido para UTF-8.

## Como executar

### Pipeline novo

Exportar registros WFDB para CSV:

```bash
python scripts/make_dataset.py
```

Pré-processar sinais e salvar versões limpas:

```bash
python scripts/preprocess_mitbih.py
```

Extrair atributos simbólicos:

```bash
python scripts/extract_symbolic_features.py
```

Treinar e avaliar o baseline clássico por registro:

```bash
python scripts/train_classical_ml.py
```

Esse baseline monta janelas por batimento usando os pontos anotados da MIT-BIH como referência, extrai atributos morfológicos, simbólicos e RR, e treina um `RandomForestClassifier` com split por registro.
Antes do treino, o pipeline filtra anotações que não representam batimentos e mapeia os rótulos válidos para as classes AAMI `N`, `S`, `V`, `F` e `Q`.

Executar via despachante simples:

```bash
python scripts/run_experiment.py make_dataset
python scripts/run_experiment.py preprocess
python scripts/run_experiment.py symbolic_features
python scripts/run_experiment.py train_classical_ml
```

### Agentes em Markdown para revisao de artigos

O repositório agora suporta agentes definidos em arquivos `.md` dentro de [`agents/`](agents).

Perfis iniciais:

- [`agents/high_impact_biomedical_signals_reviewer.md`](agents/high_impact_biomedical_signals_reviewer.md): revisor altamente criterioso para periodicos de alto impacto em sinais biomédicos.
- [`agents/nonlinear_dynamics_signal_researcher.md`](agents/nonlinear_dynamics_signal_researcher.md): pesquisador sênior em dinâmica não linear, caos, processamento de sinais e classificação.

Esses agentes foram organizados para operar em ciclo:

1. o revisor emite um parecer rigoroso;
2. o pesquisador responde ao parecer e revisa o manuscrito;
3. o manuscrito revisado retorna ao revisor na rodada seguinte.

Inicializar uma sessão de revisão:

```bash
python scripts/manage_agent_cycle.py init docs/manuscript.md --output-dir reports/agent_cycles/session_001
```

Após preencher `reviewer_report.md` com o parecer do revisor, avance a sessão:

```bash
python scripts/manage_agent_cycle.py progress reports/agent_cycles/session_001
```

Quando `researcher_response.md` e `revised_manuscript.md` estiverem preenchidos, avance novamente para abrir a próxima rodada de revisão:

```bash
python scripts/manage_agent_cycle.py progress reports/agent_cycles/session_001
```

### Compatibilidade com scripts antigos

Os nomes antigos continuam disponíveis na raiz e redirecionam para o pipeline novo ou para uma demonstração equivalente:

```bash
python data_load.py
python filter_data.py
python baseline_wander.py
python filtro_nk2_picos_manuais_alteracoes.py
```

## Testes e validação

O repositório possui testes automatizados para verificar a reorganização estrutural e os módulos principais.

Cobertura atual da suíte:

- resolução e criação da estrutura de diretórios com `ProjectPaths`;
- exportação de CSV com política de sobrescrita;
- normalização e extração de atributos simbólicos;
- utilidades de segmentação e alinhamento de anotações;
- métricas de avaliação e divisão por registros;
- pré-processamento de registros em `scripts/preprocess_mitbih.py`;
- construção do dataset por batimento e treino do baseline clássico;
- delegação dos scripts de entrada e wrappers de compatibilidade.

Executar a suíte:

```bash
pytest -q
```

Status validado localmente após a reorganização:

```text
24 passed
```

## Limitações atuais

O repositório está melhor estruturado, mas ainda não está completo do ponto de vista científico:

- o baseline clássico já existe, mas ainda precisa de iteração de features e tratamento de desbalanceamento;
- o baseline clássico já filtra anotações não-batimento e usa mapeamento AAMI, mas ainda precisa de iteração de features e validação mais forte;
- o baseline LSTM ainda é apenas scaffold;
- não há protocolo final de treino/validação/teste implementado ponta a ponta;
- a dinâmica simbólica está em versão inicial;
- os notebooks ainda são placeholders;
- faltam logs de experimento mais ricos e avaliação estatística mais abrangente.

## Próximos passos recomendados

1. implementar o pipeline de segmentação batimento a batimento com persistência de janelas;
2. formalizar o protocolo de treino/validação/teste por registro;
3. expandir a dinâmica simbólica para o procedimento metodológico completo;
4. implementar o baseline LSTM real sob o mesmo protocolo;
5. gerar tabelas e figuras automaticamente em `reports/`.
