# Teste de codificações simbólicas temporais

Este diretório contém dois testes aplicados às mesmas janelas simbólicas da
LSTM atual: 250 amostras por batimento, normalização min–max por janela e os
limites salvos em `reports/symbolic_lstm/symbol_partition_limits.csv`.

Também contém o teste de coarse-graining com escala `τ = 5`: a janela contínua
é dividida em 50 blocos não sobrepostos, cada bloco é representado pela média
de suas 5 amostras e somente depois é convertido para um dos 7 símbolos.

Arquivo: `block5_symbol_sequences.csv`.

## 1. Agrupamento de símbolos consecutivos

É usado run-length encoding. A sequência original:

```text
2 2 2 3 3 4 4 4
```

torna-se:

```text
2:3|3:2|4:3
```

Essa codificação reduz o eixo temporal e preserva exatamente a informação
simbólica: a sequência original pode ser reconstruída usando os comprimentos.

Arquivo: `grouped_symbol_sequences.csv`.

## 2. Máximos, mínimos e direção

São detectados máximos (`M`) e mínimos (`m`) locais com:

- proeminência mínima: `0,03` na amplitude normalizada;
- distância mínima entre extremos: `5` amostras.

Entre os extremos são registrados segmentos ascendentes (`U`) ou descendentes
(`D`). A sequência típica é:

```text
I U M D m U M D m U E
```

onde `I` e `E` representam início e fim da janela.

Arquivo: `extrema_direction_sequences.csv`.

## Resultados agregados

`encoding_lengths.csv` contém os comprimentos das quatro representações por
batimento. `encoding_lengths_by_class.csv` resume esses comprimentos por classe
AAMI. As figuras em `figures/` mostram o ECG, a sequência original, a sequência
agrupada e os extremos detectados.

## Reexecução

```bash
MPLCONFIGDIR=/private/tmp/ecg-mpl-cache \
.venv_lstm/bin/python -m scripts.test_symbolic_encodings
```
