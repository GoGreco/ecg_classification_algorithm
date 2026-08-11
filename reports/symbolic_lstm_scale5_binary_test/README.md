# LSTM binária com escala temporal 5

Experimento controlado para comparar duas entradas com 50 posições, usando a
mesma divisão por registro e o mesmo rótulo:

```text
N       -> 0 (normal)
S,V,F,Q -> 1 (não-N)
```

## Entradas

### Contínua

Cada janela de 250 amostras é normalizada individualmente, dividida em 50
blocos de 5 amostras e reduzida pela média de cada bloco.

### Simbólica

Usa os mesmos 50 valores contínuos da entrada anterior. Os limites da partição
e o número de símbolos são calculados exclusivamente no conjunto de treino.
O experimento selecionou 7 símbolos.

## Resultado no teste

| Modelo | Balanced accuracy | Recall não-N | Especificidade N | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| Contínuo | 0,8440 | 0,6886 | 0,9995 | 0,9881 | 0,9741 |
| Simbólico | 0,8772 | 0,8503 | 0,9040 | 0,9399 | 0,8217 |

O modelo simbólico, no limiar escolhido por F2 na validação, detectou mais
batimentos não-N e apresentou maior balanced accuracy, ao custo de mais falsos
positivos para a classe não-N. O modelo contínuo apresentou maior ROC-AUC e
average precision, indicando melhor ordenação probabilística antes da escolha
do limiar.

Os resultados não constituem ainda uma conclusão definitiva: é necessário
fixar o split para todos os experimentos, avaliar múltiplas sementes e ajustar
o limiar segundo a meta clínica/metodológica de sensibilidade e especificidade.
