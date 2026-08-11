# Varredura de escalas temporais

O experimento testa `t=2,...,10`. Para cada `t`, a janela de 250 amostras é
normalizada, dividida em blocos de `t` amostras e reduzida pela média. Quando
250 não é divisível por `t`, as amostras finais são descartadas:

```text
usable = floor(250 / t) * t
```

Depois da média, a versão simbólica é obtida usando uma partição calculada
exclusivamente no conjunto de treino de cada repetição. A versão contínua usa
os mesmos valores médios antes da simbolização.

Para cada escala e repetição são treinados os dois modelos, usando a mesma
divisão por registro. O rótulo positivo é `não-N`.

Arquivos principais:

- `repeated_fold_metrics.csv`: métricas de cada escala, modelo e repetição;
- `repeated_summary_metrics.csv`: média e desvio-padrão;
- `record_splits.json`: registros de cada divisão;
- `partitions/`: limites e histórico de entropia por escala/repetição.

Após o treinamento, gere os gráficos com:

```bash
.venv_lstm/bin/python -m scripts.plot_scale_sweep_results
```

Os gráficos serão salvos neste diretório.
