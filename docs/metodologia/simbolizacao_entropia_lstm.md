# Protocolo metodológico — simbolização, entropia e LSTM

## Objetivo

Construir um pipeline reprodutível para transformar séries temporais de ECG em
sequências simbólicas e utilizar essas sequências como entrada de um modelo LSTM
para classificação dos batimentos segundo as classes AAMI.

Este documento registra as decisões metodológicas antes da implementação do
treinamento simbólico.

## Fundamentação

A tese de Pereira (2022) descreve a dinâmica simbólica como uma redução de uma
série temporal contínua para uma sequência de símbolos. O procedimento envolve:

1. condicionamento e normalização da série;
2. escolha de uma série de referência;
3. cálculo dos limites de uma partição por distribuição de probabilidade;
4. simbolização da série;
5. cálculo da entropia de Shannon para diferentes quantidades de partições;
6. análise da variação da entropia;
7. escolha do tamanho do alfabeto;
8. construção de palavras ou janelas simbólicas.

A simbolização não deve ser confundida com uma binarização fixa. O alfabeto
pode ter dois, três, quatro ou mais símbolos, conforme o comportamento da
entropia.

## Etapa 1 — padronização da série

Antes da partição, as séries devem ser colocadas em uma escala comparável.
Devemos avaliar duas alternativas:

- normalização min–max:

  `x_norm = (x - min(x)) / (max(x) - min(x))`

- padronização por z-score:

  `x_std = (x - média(x)) / desvio-padrão(x)`

A escolha precisa ser mantida fixa em todas as etapas do experimento. Os
parâmetros usados para validação e teste não podem ser reestimados com esses
próprios dados.

Para a primeira reprodução do método da tese, a alternativa de referência será
a normalização min–max da série contínua condicionada. A comparação com z-score
será um experimento separado.

## Etapa 2 — partição por entropia

Para cada quantidade de símbolos `k`, começando em `k = 2`, devem ser calculados:

1. os limites da partição;
2. a sequência simbólica correspondente;
3. a probabilidade de cada símbolo;
4. a entropia de Shannon:

   `H(k) = - Σ p_i log2(p_i)`

5. a variação da entropia:

   `ΔH(k) = H(k) - H(k-1)`

O valor de `k` será escolhido quando `ΔH(k)` atingir o regime de saturação,
usando o limiar definido no experimento. O valor não deve ser fixado como
binário antes da análise.

É obrigatório salvar, para cada execução:

- método de padronização;
- série usada para definir a partição;
- limites de cada partição;
- valores de `H(k)`;
- valores de `ΔH(k)`;
- valor final de `k`;
- distribuição dos símbolos;
- registros usados na definição da partição.

## Controle de vazamento

A partição deve ser calculada usando somente os registros de treinamento. Os
mesmos limites devem ser aplicados à validação e ao teste.

Não devemos usar o registro `100` como referência fixa se ele estiver no teste.
A escolha da referência precisa ser feita dentro do treino ou substituída por
uma série de referência construída exclusivamente com dados de treino.

## Qual série será simbolizada?

Esta decisão precisa ser explicitada porque há duas possibilidades diferentes:

### Série completa

Cada amostra do ECG é mapeada para um símbolo. Uma janela de ECG produz uma
sequência simbólica que preserva informação morfológica das ondas P, QRS e T.

### Série reduzida de batimentos

Os valores associados aos batimentos, como amplitudes dos picos ou intervalos
RR, são mapeados para símbolos. A sequência resultante representa a dinâmica
entre batimentos, mas não a forma completa de cada batimento.

O arquivo atualmente gerado em `data/processed/symbolic_sequences.csv` pertence
à segunda categoria: ele contém um símbolo por batimento, baseado na amplitude
do pico. No registro `100`, essa sequência ficou praticamente degenerada, com
2272 ocorrências de um símbolo e apenas 1 ocorrência de outro. Portanto, ela não
é uma boa entrada para a LSTM sem uma investigação adicional.

## Relação com a LSTM

Depois da escolha da série e do alfabeto, a entrada da LSTM deverá ser uma
janela de símbolos, por exemplo:

`[s(t-10), ..., s(t), ..., s(t+10)]`

O rótulo será definido para o evento central da janela. O modelo receberá
índices simbólicos ou uma representação one-hot/embedding, e não amplitudes
contínuas do ECG.

O split continuará sendo feito por registro, impedindo que símbolos de um mesmo
paciente apareçam em treino e teste.

## Diagnóstico do pipeline atual

O treinamento executado por `lstm_training.py` ainda usa `amplitude_filtered` e
portanto não testa o método simbólico.

A implementação existente em `scripts/extract_symbolic_features.py` já contém
rotinas para normalização min–max, partição por distribuição de probabilidade,
cálculo de entropia, cálculo da variação da entropia e geração de palavras
simbólicas.

Entretanto, ela atualmente:

- usa o registro `100` como referência fixa;
- calcula a entropia sobre os valores dos picos;
- pode escolher `k = 2` devido à degeneração da série de picos;
- não gera diretamente janelas simbólicas rotuladas para o treinamento da LSTM.

## Auditoria de branches

Na auditoria realizada em 2026-08-05 foram encontradas as referências:

- `main`;
- `feat-project-reorg-2026-04-09`;
- respectivas referências remotas.

Não foi localizada outra branch contendo uma implementação anterior de
normalização, partição por entropia ou LSTM simbólica. A implementação simbólica
atual aparece na linha de desenvolvimento deste projeto, especialmente nos
arquivos `src/ecg_classification/features/symbolic_dynamics.py` e
`scripts/extract_symbolic_features.py`.

## Próxima decisão

Foi escolhida a opção 1: modelar a morfologia de cada batimento, simbolizando
as amostras da janela ECG.

Cada exemplo do treinamento será construído assim:

```text
janela contínua do batimento (250 amostras)
→ normalização/padronização
→ partição definida por entropia
→ sequência simbólica (250 símbolos)
→ rótulo AAMI do batimento
```

A janela continuará centrada na anotação do batimento, com 100 amostras antes e
150 depois do pico. A sequência temporal da LSTM será, portanto, a evolução
morfológica dentro de um único batimento, e não a sequência de batimentos do
registro.

O protocolo não usará a amplitude contínua diretamente na LSTM. A partição e o
alfabeto serão determinados antes do treinamento, exclusivamente com dados de
treino, e os mesmos limites serão aplicados à validação e ao teste.

Essa decisão deve ser mantida separada de um futuro experimento de dinâmica
entre batimentos baseado em intervalos RR ou amplitudes de picos.

## Primeiro resultado experimental

O primeiro treinamento foi executado por `symbolic_lstm_training.py` em
2026-08-05. A configuração utilizou normalização min–max por janela,
particionamento por distribuição de probabilidade e limiar `epsilon = 0,2`.

O critério de entropia escolheu sete símbolos:

```text
H(7) = 2,8074
ΔH(7) = 0,2224
ΔH(8) = 0,1926
```

Como `ΔH(8) < 0,2`, o alfabeto final foi `k = 7`. A distribuição dos símbolos
no treino ficou praticamente uniforme, conforme esperado para a partição por
distribuição de probabilidade.

Resultados no teste:

```text
acurácia                 = 0,5108
acurácia balanceada      = 0,3229
macro-F1                 = 0,2296
```

Esse resultado ainda não supera o baseline contínuo, que apresentou acurácia
balanceada de `0,3461`. A LSTM simbólica aumentou o recall da classe `V` para
`0,7635`, mas continuou com desempenho muito baixo em `F` e `S` e reduziu o
recall de `Q` para `0,1563`.

O resultado deve ser interpretado como uma validação do pipeline simbólico, não
como a configuração final. Os próximos testes devem comparar a normalização
min–max por janela com uma normalização por registro e revisar a validação das
classes raras.

## Objetivo prioritário: detectar trechos não normais

O objetivo principal do experimento será detectar batimentos ou trechos que não
pertençam à classe `N`. A classificação detalhada entre `S`, `V`, `F` e `Q` será
uma segunda tarefa.

Essa decisão evita que a acurácia global seja dominada pelos batimentos normais.
O relatório principal deverá incluir:

- recall/sensibilidade de `não-N`;
- especificidade para `N`;
- precisão de `não-N`;
- F1 de `não-N`;
- balanced accuracy;
- matriz de confusão binária `N` versus `não-N`.

O rótulo binário operacional será:

```text
N       → normal
S,V,F,Q → não-N
```

Essa convenção deve ser descrita como detecção de `não-N`, e não como diagnóstico
clínico de patologia. Em particular, a classe `Q` deve ser analisada
separadamente, pois representa batimentos não classificáveis/indeterminados no
mapeamento AAMI.

## Estratégia de classificação

A primeira versão deve usar uma estratégia hierárquica:

1. detector simbólico binário `N` versus `não-N`;
2. classificador secundário entre `S`, `V`, `F` e `Q`, aplicado somente quando o
   primeiro estágio indicar `não-N`.

O limiar do primeiro estágio será ajustado usando a validação, priorizando o
recall de `não-N` e impondo uma especificidade mínima aceitável para `N`. O
conjunto de teste será usado somente uma vez para a avaliação final.

Como `F` e `S` possuem poucos exemplos em alguns registros, o resultado deverá
ser apresentado também por registro e, futuramente, em múltiplos splits por
paciente. Uma única acurácia agregada não será suficiente para concluir que os
trechos patológicos foram detectados.

## Primeiro resultado do detector `N` versus `não-N`

O detector binário foi executado por `symbolic_lstm_abnormal_training.py` em
2026-08-05. O limiar foi ajustado na validação por F2 e resultou em `0,460`.
No teste, a matriz de confusão foi:

```text
                 predito N   predito não-N
real N               10644          805
real não-N            1508          649
```

Métricas principais:

```text
sensibilidade de não-N    = 0,3009
especificidade de N       = 0,9297
F1 de não-N               = 0,3595
balanced accuracy         = 0,6153
```

O detector é conservador: reconhece bem os batimentos `N`, mas deixa passar
cerca de 70% dos `não-N`. Portanto, ainda não atende à prioridade de detectar
trechos potencialmente patológicos. O próximo ajuste deve favorecer recall de
`não-N`, com análise da curva precisão-recall e limiar definido por uma meta de
sensibilidade, em vez de usar somente o maior F2.

Esse teste utilizou uma divisão de registros diferente da LSTM multiclasse,
porque a estratificação foi recalculada para o rótulo binário. Para comparar
configurações de maneira rigorosa, os próximos experimentos devem usar uma
lista fixa de registros em treino, validação e teste.
