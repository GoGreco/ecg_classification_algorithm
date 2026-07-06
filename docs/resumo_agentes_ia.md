# Resumo dos Agentes de IA

Este documento apresenta um resumo dos agentes de IA definidos no projeto para apoiar a revisao cientifica de manuscritos e a melhoria iterativa da qualidade metodologica.

## Visao geral

O projeto utiliza dois agentes complementares:

1. um agente revisor, com foco em criticidade cientifica e padrao editorial elevado;
2. um agente pesquisador, com foco em resposta tecnica, revisao do manuscrito e fortalecimento experimental.

Os dois agentes operam em ciclo, alternando entre avaliacao critica e resposta metodologica. Esse arranjo foi desenhado para simular uma rodada de revisao por pares com maior rigor e rastreabilidade.

## Agente 1: High Impact Biomedical Signals Reviewer

- Nome interno: `high_impact_biomedical_signals_reviewer`
- Papel: revisor de periodico de alto impacto em sinais biomedicos
- Objetivo: emitir parecer extremamente criterioso sobre metodologia, resultados, validade estatistica e relevancia clinica
- Modelo configurado: `gpt-4.1`
- Temperatura: `0.1`
- Encaminhamento seguinte: `nonlinear_dynamics_signal_researcher`

### Responsabilidades principais

- avaliar a definicao do problema e a motivacao cientifica;
- identificar fragilidades no protocolo experimental;
- verificar risco de vazamento de dados;
- inspecionar a qualidade do split por paciente ou por registro;
- questionar baseline, comparadores e ablaçoes;
- analisar metricas, desbalanceamento e relevancia clinica;
- cobrar significancia estatistica, robustez e reproducibilidade;
- revisar clareza de texto, tabelas, figuras e narrativa dos resultados.

### Saidas esperadas

O agente deve produzir:

1. decisao editorial provisoria;
2. resumo executivo do risco cientifico;
3. lista priorizada de falhas criticas;
4. comentario detalhado por secao do manuscrito;
5. demandas objetivas de revisao para o pesquisador;
6. verificacoes adicionais obrigatorias antes de considerar aceitacao.

## Agente 2: Nonlinear Dynamics Signal Researcher

- Nome interno: `nonlinear_dynamics_signal_researcher`
- Papel: pesquisador senior em dinamica nao linear, caos, processamento de sinais e classificacao
- Objetivo: responder tecnicamente ao parecer, revisar o manuscrito e fortalecer metodologia, resultados e narrativa cientifica
- Modelo configurado: `gpt-4.1`
- Temperatura: `0.2`
- Encaminhamento seguinte: `high_impact_biomedical_signals_reviewer`

### Responsabilidades principais

- responder tecnicamente a cada critica recebida;
- transformar criticas em acoes concretas no manuscrito;
- propor experimentos adicionais, controles e testes estatisticos;
- revisar secoes de resultados e discussao;
- explicitar limitacoes reais do estudo;
- preservar rigor matematico e coerencia com dinamica nao linear.

### Saidas esperadas

O agente deve produzir:

1. resposta ponto a ponto ao parecer;
2. versao revisada do manuscrito;
3. lista de alteracoes executadas;
4. pendencias remanescentes e seu racional;
5. indicacao clara do que deve retornar ao revisor na proxima rodada.

## Como os agentes se complementam

O revisor atua como filtro de qualidade cientifica. Ele tenta encontrar falhas metodologicas, inconsistencias e lacunas de validacao antes que o trabalho avance.

O pesquisador atua como agente de correcao e fortalecimento tecnico. Ele converte o parecer em revisoes reais, melhorias no texto, ajustes metodologicos e propostas de novos experimentos.

Em conjunto, os agentes criam um ciclo com duas funcoes centrais:

- aumentar o rigor da avaliacao interna do manuscrito;
- melhorar a qualidade da resposta cientifica antes de uma submissao formal.

## Fluxo de interacao

O fluxo esperado no projeto e:

1. o manuscrito inicial e apresentado ao agente revisor;
2. o revisor gera um parecer critico estruturado;
3. o parecer e encaminhado ao agente pesquisador;
4. o pesquisador responde ponto a ponto e revisa o manuscrito;
5. a nova versao retorna ao revisor para nova rodada;
6. o ciclo continua ate reduzir riscos cientificos relevantes.

## Valor para o projeto

Esse sistema de agentes e util para:

- organizar revisoes tecnicas de forma repetivel;
- documentar iteracoes de melhoria do manuscrito;
- elevar o padrao de argumentacao metodologica;
- antecipar objecoes de revisores humanos;
- apoiar trabalhos em ECG, dinamica simbolica, classificacao de arritmias e validacao experimental.

## Arquivos relacionados

- `agents/high_impact_biomedical_signals_reviewer.md`
- `agents/nonlinear_dynamics_signal_researcher.md`
- `scripts/manage_agent_cycle.py`

