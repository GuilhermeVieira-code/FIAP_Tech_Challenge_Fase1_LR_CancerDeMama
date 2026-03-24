# Relatório Técnico — Tech Challenge Fase 2
## Otimização de Modelos de Diagnóstico para Saúde da Mulher via Algoritmo Genético + LLM

**Instituição:** FIAP — Pós Tech
**Fase:** 2 — Algoritmos Genéticos e LLMs
**Projeto:** Projeto 1 — Otimização de Modelos de Diagnóstico
**Dataset:** Breast Cancer Wisconsin (UCI)

---

## 1. Introdução e Contexto

O objetivo deste projeto é otimizar automaticamente os hiperparâmetros do modelo de diagnóstico de câncer de mama desenvolvido na Fase 1, utilizando Algoritmo Genético (AG). Adicionalmente, integramos um LLM local (flan-t5-base) para transformar os resultados numéricos em linguagem clínica acessível aos profissionais de saúde.

O modelo da Fase 1 utilizava Regressão Logística com hiperparâmetros definidos manualmente:
- **Scaler:** RobustScaler
- **PCA:** 90% de variância
- **C:** 1.0 (regularização padrão)
- **Solver:** lbfgs

A hipótese central é que o AG pode encontrar uma combinação superior de hiperparâmetros ao explorar o espaço de busca de forma inteligente, sem a necessidade de grid search exaustivo.

---

## 2. Implementação do Algoritmo Genético

### 2.1 Representação dos Genes (Cromossomo)

Adotamos **codificação real** (real-valued encoding): cada indivíduo é um vetor de **6 genes reais em [0, 1]**.

| Gene | Hiperparâmetro | Mapeamento |
|---|---|---|
| 0 | Scaler | Categórico: StandardScaler / RobustScaler / MinMaxScaler |
| 1 | Usar PCA | Binário: ≥ 0.5 = True |
| 2 | PCA variance | Contínuo: 0.80 – 0.99 |
| 3 | C (LogReg) | Log-escala: 10^(−3 a +2) = 0.001 a 100 |
| 4 | Solver | Categórico: lbfgs / liblinear / saga |
| 5 | max_iter | Inteiro: 200 – 2000 |

**Justificativa:** A codificação real é superior à binária para hiperparâmetros contínuos como C, que varia em ordens de magnitude. O crossover e a mutação operam diretamente nos valores reais, permitindo refinamento fino do espaço de busca.

### 2.2 Função Fitness

```
fitness = 0.45 × Recall + 0.25 × F1-score + 0.20 × Especificidade + 0.10 × Equidade
```

**Justificativa dos pesos:**

- **Recall (45%):** No contexto oncológico, falsos negativos (câncer não detectado) são clinicamente mais graves que falsos positivos. Uma paciente com diagnóstico perdido pode perder a janela de tratamento curativo.
- **F1-score (25%):** Garante equilíbrio geral entre precisão e recall, evitando que o modelo maximize recall às custas de especificidade extremamente baixa.
- **Especificidade (20%):** Controla os falsos positivos — pacientes saudáveis diagnosticadas erroneamente geram ansiedade, biópsias desnecessárias e sobrecarga do sistema de saúde.
- **Equidade (10%):** Mede a consistência do recall entre grupos demográficos (quartis de mean radius). Um modelo equitativo tem desempenho consistente independentemente do perfil da paciente.

A avaliação utiliza **validação cruzada estratificada com 5 folds** para garantir que os resultados de fitness reflitam a capacidade de generalização do modelo, não memorização dos dados de treino.

### 2.3 Operadores Genéticos

**Seleção — Torneio (k=3):**
Três indivíduos são sorteados aleatoriamente; o de maior fitness é selecionado para reprodução. O torneio foi escolhido por criar pressão seletiva consistente independente da escala absoluta dos valores de fitness — essencial quando a população já está bem convergida (todos com fitness entre 0.96 e 0.98).

**Crossover — Uniforme e Aritmético (alternados):**
- *Uniforme:* cada gene é herdado de um dos pais com probabilidade 0.5. Adequado para genes independentes.
- *Aritmético:* filho = α×pai1 + (1−α)×pai2, com α ∈ [0.3, 0.7]. Ideal para genes contínuos (C, PCA variance), gerando filhos com valores intermediários.

O crossover OX (Order Crossover) foi descartado por ser projetado para permutações (problema do caixeiro viajante), onde a ordem dos genes importa. Aqui, os genes são independentes entre si.

**Mutação — Gaussiana:**
Cada gene sofre perturbação N(0, σ) com probabilidade `mutation_rate`. A distribuição gaussiana realiza pequenas perturbações ao redor do valor atual, ideal para refinamento fino. Após a perturbação, os genes são recortados em [0, 1].

**Elitismo:**
Os 2 melhores indivíduos de cada geração passam diretamente para a próxima, garantindo que o melhor modelo encontrado nunca seja perdido.

---

## 3. Experimentos Realizados

Foram realizados 3 experimentos com diferentes configurações do AG:

| Experimento | Pop | Mutação | Gerações | Objetivo |
|---|---|---|---|---|
| 1 | 30 | 0.15 | 20 | Convergência rápida (baseline de comparação) |
| 2 | 50 | 0.10 | 30 | Equilíbrio exploração/explotação |
| 3 | 80 | 0.20 | 30 | Alta diversidade — maior exploração do espaço |

### 3.1 Resultados por Experimento

| Experimento | Fitness Final | Tempo | Hiperparâmetros Ótimos |
|---|---|---|---|
| Exp 1 | 0.9707 | 27.8s | StandardScaler \| sem PCA \| C=0.0961 \| lbfgs |
| Exp 2 | 0.9714 | 65.9s | StandardScaler \| sem PCA \| C=0.1673 \| liblinear |
| **Exp 3** | **0.9730** | 157.4s | **StandardScaler \| PCA(0.98) \| C=0.2706 \| lbfgs** |

### 3.2 Análise dos Experimentos

O Experimento 3 obteve o melhor fitness por dois fatores:
1. **População maior (80):** Maior diversidade genética inicial, reduzindo o risco de convergência prematura em ótimos locais.
2. **Mutação maior (0.20):** Maior capacidade de exploração — o AG conseguiu sair de regiões subótimas onde os experimentos menores ficaram presos.

O trade-off é o tempo de execução (157s vs 27s do Exp 1), aceitável para um processo de otimização periódica offline.

---

## 4. Comparativo de Desempenho

Avaliação no conjunto de teste (20% dos dados, 114 pacientes):

| Métrica | Baseline (Fase 1) | Otimizado (AG — Exp 3) | Melhoria |
|---|---|---|---|
| **Recall** | 0.9524 | **0.9762** | **+0.0238** |
| **F1-score** | 0.9524 | **0.9762** | **+0.0238** |
| **Acurácia** | 0.9649 | **0.9825** | **+0.0176** |
| Especificidade | — | 0.9861 | — |
| Precision (Maligno) | 0.95 | 0.98 | +0.03 |
| Precision (Benigno) | 0.97 | 0.99 | +0.02 |

### 4.1 Descobertas do AG

O AG revelou que os hiperparâmetros escolhidos manualmente na Fase 1 eram subótimos:

| Hiperparâmetro | Fase 1 | AG encontrou | Interpretação |
|---|---|---|---|
| Scaler | RobustScaler | **StandardScaler** | Dataset não tem outliers extremos que justifiquem RobustScaler |
| PCA | 90% variância | **98% variância** | Manter mais componentes preserva informação diagnóstica |
| C | 1.0 | **0.27** | Regularização mais forte — modelo Fase 1 estava sub-regularizado |
| Solver | lbfgs | **lbfgs** | Correto para regularização L2 — AG confirmou |

### 4.2 Impacto Clínico

A melhoria de +0.0238 no recall significa que, em um conjunto de 42 pacientes malignas (como no conjunto de teste), o modelo otimizado detecta **1 caso a mais** que passaria despercebido pelo modelo original. Em escala hospitalar de 1.000 pacientes/mês, isso representa aproximadamente **24 mulheres diagnosticadas a mais por mês**.

---

## 5. Integração com LLM

### 5.1 Modelo Escolhido

**google/flan-t5-base** — modelo seq2seq de 250M parâmetros (~990 MB), executado 100% localmente sem necessidade de API ou internet após o primeiro download.

**Justificativa da escolha local:**
- **Privacidade:** Dados de pacientes não saem da instituição
- **Custo zero:** Sem dependência de APIs pagas
- **Disponibilidade:** Funciona offline — hospitais não podem depender de internet

### 5.2 Técnicas de Prompt Engineering

Todos os prompts foram projetados com 3 princípios:
1. **Contexto médico feminino:** especialização em oncologia mamária
2. **Sensibilidade de gênero:** linguagem empática e acolhedora
3. **Privacidade:** nenhum dado identificável é enviado ao modelo

Os prompts foram otimizados para o modelo flan-t5 (seq2seq), usando instruções curtas e diretas que maximizam a qualidade da geração. Um sistema de fallback baseado em templates estruturados garante respostas úteis mesmo quando o modelo não gera texto adequado.

### 5.3 Outputs do LLM

| Output | Público-alvo | Conteúdo |
|---|---|---|
| `explicacao_diagnostico.txt` | Médico (paciente) | Resultado, confiança, próximos passos, comunicação |
| `comparacao_modelos.txt` | Gestão hospitalar | Impacto clínico da melhoria, recomendação |
| `analise_experimentos.txt` | Time técnico | Melhor configuração, análise do AG |

### 5.4 Avaliação de Qualidade do LLM

Cada resposta é avaliada automaticamente em 4 critérios:

| Critério | Peso | Métrica |
|---|---|---|
| Completude | 30% | Mínimo de 150 palavras |
| Terminologia médica | 40% | Presença de termos clínicos relevantes |
| Adequação | 20% | Ausência de linguagem alarmista |
| Idioma (PT-BR) | 10% | Marcadores do português |

As respostas avaliadas como "Boa" (score ≥ 0.7) são salvas em `llm_responses/*.json` no formato prompt+resposta, constituindo o dataset inicial para fine-tuning na Fase 3.

---

## 6. Escalabilidade e Monitoramento

O sistema foi projetado para execução periódica offline, sem necessidade de infraestrutura em nuvem. A escalabilidade é garantida por:

- **Cache de dados:** O dataset é carregado uma única vez em memória
- **Singleton do modelo LLM:** O pipeline flan-t5 é inicializado uma vez e reutilizado
- **Parâmetros configuráveis via CLI:** `--cv-folds` permite balancear velocidade vs. precisão
- **Logs estruturados:** Todos os resultados são salvos em JSON para auditoria e reprodutibilidade

---

## 7. Desafios e Soluções

| Desafio | Solução |
|---|---|
| Espaço em disco limitado para PyTorch | Instalação CPU-only (`--index-url .../whl/cpu`) |
| flan-t5 ecoando prompts longos | Prompts mais curtos + fallback com templates estruturados |
| Avaliação de equidade sem dados demográficos | Proxy por quartis de `mean_radius` (feature 0) |
| Tempo de avaliação do AG elevado | `--cv-folds 3` para desenvolvimento; 5 folds para resultados finais |

---

## 8. Considerações Éticas

- **Privacidade:** Nenhum dado identificável de pacientes é utilizado ou transmitido
- **Equidade:** A função fitness inclui explicitamente uma métrica de equidade demográfica
- **Responsabilidade:** O sistema é explicitamente apresentado como ferramenta de apoio — a decisão final é sempre do médico
- **Transparência:** Todas as decisões de design e limitações do modelo são documentadas
- **LGPD:** Os dados sintéticos/anônimos do dataset Wisconsin não contêm informações pessoais

---

## 9. Contribuição para a Fase 3

Os arquivos em `llm_responses/*.json` contêm pares (prompt, resposta) com avaliação de qualidade, constituindo o dataset inicial para fine-tuning de um LLM especializado em saúde feminina na Fase 3. O contexto médico feminino estabelecido nos prompts desta Fase 2 será diretamente reutilizado como base para o assistente médico completo.

---

## 10. Conclusão

O Algoritmo Genético demonstrou eficácia na otimização de hiperparâmetros do modelo de diagnóstico de câncer de mama, superando o modelo original em todas as métricas relevantes. A melhoria de +2.38% no recall tem impacto clínico direto na detecção precoce. A integração com LLM local resolve o problema de comunicabilidade dos resultados, tornando o sistema utilizável por profissionais de saúde sem background técnico em ML.
