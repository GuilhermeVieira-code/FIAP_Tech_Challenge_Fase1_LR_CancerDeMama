# Tech Challenge FIAP — Pós Tech

Repositório com os projetos do Tech Challenge da Pós Tech FIAP.

---

# 🏥 Fase 1 — Diagnóstico de Câncer de Mama (Machine Learning)

Sistema de suporte ao diagnóstico de câncer de mama, classificando tumores como benignos ou malignos a partir do Breast Cancer Wisconsin Dataset.

## Dataset

**Fonte:** [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- 569 pacientes | 30 features numéricas | Target: B=Benigno / M=Maligno

## Modelo Final

✅ **Regressão Logística + RobustScaler + PCA (90% variância)**

| Métrica | Valor |
|---|---|
| Accuracy | 0.9825 |
| Recall | 0.9524 |
| F1-Score | 0.9756 |

## Estrutura — Fase 1

```
data/
└── breast_cancer_dataset.csv
models/
└── best_model.pkl
notebooks/
├── 01_eda.ipynb
└── 02_preprocessing_and_modeling.ipynb
Dockerfile
requirements.txt
```

## Como Executar — Fase 1

```bash
# Docker
docker build -t breast-cancer-diagnosis .
docker run -p 8888:8888 breast-cancer-diagnosis

# Local
pip install -r requirements.txt
jupyter notebook
```

---

# 🧬 Fase 2 — Otimização de Hiperparâmetros via Algoritmo Genético + LLM

Utiliza Algoritmo Genético para encontrar automaticamente a melhor combinação de hiperparâmetros para o pipeline de diagnóstico da Fase 1. Um LLM local (flan-t5-base) gera explicações clínicas em linguagem natural para os profissionais de saúde.

## Arquitetura

```
fase2_ga_otimizacao/
├── genetic_algorithm.py   # AG completo: cromossomo, avaliador, operadores genéticos
├── llm.py                 # LLM local (flan-t5-base): prompts, geração e avaliação
├── main.py                # Pipeline principal: dados → AG → gráficos → LLM
├── tests.py               # Testes automatizados (pytest)
├── requirements.txt       # Dependências da Fase 2
├── results/               # Outputs gerados (gráficos, JSON, textos do LLM)
└── llm_responses/         # Respostas do LLM salvas em JSON para fine-tuning (Fase 3)
```

```
                    breast_cancer_dataset.csv
                            │
                    ┌───────▼────────┐
                    │  load_data()   │
                    └───────┬────────┘
                            │
              ┌─────────────▼──────────────┐
              │   Algoritmo Genético (AG)   │
              │  ┌─────────────────────┐   │
              │  │  Cromossomo [0,1]^6 │   │
              │  │  Gene 0 → Scaler    │   │
              │  │  Gene 1 → PCA on/off│   │
              │  │  Gene 2 → PCA var   │   │
              │  │  Gene 3 → C (log)   │   │
              │  │  Gene 4 → Solver    │   │
              │  │  Gene 5 → max_iter  │   │
              │  └────────┬────────────┘   │
              │  Fitness = 0.45·Recall     │
              │          + 0.25·F1         │
              │          + 0.20·Spec.      │
              │          + 0.10·Equity     │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  3 Experimentos             │
              │  Exp1: Pop=30  Mut=0.15    │
              │  Exp2: Pop=50  Mut=0.10    │
              │  Exp3: Pop=80  Mut=0.20    │
              └─────────────┬──────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    convergence.png  metrics_comparison  results.json
                            │
              ┌─────────────▼──────────────┐
              │   LLM Local (flan-t5-base)  │
              │  • Explicação diagnóstico   │
              │  • Comparação de modelos    │
              │  • Análise dos experimentos │
              │  • Avaliação de qualidade   │
              └─────────────┬──────────────┘
                            │
                   llm_responses/*.json
                   (dataset para Fase 3)
```

## O que o AG Otimiza

| Gene | Hiperparâmetro | Espaço de Busca |
|---|---|---|
| 0 | Scaler | StandardScaler / RobustScaler / MinMaxScaler |
| 1 | Usar PCA | Sim / Não |
| 2 | PCA variance | 0.80 – 0.99 |
| 3 | C (LogReg) | 0.001 – 100 (escala logarítmica) |
| 4 | Solver | lbfgs / liblinear / saga |
| 5 | max_iter | 200 – 2000 |

## Função Fitness

```
fitness = 0.45 × Recall + 0.25 × F1 + 0.20 × Especificidade + 0.10 × Equidade
```

- **Recall (45%):** falsos negativos = câncer não detectado → risco de vida
- **F1-score (25%):** equilíbrio geral entre precisão e recall
- **Especificidade (20%):** evita alarmes falsos e biópsias desnecessárias
- **Equidade (10%):** consistência do recall entre quartis demográficos

## Resultados

| Métrica | Baseline (Fase 1) | Otimizado (AG) | Melhoria |
|---|---|---|---|
| Recall | 0.9524 | **0.9762** | +0.0238 |
| F1-score | 0.9524 | **0.9762** | +0.0238 |
| Acurácia | 0.9649 | **0.9825** | +0.0176 |

**Hiperparâmetros ótimos encontrados:** StandardScaler + PCA(0.98) + LogReg(C=0.27, lbfgs)

## Como Executar — Fase 2

```bash
# 1. Criar e ativar ambiente virtual
py -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# ou
.venv\Scripts\activate.bat    # Windows CMD

# 2. Instalar dependências
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r fase2_ga_otimizacao/requirements.txt

# 3. Pipeline completo (AG + LLM)
py -m fase2_ga_otimizacao.main

# Modo rápido para testes (3 folds, sem LLM)
py -m fase2_ga_otimizacao.main --skip-llm --cv-folds 3

# 4. Testes automatizados
py -m pytest fase2_ga_otimizacao/tests.py -v
```

## Experimentos do AG

| # | Pop | Mutação | Gerações | Objetivo |
|:---:|:---:|:---:|:---:|---|
| 1 | 30 | 0.15 | 20 | Convergência rápida — baseline de comparação |
| 2 | 50 | 0.10 | 30 | Equilíbrio exploração/explotação |
| 3 | 80 | 0.20 | 30 | Alta diversidade — vencedor |

## LLM Local — flan-t5-base

| Característica | Detalhe |
|---|---|
| Modelo | `google/flan-t5-base` (~990 MB) |
| Execução | 100% local — sem API key, sem internet após download |
| Idioma | Português brasileiro |
| Outputs | Explicação clínica, comparação de modelos, análise do AG |
| Avaliação | Score automático: completude, terminologia médica, adequação |
| Fase 3 | Respostas salvas em `llm_responses/*.json` para fine-tuning |

## Outputs Gerados

Salvos em `fase2_ga_otimizacao/results/`:

| Arquivo | Conteúdo |
|---|---|
| `convergence.png` | Curvas de convergência dos 3 experimentos |
| `metrics_comparison.png` | Baseline Fase 1 vs. modelo otimizado |
| `experiment_results.json` | Métricas e hiperparâmetros completos |
| `explicacao_diagnostico.txt` | Explicação clínica gerada pelo LLM |
| `comparacao_modelos.txt` | Análise comparativa baseline vs. otimizado |
| `analise_experimentos.txt` | Análise dos 3 experimentos do AG |

---

**Nota:** Projeto acadêmico — Pós Tech FIAP. Não utilizar para diagnóstico médico real sem validação clínica e aprovação regulatória (ANVISA/CFM).
