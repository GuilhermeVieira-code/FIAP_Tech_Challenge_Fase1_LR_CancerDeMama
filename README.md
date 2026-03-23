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
# Acesse http://localhost:8888

# Local
pip install -r requirements.txt
jupyter notebook
```

---

# 🧬 Fase 2 — Otimização de Hiperparâmetros via Algoritmo Genético + LLM

Usa AG para encontrar automaticamente a melhor combinação de hiperparâmetros para o pipeline de diagnóstico da Fase 1. O Google Gemini gera explicações clínicas sensíveis ao gênero a partir dos resultados do modelo.

## O que o AG Otimiza

| Gene | Espaço de Busca |
|---|---|
| Scaler | StandardScaler / RobustScaler / MinMaxScaler |
| Usar PCA | Sim / Não |
| PCA variance | 0.80 – 0.99 |
| C (LogReg) | 0.001 – 100 (escala log) |
| Solver | lbfgs / liblinear / saga |
| max_iter | 200 – 2000 |

**Fitness:** `0.50 × Recall + 0.30 × F1 + 0.20 × Especificidade`

Prioriza recall pois falsos negativos (câncer não detectado) são clinicamente mais graves.

## Estrutura — Fase 2

```
fase2_ga_otimizacao/
├── src/
│   ├── chromosome.py        # Representação e decodificação dos genes
│   ├── genetic_algorithm.py # AG: torneio, crossover uniforme/aritmético, mutação gaussiana
│   └── model_evaluator.py   # Pipeline sklearn + validação cruzada estratificada
├── llm/
│   ├── prompts.py           # Templates Gemini (contexto médico feminino)
│   ├── report_generator.py  # Explicação clínica + comparação + análise AG
│   └── llm_responses/       # Respostas salvas para fine-tuning na Fase 3
├── tests/
│   ├── test_ga.py
│   └── test_fitness.py
├── results/
├── main.py
└── requirements.txt
```

## Como Executar — Fase 2

```bash
# 1. Instalar dependências
pip install -r fase2_ga_otimizacao/requirements.txt

# 2. Configurar API Gemini (opcional)
cp .env.example .env
# Edite .env com sua GEMINI_API_KEY

# 3. Pipeline completo
python -m fase2_ga_otimizacao.main

# Sem API Gemini (mais rápido)
python -m fase2_ga_otimizacao.main --skip-llm

# CV rápido para testes locais
python -m fase2_ga_otimizacao.main --skip-llm --cv-folds 3

# 4. Testes
pytest fase2_ga_otimizacao/tests/ -v
```

## Experimentos do AG

| # | Pop | Mutação | Gerações | Objetivo |
|:---:|:---:|:---:|:---:|---|
| 1 | 30 | 0.15 | 20 | Convergência rápida |
| 2 | 50 | 0.10 | 30 | Equilíbrio exploração/explotação |
| 3 | 80 | 0.20 | 30 | Alta diversidade |

## Outputs Gerados

Salvos em `fase2_ga_otimizacao/results/`:

| Arquivo | Conteúdo |
|---|---|
| `convergence.png` | Curvas de convergência dos 3 experimentos |
| `metrics_comparison.png` | Baseline Fase 1 vs. modelo otimizado |
| `experiment_results.json` | Métricas e hiperparâmetros completos |
| `explicacao_diagnostico.txt` | Explicação clínica gerada pelo Gemini |
| `comparacao_modelos.txt` | Análise comparativa baseline vs. otimizado (Gemini) |
| `analise_experimentos.txt` | Análise dos 3 experimentos do AG (Gemini) |

---

**Nota:** Projeto acadêmico — Pós Tech FIAP. Não utilizar para diagnóstico médico real sem validação e aprovação regulatória (ANVISA).
