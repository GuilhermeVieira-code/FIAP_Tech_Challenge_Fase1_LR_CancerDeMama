# 🏥 Sistema de Suporte ao Diagnóstico de Câncer de Mama

## 📋 Sobre o Projeto

Este projeto faz parte do **Tech Challenge - Fase 1** e implementa uma solução de Machine Learning para auxiliar no diagnóstico de câncer de mama, classificando tumores como **benignos** ou **malignos** a partir de dados estruturados de exames.

O sistema foi desenvolvido como prova de conceito de um sistema inteligente de suporte ao diagnóstico médico, com foco em acurácia e, principalmente, na **redução de falsos negativos** (Recall elevado).

## 🎯 Objetivo

Construir uma solução de IA baseada em Machine Learning capaz de processar dados clínicos e fornecer predições que auxiliem profissionais de saúde na análise inicial de exames, otimizando o tempo de triagem e apoiando decisões médicas.

## 📊 Dataset

**Fonte:** [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

- **Total de registros:** 569 pacientes
- **Features:** 30 variáveis numéricas extraídas de exames de imagem
- **Target:** Diagnóstico (B = Benigno, M = Maligno)
- **Distribuição:** 357 benignos (62.7%) | 212 malignos (37.3%)

As features incluem medidas como raio, textura, perímetro, área, suavidade, compacidade, concavidade, entre outras, calculadas em três níveis: média (mean), erro padrão (se) e pior caso (worst).

## 🗂️ Estrutura do Projeto

```
Tech_Challenge_Fase1/
│
├── data/
│   └── breast_cancer_dataset.csv      # Dataset original
│
├── models/
│   └── best_model.pkl                 # Modelo treinado final
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Análise Exploratória de Dados
│   └── 02_preprocessing_and_modeling.ipynb  # Pré-processamento e Modelagem
│
├── Dockerfile                         # Containerização do projeto
├── requirements.txt                   # Dependências Python
├── .dockerignore                      # Arquivos ignorados no build Docker
└── README.md                          # Documentação do projeto
```

## 🔍 Metodologia

### 1. Análise Exploratória (EDA)

No notebook `01_eda.ipynb`, foram realizadas as seguintes análises:

- Compreensão da estrutura do dataset (569 registros, 33 colunas)
- Análise de estatísticas descritivas
- Visualização da distribuição da variável alvo
- Análise de correlação entre variáveis
- Identificação de outliers

**Principais descobertas:**
- Dataset com estrutura adequada para classificação binária
- Variáveis majoritariamente numéricas (float64)
- Identificadas colunas irrelevantes: `id` e `Unnamed: 32` (coluna vazia)
- Escalas muito distintas entre variáveis, indicando necessidade de padronização
- Leve desbalanceamento entre classes
- Matriz de correlação mostrou alta correlação entre variáveis relacionadas ao tamanho (radius, perimeter, area), sugerindo redundância

### 2. Pré-processamento

No notebook `02_preprocessing_and_modeling.ipynb`:

- **Limpeza:** Remoção de colunas irrelevantes (`id`, `Unnamed: 32`)
- **Codificação:** Variável target convertida de categórica (B/M) para numérica (0/1)
- **Divisão:** 80% treino / 20% teste com estratificação
- **Padronização:** Testadas abordagens com StandardScaler e RobustScaler
- **Redução de Dimensionalidade:** Aplicação de PCA com 90% da variância explicada

### 3. Modelagem e Comparação

Foram treinados e avaliados diferentes algoritmos de classificação. A tabela abaixo mostra a comparação final dos melhores modelos, priorizando as métricas **Recall** e **F1-score** devido ao contexto médico:

| Modelo | Accuracy | Recall | F1-Score |
|--------|----------|--------|----------|
| **LogReg (RobustScaler + PCA 90%)** | **0.9825** | **0.9524** | **0.9756** |
| Random Forest | 0.9737 | 0.9286 | 0.9630 |
| SVM (StandardScaler) | 0.9737 | 0.9286 | 0.9630 |

### 4. Modelo Final Selecionado

✅ **Regressão Logística + RobustScaler + PCA (90% variância)**

**Métricas no conjunto de teste:**
- **Accuracy:** 0.9825
- **Recall:** 0.9524
- **F1-Score:** 0.9756

**Validação Cruzada (k=5) no conjunto de treino:**
- **Recall médio:** 0.9294
- **Desvio padrão:** 0.0606

**Justificativa da escolha:**

Conforme documentado na conclusão do notebook, o modelo foi selecionado com base no desempenho obtido na classe maligna (1), priorizando métricas críticas no contexto médico, especialmente o **Recall** (redução de falsos negativos). A validação cruzada reforçou a consistência do modelo, indicando bom desempenho e estabilidade.

## 🚀 Como Executar

### Opção 1: Com Docker (Recomendado)

```bash
# Build da imagem
docker build -t breast-cancer-diagnosis .

# Executar container
docker run -p 8888:8888 breast-cancer-diagnosis
```

Acesse o Jupyter no navegador: `http://localhost:8888`

### Opção 2: Ambiente Local

**Requisitos:**
- Python 3.9+
- pip

**Instalação:**

```bash
# Clonar repositório
git clone <seu-repositorio>
cd Tech_Challenge_Fase1

# Criar ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Iniciar Jupyter
jupyter notebook
```

**Executar notebooks na ordem:**
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing_and_modeling.ipynb`

## 📈 Interpretação dos Resultados

Conforme documentado no notebook de modelagem, o modelo final demonstrou:

- **Alta capacidade de identificar casos malignos** (Recall de 95.24%), minimizando o risco de falsos negativos
- **Excelente precisão geral** (Accuracy de 98.25%)
- **Consistência em diferentes subconjuntos dos dados** (validação cruzada com Recall médio de 92.94%)

Os experimentos realizados mostraram que a combinação de RobustScaler (mais resistente a outliers) com PCA (redução de dimensionalidade mantendo 90% da variância) proporcionou o melhor equilíbrio entre as métricas relevantes para o contexto médico.

## ⚠️ Considerações Práticas e Limitações

### Aplicabilidade Clínica

O modelo desenvolvido apresenta resultados promissores, mas deve ser entendido como um **sistema de suporte à decisão**, não como substituto do julgamento médico.

**Pontos importantes:**
- ✅ **Pode ser usado para:** Triagem inicial, segunda opinião automatizada, priorização de casos
- ❌ **NÃO substitui:** Avaliação médica completa, contexto clínico do paciente, outros exames
- ⚕️ **Decisão final:** Sempre deve ficar com o(a) médico(a)

### Próximos Passos para Produção

Antes de uso em ambiente real, seria necessário:
1. **Validação externa:** Testar em outros datasets de diferentes hospitais
2. **Análise de viés:** Verificar desempenho em diferentes grupos demográficos
3. **Calibração:** Ajustar probabilidades para refletir risco real
4. **Explicabilidade:** Implementar SHAP ou LIME para interpretar predições individuais
5. **Monitoramento:** Sistema de detecção de drift e degradação do modelo
6. **Aprovação regulatória:** Cumprir requisitos de dispositivos médicos (ANVISA, FDA)

### Limitações Conhecidas

- Dataset relativamente pequeno (569 amostras)
- Origem única dos dados (Wisconsin)
- Não considera histórico clínico do paciente
- Não avalia progressão temporal da doença
- Desbalanceamento leve de classes

## 📄 Licença

Este projeto é para fins educacionais como parte do Tech Challenge - Fase 1.

## 📚 Referências

- [Breast Cancer Wisconsin Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- Scikit-learn Documentation
- Tech Challenge - Fase 1 Guidelines

---

**Nota:** Este sistema foi desenvolvido como projeto acadêmico e não deve ser utilizado para diagnóstico médico real sem validações adicionais e aprovação regulatória apropriada.

---

# 🚑 Fase 2 — Otimização de Rotas para Saúde da Mulher (VRP + AG + LLM)

Sistema de otimização de rotas para distribuição de medicamentos e atendimento especializado à mulher em São Paulo. Utiliza Algoritmo Genético para resolver o VRP com restrições de prioridade clínica, capacidade e janelas de tempo, integrado com Google Gemini para geração de documentos operacionais.

## Tipos de Atendimento (por prioridade)

| Prioridade | Tipo | Descrição |
|:---:|---|---|
| 1 | Emergência Obstétrica | Risco de vida — atendimento imediato |
| 2 | Violência Doméstica | Janela crítica de intervenção |
| 3 | Medicamento Hormonal | Distribuição de anticoncepcionais/hormonais |
| 4 | Pós-Parto | Acompanhamento puerpério |

## 🗂️ Estrutura — Fase 2

```
fase2_vrp/
├── src/
│   ├── data_generator.py    # 20-25 pontos sintéticos em São Paulo
│   ├── genetic_algorithm.py # AG: torneio, OX crossover, swap/inversão
│   ├── fitness.py           # distância + penalidade_prioridade + capacidade + janela
│   ├── constraints.py       # Restrições e cálculo de penalidades
│   └── route_visualizer.py  # Mapa interativo Folium
├── llm/
│   ├── prompts.py           # Templates Gemini (linguagem sensível)
│   ├── report_generator.py  # Manual de instruções + roteiro narrativo
│   └── route_chat.py        # Chat em linguagem natural sobre a rota
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_ga_optimization.ipynb   # 3 experimentos do AG
├── tests/
│   ├── test_ga.py
│   └── test_fitness.py
├── results/                 # Outputs gerados (mapas, gráficos, relatórios)
├── docs/
│   └── relatorio_tecnico.md
├── main.py                  # Pipeline completo
└── requirements.txt
```

## 🚀 Como Executar — Fase 2

```bash
# 1. Instalar dependências
pip install -r fase2_vrp/requirements.txt

# 2. Configurar API Gemini (opcional)
cp .env.example .env
# Edite .env e coloque sua GEMINI_API_KEY

# 3. Executar pipeline completo
python -m fase2_vrp.main

# Sem chamadas à API Gemini (mais rápido para testes)
python -m fase2_vrp.main --skip-llm

# 4. Executar testes
pytest fase2_vrp/tests/ -v
```

## Experimentos do AG

| # | Pop | Mutação | Gerações |
|:---:|:---:|:---:|:---:|
| 1 | 50  | 0.10 | 30 |
| 2 | 100 | 0.05 | 50 |
| 3 | 200 | 0.15 | 50 |

## Outputs Gerados

Salvos em `fase2_vrp/results/`:

| Arquivo | Conteúdo |
|---|---|
| `route_map.html` | Mapa interativo da melhor rota (Folium) |
| `comparison_map.html` | Comparação visual dos 3 experimentos |
| `convergence.png` | Curvas de convergência |
| `experiment_results.json` | Métricas completas |
| `manual_instrucoes.txt` | Manual operacional (Gemini) |
| `roteiro_visitas.txt` | Roteiro narrativo (Gemini) |
| `analise_experimentos.txt` | Análise comparativa (Gemini) |