# Relatório Técnico — Fase 2
## Otimização de Rotas para Distribuição de Medicamentos e Atendimento Especializado à Mulher

**Tech Challenge FIAP — Pós Tech**
**Projeto 2: VRP com Algoritmo Genético + LLM**

---

## 1. Introdução

Este projeto aborda o problema de otimização de rotas (Vehicle Routing Problem — VRP) aplicado
ao contexto de saúde pública feminina no estado de São Paulo. O sistema otimiza a sequência de
visitas a pontos de atendimento especializado à mulher, considerando restrições realistas de
prioridade clínica, capacidade logística e janelas de tempo seguras.

A integração com um Large Language Model (Google Gemini) transforma os resultados técnicos em
documentos operacionais compreensíveis para equipes de campo.

---

## 2. Motivação e Impacto Social

O Brasil enfrenta desafios críticos na atenção à saúde da mulher:

- **Mortalidade materna**: 68 mortes por 100 mil nascidos vivos (2022), acima da meta OMS
- **Violência doméstica**: 1,3 milhão de casos registrados por ano (IPEA, 2023)
- **Acesso a medicamentos hormonais**: Distribuição irregular em periferias urbanas
- **Pós-parto vulnerável**: 1 em cada 5 mulheres sofre depressão pós-parto sem suporte

Uma rota otimizada que priorize emergências obstétricas pode reduzir o tempo de resposta em
casos críticos, potencialmente salvando vidas. O sistema foi projetado com foco em:

- **Equidade**: Garantir cobertura de regiões periféricas com maior vulnerabilidade
- **Privacidade**: Nenhum dado pessoal identificável é processado pelo sistema
- **Eficiência**: Minimizar distâncias e tempos de espera para todas as pacientes

---

## 3. Formulação do Problema

### 3.1 Modelagem do VRP

O problema é modelado como um VRP com um único veículo, restrições de capacidade,
janelas de tempo e prioridade de atendimento:

**Dados de entrada:**
- Depósito central (ponto de origem e retorno)
- N pontos de atendimento com coordenadas geográficas (latitude/longitude)
- Demanda de suprimentos por ponto
- Janelas de tempo [t_start, t_end] por ponto
- Nível de prioridade (1–4) por tipo de atendimento

**Objetivo:** Minimizar a função fitness:

```
fitness = distância_total
        + penalidade_prioridade
        + penalidade_capacidade
        + penalidade_janela_tempo
```

### 3.2 Tipos de Atendimento e Prioridades

| Tipo | Prioridade | Justificativa clínica |
|------|:----------:|----------------------|
| Emergência Obstétrica | 1 | Risco de vida imediato para mãe e bebê |
| Violência Doméstica | 2 | Risco de recidiva; janela de intervenção crítica |
| Medicamento Hormonal | 3 | Interrupção causa impacto hormonal significativo |
| Pós-Parto | 4 | Importante, mas pode tolerar pequeno atraso |

### 3.3 Restrições Implementadas

**Restrição 1 — Prioridade de Atendimento:**
Emergências obstétricas devem ser visitadas no primeiro terço da rota.
Atendimentos de violência doméstica devem estar na primeira metade.
Penalidade proporcional à posição relativa na rota e à urgência.

**Restrição 2 — Capacidade do Veículo:**
- Capacidade máxima: 120 unidades de suprimento
- Distância máxima: 200 km por rota
- Penalidade: 500× o excesso de demanda; 300× o excesso de distância

**Restrição 3 — Janelas de Tempo:**
- Cada ponto possui horário de atendimento permitido (ex: 07:00–13:00)
- Velocidade média considerada: 30 km/h (tráfego urbano de São Paulo)
- Chegada antecipada → aguarda (penalidade leve de ociosidade)
- Chegada atrasada → penalidade proporcional ao atraso e à urgência

---

## 4. Algoritmo Genético

### 4.1 Representação

Cromossomo: permutação dos IDs dos pontos de atendimento.

```
Exemplo: [5, 2, 8, 1, 3, 7, 4, 6, ...]
          ↑ visita ponto 5 primeiro, depois 2, etc.
```

### 4.2 Operadores

**Inicialização:**
- 1 indivíduo greedy (ordenado por prioridade + distância do depósito)
- Demais indivíduos: permutações aleatórias

**Seleção — Torneio (tournament_size=3):**
- Seleciona k indivíduos aleatoriamente
- Retorna o de menor fitness

**Crossover — OX (Order Crossover):**
1. Copia segmento [a, b] do Parent1 para o filho
2. Preenche posições restantes na ordem de aparição em Parent2
3. Preserva sub-sequências relativas — crucial para VRP

```
Parent1: [1, 2 | 3, 4, 5 | 6, 7]
Parent2: [3, 7, 4, 1, 5, 6, 2]

Segmento copiado: [3, 4, 5]
Filho:   [7, 1, | 3, 4, 5 | 6, 2]
```

**Mutação — Swap ou Inversão (escolha aleatória):**
- Swap: troca dois genes de posição
- Inversão: reverte uma subsequência

**Elitismo:** 2 melhores indivíduos preservados por geração.

### 4.3 Experimentos

| # | Tamanho da Pop. | Taxa de Mutação | Gerações |
|---|:-:|:-:|:-:|
| 1 | 50  | 0.10 | 30 |
| 2 | 100 | 0.05 | 50 |
| 3 | 200 | 0.15 | 50 |

---

## 5. Integração com LLM (Google Gemini)

### 5.1 Modelo Utilizado
- **API:** Google Gemini 2.0 Flash (gratuita)
- **SDK:** `google-generativeai` (Python)

### 5.2 Documentos Gerados

**Manual de Instruções Operacionais:**
Documento completo para a equipe de transporte com protocolos por tipo de
atendimento, procedimentos de segurança, contatos de emergência e orientações
de abordagem sensível.

**Roteiro Narrativo de Visitas:**
Transforma a lista técnica de paradas (IDs, coordenadas, horários) em um
briefing humanizado e motivador para a equipe de campo.

**Análise Comparativa dos Experimentos:**
Interpretação técnica e prática dos resultados dos 3 experimentos do AG,
com recomendação da configuração ideal para produção.

### 5.3 Chat de Perguntas Frequentes

Interface para perguntas em linguagem natural:
- "Qual o próximo atendimento prioritário?"
- "Quantas paradas de emergência temos hoje?"
- "Há alguma janela de tempo apertada?"

Privacidade garantida: respostas baseadas apenas em dados agregados.

### 5.4 Design dos Prompts

Todos os prompts incluem:
- **Contexto de saúde feminina**: orienta o modelo para o domínio específico
- **Linguagem sensível**: evita estigmatização de pacientes em situação de vulnerabilidade
- **Privacidade**: instrui o modelo a nunca citar dados identificáveis
- **Foco prático**: respostas acionáveis para profissionais de campo

---

## 6. Visualização de Rotas

O módulo `route_visualizer.py` gera mapas HTML interativos com Folium:

- **Marcadores coloridos** por tipo de atendimento:
  - Vermelho: Emergência Obstétrica
  - Coral: Violência Doméstica
  - Azul: Medicamento Hormonal
  - Verde: Pós-Parto
- **Linha tracejada** mostrando a rota otimizada com setas de direção
- **Popups informativos** com tipo, horário previsto, janela de tempo e demanda
- **Legenda** com estatísticas da rota
- **Mapa de comparação** com as 3 rotas dos experimentos sobrepostas

---

## 7. Resultados

*(Os resultados são gerados automaticamente ao executar `python main.py`
e salvos em `fase2_vrp/results/`.)*

### 7.1 Arquivos Gerados

| Arquivo | Descrição |
|---------|-----------|
| `convergence.png` | Curvas de convergência dos 3 experimentos |
| `experiment_results.json` | Métricas completas (fitness, distância, tempo) |
| `route_map.html` | Mapa interativo da melhor rota |
| `comparison_map.html` | Comparação visual das 3 rotas |
| `manual_instrucoes.txt` | Manual operacional gerado pelo Gemini |
| `roteiro_visitas.txt` | Roteiro narrativo gerado pelo Gemini |
| `analise_experimentos.txt` | Análise comparativa gerada pelo Gemini |

---

## 8. Considerações Éticas

### 8.1 Privacidade de Dados

- Sistema opera com **dados sintéticos** gerados programaticamente
- Nenhum dado real de pacientes foi utilizado ou armazenado
- Em produção, o sistema deve ser integrado com sistemas de saúde
  sob regulamentação LGPD e HIPAA

### 8.2 Equidade no Atendimento

- O sistema prioriza tipos de atendimento com base em critérios clínicos objetivos
- A prioridade reflete **urgência médica**, não critérios socioeconômicos
- Cobertura geográfica inclui regiões periféricas de São Paulo (Grajaú, São Mateus, etc.)

### 8.3 Viés Algorítmico

- O AG minimiza uma função de custo que pode, em teoria, sistematicamente
  desfavorecer regiões distantes do depósito
- Mitigação: penalidade de prioridade garante que emergências sejam atendidas
  independentemente da localização

### 8.4 Uso Responsável do LLM

- O Gemini é usado para geração de **documentos de apoio**, não para decisões médicas
- Os prompts explicitamente proíbem o modelo de dar diagnósticos ou substituir julgamento médico
- Respostas do LLM devem ser revisadas por profissionais de saúde antes de uso

---

## 9. Limitações e Trabalhos Futuros

### Limitações Atuais

1. **Veículo único**: O sistema otimiza para um único veículo; frota múltipla requer extensão
2. **Dados sintéticos**: Validação com dados reais é necessária para uso em produção
3. **Velocidade estática**: Tráfego real de São Paulo varia drasticamente (rush hours)
4. **Sem realocação dinâmica**: Não suporta reagendamento em tempo real

### Extensões Propostas

1. **VRP com frota múltipla** e diferentes tipos de veículo (ambulância vs. caminhonete)
2. **Dados em tempo real**: Integração com APIs de trânsito (Google Maps, WAZE)
3. **Otimização multi-objetivo**: Distância + tempo + cobertura + equidade geográfica
4. **Reinforcement Learning**: Política adaptativa para reagendamento dinâmico
5. **Interface web**: Dashboard para equipes de saúde com mapa em tempo real
6. **Integração com REDE CEGONHA**: Conectar com o sistema nacional de saúde materno-infantil

---

## 10. Instruções de Execução

### Pré-requisitos

```bash
# Instalar dependências da Fase 2
pip install -r fase2_vrp/requirements.txt
```

### Configurar API Gemini

```bash
# Criar arquivo .env na raiz do repositório
echo "GEMINI_API_KEY=sua_chave_aqui" > .env
```

### Executar o pipeline completo

```bash
# Com geração de documentos LLM
python -m fase2_vrp.main

# Sem chamadas à API Gemini (mais rápido)
python -m fase2_vrp.main --skip-llm

# Customizar número de pontos e semente
python -m fase2_vrp.main --n-points 25 --seed 123
```

### Executar os testes

```bash
pytest fase2_vrp/tests/ -v
```

---

## 11. Referências

1. Laporte, G. (1992). *The vehicle routing problem: An overview of exact and approximate algorithms*. European Journal of Operational Research.
2. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
3. Davis, L. (1985). *Applying adaptive algorithms to epistatic domains*. IJCAI.
4. IBGE (2023). *Pesquisa Nacional de Saúde — Saúde da Mulher*.
5. Ministério da Saúde (2022). *Rede Cegonha — Diretrizes e Protocolos*.
6. Google AI (2024). *Gemini API Documentation*. https://ai.google.dev/
7. Python Folium (2024). *Interactive Leaflet Maps*. https://python-visualization.github.io/folium/
