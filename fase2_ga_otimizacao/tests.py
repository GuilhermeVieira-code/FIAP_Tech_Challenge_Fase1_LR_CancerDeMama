"""
tests.py
========
Testes unitários para o AG e a avaliação do modelo.

Execução:
    pytest fase2_ga_otimizacao/tests.py -v
    pytest fase2_ga_otimizacao/tests.py -v -k "TestChromosome"   # só um grupo
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import pytest
import numpy as np
from sklearn.model_selection import train_test_split

from fase2_ga_otimizacao.genetic_algorithm import (
    # Cromossomo
    N_GENES, decode, random_genes, baseline_genes, clip_genes,
    SCALERS, SOLVERS, PCA_VAR_MIN, PCA_VAR_MAX,
    # Avaliador
    load_data, build_pipeline, evaluate_genes, evaluate_full, compute_fitness,
    # AG
    Individual, GAConfig,
    uniform_crossover, arithmetic_crossover,
    gaussian_mutation, tournament_selection,
    run_genetic_algorithm,
)


# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    return load_data()

@pytest.fixture(scope="module")
def split_data(data):
    X, y = data
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------------------------------------------------------------
# Cromossomo e decodificação
# ---------------------------------------------------------------------------

class TestChromosome:
    def test_tamanho_correto(self):
        assert len(random_genes()) == N_GENES

    def test_genes_em_intervalo(self):
        for _ in range(20):
            assert all(0.0 <= g <= 1.0 for g in random_genes())

    def test_baseline_decodifica_fase1(self):
        hp = decode(baseline_genes())
        assert hp["scaler"] == "RobustScaler"
        assert hp["use_pca"] is True
        assert abs(hp["pca_variance"] - 0.90) < 0.02
        assert hp["solver"] == "lbfgs"

    def test_decode_scaler_valido(self):
        for _ in range(30):
            assert decode(random_genes())["scaler"] in SCALERS

    def test_decode_solver_valido(self):
        for _ in range(30):
            assert decode(random_genes())["solver"] in SOLVERS

    def test_decode_pca_var_no_intervalo(self):
        for _ in range(30):
            var = decode(random_genes())["pca_variance"]
            assert PCA_VAR_MIN <= var <= PCA_VAR_MAX

    def test_decode_C_positivo(self):
        for _ in range(30):
            assert decode(random_genes())["C"] > 0

    def test_clip_corrige_fora_do_intervalo(self):
        g = [1.5, -0.3, 0.5, 1.1, 0.0, -0.1]
        assert all(0.0 <= x <= 1.0 for x in clip_genes(g))


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------

class TestCrossover:
    def test_uniforme_tamanho(self):
        c1, c2 = uniform_crossover(random_genes(), random_genes())
        assert len(c1) == len(c2) == N_GENES

    def test_uniforme_valores_dos_pais(self):
        p1, p2 = random_genes(), random_genes()
        c1, c2 = uniform_crossover(p1, p2)
        for i in range(N_GENES):
            assert c1[i] in (p1[i], p2[i])
            assert c2[i] in (p1[i], p2[i])

    def test_aritmetico_tamanho(self):
        c1, c2 = arithmetic_crossover(random_genes(), random_genes())
        assert len(c1) == len(c2) == N_GENES

    def test_aritmetico_em_intervalo(self):
        for _ in range(20):
            c1, c2 = arithmetic_crossover(random_genes(), random_genes())
            assert all(0.0 <= x <= 1.0 for x in c1 + c2)


class TestMutacao:
    def test_gaussiana_tamanho(self):
        assert len(gaussian_mutation(random_genes(), 1.0, 0.1)) == N_GENES

    def test_gaussiana_em_intervalo(self):
        for _ in range(50):
            assert all(0.0 <= x <= 1.0 for x in gaussian_mutation(random_genes(), 1.0, 0.5))

    def test_taxa_zero_sem_mudanca(self):
        g = random_genes()
        assert gaussian_mutation(g[:], 0.0, 0.5) == g


class TestTorneio:
    def test_retorna_individual(self):
        pop = [Individual(genes=random_genes(), fitness=random.random()) for _ in range(10)]
        assert isinstance(tournament_selection(pop, 3), Individual)

    def test_prefere_maior_fitness(self):
        best = Individual(genes=random_genes(), fitness=1.0)
        pop  = [best] + [Individual(genes=random_genes(), fitness=0.1) for _ in range(9)]
        winners = [tournament_selection(pop, 5) for _ in range(30)]
        assert all(w.fitness >= 0.1 for w in winners)


# ---------------------------------------------------------------------------
# Avaliador de modelos
# ---------------------------------------------------------------------------

class TestDados:
    def test_shape(self, data):
        X, y = data
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 30
        assert set(y) == {0, 1}

    def test_sem_nan(self, data):
        X, _ = data
        assert not np.isnan(X).any()


class TestPipeline:
    def test_baseline_tem_scaler_e_clf(self):
        pipe = build_pipeline(decode(baseline_genes()))
        assert "scaler" in pipe.named_steps
        assert "clf" in pipe.named_steps

    def test_sem_pca(self):
        genes = random_genes(); genes[1] = 0.0
        assert "pca" not in build_pipeline(decode(genes)).named_steps

    def test_com_pca(self):
        genes = random_genes(); genes[1] = 1.0
        assert "pca" in build_pipeline(decode(genes)).named_steps


class TestFitness:
    def test_recall_tem_maior_peso(self):
        assert compute_fitness(1.0, 0.0, 0.0) > compute_fitness(0.0, 1.0, 0.0)

    def test_modelo_perfeito(self):
        assert abs(compute_fitness(1.0, 1.0, 1.0) - 1.0) < 1e-9

    def test_modelo_zero(self):
        # equity=0.0 explicitamente — modelo sem nenhuma métrica positiva
        assert compute_fitness(0.0, 0.0, 0.0, equity=0.0) == 0.0

    def test_formula_correta(self, data):
        X, y = data
        m = evaluate_genes(baseline_genes(), X, y, cv_folds=3)
        # equity agora faz parte do fitness — precisa ser passado
        expected = compute_fitness(m["recall"], m["specificity"], m["f1"], m["equity"])
        assert abs(m["fitness"] - expected) < 1e-6

    def test_metricas_em_intervalo(self, data):
        X, y = data
        m = evaluate_genes(random_genes(), X, y, cv_folds=3)
        for k in ("recall", "specificity", "f1", "accuracy", "fitness"):
            assert 0.0 <= m[k] <= 1.0

    def test_baseline_recall_acima_limiar(self, data):
        X, y = data
        m = evaluate_genes(baseline_genes(), X, y, cv_folds=3)
        assert m["recall"] >= 0.85


class TestAvaliacaoFull:
    def test_matriz_confusao_fecha(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        cm = evaluate_full(baseline_genes(), X_train, X_test, y_train, y_test)["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_test)

    def test_recall_baseline_acima_90(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        assert evaluate_full(baseline_genes(), X_train, X_test, y_train, y_test)["recall"] >= 0.90

    def test_relatorio_contem_classes(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        report = evaluate_full(baseline_genes(), X_train, X_test, y_train, y_test)["report"]
        assert "Benigno" in report and "Maligno" in report


# ---------------------------------------------------------------------------
# Execução do AG
# ---------------------------------------------------------------------------

class TestAG:
    def test_fitness_nao_regride(self, data):
        X, y = data
        cfg = GAConfig(population_size=10, n_generations=3, cv_folds=3, seed=0)
        result = run_genetic_algorithm(cfg, X, y, verbose=False)
        assert result.best_individual.fitness >= result.best_fitness_hist[0] - 1e-6

    def test_historico_com_tamanho_correto(self, data):
        X, y = data
        n_gen = 4
        cfg = GAConfig(population_size=8, n_generations=n_gen, cv_folds=3, seed=0)
        result = run_genetic_algorithm(cfg, X, y, verbose=False)
        assert len(result.best_fitness_hist) == n_gen
        assert len(result.avg_fitness_hist)  == n_gen

    def test_genes_em_intervalo(self, data):
        X, y = data
        cfg = GAConfig(population_size=8, n_generations=3, cv_folds=3, seed=0)
        result = run_genetic_algorithm(cfg, X, y, verbose=False)
        assert all(0.0 <= g <= 1.0 for g in result.best_individual.genes)

    def test_fitness_positivo(self, data):
        X, y = data
        cfg = GAConfig(population_size=8, n_generations=3, cv_folds=3, seed=0)
        result = run_genetic_algorithm(cfg, X, y, verbose=False)
        assert result.best_individual.fitness > 0
