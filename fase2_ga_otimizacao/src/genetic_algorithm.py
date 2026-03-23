"""
genetic_algorithm.py
Algoritmo Genético para otimização de hiperparâmetros do modelo de diagnóstico.

Cromossomo: vetor real em [0,1]^6  (ver chromosome.py)
Seleção:    Torneio
Crossover:  Uniforme (cada gene escolhido de um dos pais com prob 0.5)
Mutação:    Gaussiana (perturbação N(0, sigma) com clipping)
Elitismo:   melhores k indivíduos preservados sem modificação
"""

import random
import time
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

import numpy as np

from .chromosome import N_GENES, random_genes, baseline_genes, clip_genes, genes_to_str, decode
from .model_evaluator import evaluate_genes, load_data


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    genes:   List[float]
    fitness: float = -1.0

    def __lt__(self, other: "Individual") -> bool:
        return self.fitness > other.fitness   # ordenação decrescente (maximizar)

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness:.4f}, {genes_to_str(self.genes)})"


@dataclass
class GAConfig:
    population_size: int   = 50
    n_generations:   int   = 30
    mutation_rate:   float = 0.10
    mutation_sigma:  float = 0.15   # desvio padrão da mutação gaussiana
    crossover_rate:  float = 0.85
    tournament_size: int   = 3
    elitism_size:    int   = 2
    cv_folds:        int   = 5
    seed:            int   = 42


@dataclass
class GAResult:
    best_individual:   Individual
    best_fitness_hist: List[float]
    avg_fitness_hist:  List[float]
    execution_time_s:  float
    config:            GAConfig

    def summary(self) -> str:
        hp = decode(self.best_individual.genes)
        pca_str = f"PCA({hp['pca_variance']:.2f})" if hp["use_pca"] else "sem PCA"
        lines = [
            "=" * 58,
            " RESULTADO DO ALGORITMO GENÉTICO",
            "=" * 58,
            f"  Pop={self.config.population_size} | "
            f"Gerações={self.config.n_generations} | "
            f"Mutação={self.config.mutation_rate}",
            f"  Melhor fitness : {self.best_individual.fitness:.4f}",
            f"  Fitness inicial: {self.best_fitness_hist[0]:.4f}",
            f"  Melhoria       : "
            f"{(self.best_individual.fitness - self.best_fitness_hist[0]):.4f}",
            f"  Tempo          : {self.execution_time_s:.1f}s",
            f"  Hiperparâmetros ótimos:",
            f"    Scaler   : {hp['scaler']}",
            f"    PCA      : {pca_str}",
            f"    C        : {hp['C']:.4f}",
            f"    Solver   : {hp['solver']}",
            f"    max_iter : {hp['max_iter']}",
            "=" * 58,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------

def uniform_crossover(
    parent1: List[float],
    parent2: List[float],
) -> Tuple[List[float], List[float]]:
    """
    Crossover uniforme: cada gene é herdado de um dos pais com prob 0.5.
    Adequado para cromossomos reais com genes independentes.
    """
    child1, child2 = [], []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(g1); child2.append(g2)
        else:
            child1.append(g2); child2.append(g1)
    return child1, child2


def arithmetic_crossover(
    parent1: List[float],
    parent2: List[float],
    alpha: float = 0.5,
) -> Tuple[List[float], List[float]]:
    """
    Crossover aritmético: filhos são combinações lineares dos pais.
    Útil para genes contínuos como C e pca_variance.
    """
    child1 = [alpha * g1 + (1 - alpha) * g2 for g1, g2 in zip(parent1, parent2)]
    child2 = [alpha * g2 + (1 - alpha) * g1 for g1, g2 in zip(parent1, parent2)]
    return clip_genes(child1), clip_genes(child2)


def gaussian_mutation(
    genes: List[float],
    mutation_rate: float,
    sigma: float,
) -> List[float]:
    """
    Mutação gaussiana: perturba cada gene com probabilidade mutation_rate.
    Perturbação: N(0, sigma), depois clipa em [0, 1].
    """
    mutated = genes[:]
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = mutated[i] + random.gauss(0, sigma)
    return clip_genes(mutated)


def tournament_selection(
    population: List[Individual],
    tournament_size: int,
) -> Individual:
    contestants = random.sample(population, min(tournament_size, len(population)))
    return max(contestants, key=lambda ind: ind.fitness)


# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------

def _eval_individual(genes, X, y, cv_folds):
    metrics = evaluate_genes(genes, X, y, cv_folds)
    return Individual(genes=genes, fitness=metrics["fitness"])


def initialize_population(
    config: GAConfig,
    X, y,
) -> List[Individual]:
    """
    Cria população inicial:
      - 1 indivíduo = configuração baseline da Fase 1
      - restantes: aleatórios
    """
    rng = random.Random(config.seed)
    population: List[Individual] = []

    # Baseline (modelo vencedor da Fase 1)
    bl_genes = baseline_genes()
    population.append(_eval_individual(bl_genes, X, y, config.cv_folds))

    # Aleatórios
    for _ in range(config.population_size - 1):
        g = [rng.random() for _ in range(N_GENES)]
        population.append(_eval_individual(g, X, y, config.cv_folds))

    return sorted(population)


# ---------------------------------------------------------------------------
# Loop principal do AG
# ---------------------------------------------------------------------------

def run_genetic_algorithm(
    config: GAConfig,
    X=None,
    y=None,
    verbose: bool = True,
    callback: Optional[Callable[[int, float], None]] = None,
) -> GAResult:
    """
    Executa o Algoritmo Genético para otimização de hiperparâmetros.

    Args:
        config:   parâmetros do AG
        X, y:     dados (carregados automaticamente se None)
        verbose:  imprime progresso
        callback: função (geração, melhor_fitness) chamada a cada geração

    Returns:
        GAResult com o melhor indivíduo e histórico
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    start_time = time.time()

    if X is None or y is None:
        X, y = load_data()

    if verbose:
        print(f"\n{'='*58}")
        print(f" AG — Pop={config.population_size} | "
              f"Ger={config.n_generations} | Mut={config.mutation_rate}")
        print(f"{'='*58}")

    population = initialize_population(config, X, y)
    best_fitness_hist: List[float] = []
    avg_fitness_hist:  List[float] = []

    for gen in range(config.n_generations):
        # Elitismo
        new_pop = copy.deepcopy(population[:config.elitism_size])

        while len(new_pop) < config.population_size:
            p1 = tournament_selection(population, config.tournament_size)
            p2 = tournament_selection(population, config.tournament_size)

            if random.random() < config.crossover_rate:
                # Alterna entre crossover uniforme e aritmético
                if random.random() < 0.5:
                    c1g, c2g = uniform_crossover(p1.genes, p2.genes)
                else:
                    alpha = random.uniform(0.3, 0.7)
                    c1g, c2g = arithmetic_crossover(p1.genes, p2.genes, alpha)
            else:
                c1g, c2g = p1.genes[:], p2.genes[:]

            c1g = gaussian_mutation(c1g, config.mutation_rate, config.mutation_sigma)
            c2g = gaussian_mutation(c2g, config.mutation_rate, config.mutation_sigma)

            c1 = _eval_individual(c1g, X, y, config.cv_folds)
            new_pop.append(c1)
            if len(new_pop) < config.population_size:
                c2 = _eval_individual(c2g, X, y, config.cv_folds)
                new_pop.append(c2)

        population = sorted(new_pop)

        best_f = population[0].fitness
        avg_f  = sum(ind.fitness for ind in population) / len(population)
        best_fitness_hist.append(best_f)
        avg_fitness_hist.append(avg_f)

        if callback:
            callback(gen, best_f)

        if verbose and (gen % 5 == 0 or gen == config.n_generations - 1):
            print(f"  Gen {gen+1:>3}/{config.n_generations} | "
                  f"Melhor: {best_f:.4f} | Média: {avg_f:.4f} | "
                  f"{genes_to_str(population[0].genes)}")

    elapsed = time.time() - start_time
    result = GAResult(
        best_individual=population[0],
        best_fitness_hist=best_fitness_hist,
        avg_fitness_hist=avg_fitness_hist,
        execution_time_s=round(elapsed, 2),
        config=config,
    )
    if verbose:
        print(result.summary())
    return result
