"""
genetic_algorithm.py
Algoritmo Genético para otimização de rotas de saúde da mulher (VRP).

Operadores:
  - Representação:  permutação de IDs dos pontos de atendimento
  - Seleção:        Torneio (tournament selection)
  - Crossover:      OX — Order Crossover (preserva sub-sequências)
  - Mutação:        Swap ou inversão de subsequência (escolha aleatória)
  - Elitismo:       os melhores indivíduos passam diretamente para a próxima geração
"""

import random
import copy
import time
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field

from .data_generator import ServicePoint
from .fitness import compute_fitness


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """Um indivíduo é uma rota — permutação dos IDs dos pontos."""
    chromosome: List[int]
    fitness: float = float("inf")

    def __lt__(self, other: "Individual") -> bool:
        return self.fitness < other.fitness

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness:.2f}, route={self.chromosome})"


@dataclass
class GAConfig:
    """Configuração do Algoritmo Genético."""
    population_size: int   = 100
    n_generations:   int   = 50
    mutation_rate:   float = 0.05
    crossover_rate:  float = 0.85
    tournament_size: int   = 3
    elitism_size:    int   = 2     # número de elites preservadas por geração
    seed:            int   = 42


@dataclass
class GAResult:
    """Resultado de uma execução do AG."""
    best_individual:   Individual
    best_fitness_hist: List[float]    # melhor fitness por geração
    avg_fitness_hist:  List[float]    # fitness médio por geração
    execution_time_s:  float
    config:            GAConfig

    def summary(self) -> str:
        lines = [
            "=" * 55,
            " RESULTADO DO ALGORITMO GENÉTICO",
            "=" * 55,
            f"  Pop={self.config.population_size}  |  "
            f"Gerações={self.config.n_generations}  |  "
            f"Mutação={self.config.mutation_rate}",
            f"  Melhor fitness final : {self.best_individual.fitness:.4f}",
            f"  Fitness inicial      : {self.best_fitness_hist[0]:.4f}",
            f"  Melhoria             : "
            f"{(1 - self.best_individual.fitness / self.best_fitness_hist[0]) * 100:.1f}%",
            f"  Tempo de execução    : {self.execution_time_s:.2f}s",
            f"  Rota ótima           : {self.best_individual.chromosome}",
            "=" * 55,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------

def _ox_crossover(
    parent1: List[int],
    parent2: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Order Crossover (OX): operador clássico para permutações.

    1. Seleciona uma sub-sequência aleatória do parent1
    2. Copia essa sub-sequência para o filho na mesma posição
    3. Preenche o restante na ordem em que aparecem no parent2

    Returns:
        Dois filhos (listas de inteiros)
    """
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))

    def _fill(p1: List[int], p2: List[int]) -> List[int]:
        child = [None] * n
        child[a:b+1] = p1[a:b+1]
        segment = set(p1[a:b+1])
        fill_vals = [x for x in p2 if x not in segment]
        idx = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill_vals[idx]
                idx += 1
        return child

    return _fill(parent1, parent2), _fill(parent2, parent1)


def _swap_mutation(chromosome: List[int]) -> List[int]:
    """Troca dois genes aleatórios de posição."""
    chrom = chromosome[:]
    i, j = random.sample(range(len(chrom)), 2)
    chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def _inversion_mutation(chromosome: List[int]) -> List[int]:
    """Inverte uma subsequência aleatória do cromossomo."""
    chrom = chromosome[:]
    a, b = sorted(random.sample(range(len(chrom)), 2))
    chrom[a:b+1] = chrom[a:b+1][::-1]
    return chrom


def mutate(chromosome: List[int], mutation_rate: float) -> List[int]:
    """Aplica mutação com probabilidade mutation_rate (swap ou inversão)."""
    if random.random() < mutation_rate:
        op = random.choice([_swap_mutation, _inversion_mutation])
        return op(chromosome)
    return chromosome


# ---------------------------------------------------------------------------
# Seleção por torneio
# ---------------------------------------------------------------------------

def tournament_selection(
    population: List[Individual],
    tournament_size: int,
) -> Individual:
    """
    Seleciona o melhor indivíduo de um subgrupo aleatório.

    Args:
        population: lista de indivíduos avaliados
        tournament_size: tamanho do torneio

    Returns:
        Indivíduo vencedor (menor fitness)
    """
    contestants = random.sample(population, min(tournament_size, len(population)))
    return min(contestants, key=lambda ind: ind.fitness)


# ---------------------------------------------------------------------------
# Inicialização da população
# ---------------------------------------------------------------------------

def _create_individual(
    point_ids: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
) -> Individual:
    chrom = point_ids[:]
    random.shuffle(chrom)
    f = compute_fitness(chrom, points, dist_matrix)
    return Individual(chromosome=chrom, fitness=f)


def initialize_population(
    config: GAConfig,
    point_ids: List[int],
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
) -> List[Individual]:
    """
    Gera a população inicial com indivíduos aleatórios.
    Inclui uma solução gulosa (greedy by priority) para acelerar convergência.
    """
    population: List[Individual] = []

    # Solução gulosa: ordena por prioridade depois distância
    greedy_chrom = sorted(point_ids, key=lambda pid: (
        next(p.priority for p in points if p.id == pid),
        dist_matrix[0][pid]
    ))
    greedy_fitness = compute_fitness(greedy_chrom, points, dist_matrix)
    population.append(Individual(chromosome=greedy_chrom, fitness=greedy_fitness))

    # Restante: aleatório
    for _ in range(config.population_size - 1):
        population.append(_create_individual(point_ids, points, dist_matrix))

    return population


# ---------------------------------------------------------------------------
# Loop principal do AG
# ---------------------------------------------------------------------------

def run_genetic_algorithm(
    points: List[ServicePoint],
    dist_matrix: List[List[float]],
    config: GAConfig,
    verbose: bool = True,
    callback: Optional[Callable[[int, float], None]] = None,
) -> GAResult:
    """
    Executa o Algoritmo Genético completo.

    Args:
        points:       lista de ServicePoint
        dist_matrix:  matriz de distâncias (índice 0 = depósito)
        config:       parâmetros do AG
        verbose:      imprime progresso a cada 10 gerações
        callback:     função (geração, melhor_fitness) chamada a cada geração

    Returns:
        GAResult com o melhor indivíduo e histórico
    """
    random.seed(config.seed)
    start_time = time.time()

    point_ids = [p.id for p in points]

    # Inicialização
    population = initialize_population(config, point_ids, points, dist_matrix)
    population.sort()

    best_fitness_hist: List[float] = []
    avg_fitness_hist:  List[float] = []

    if verbose:
        print(f"\n{'='*55}")
        print(f" AG — Pop={config.population_size} | "
              f"Gerações={config.n_generations} | Mutação={config.mutation_rate}")
        print(f"{'='*55}")

    for gen in range(config.n_generations):
        # Elitismo: preserva os melhores
        new_population = copy.deepcopy(population[:config.elitism_size])

        # Gera filhos até completar a população
        while len(new_population) < config.population_size:
            parent1 = tournament_selection(population, config.tournament_size)
            parent2 = tournament_selection(population, config.tournament_size)

            if random.random() < config.crossover_rate:
                c1_chrom, c2_chrom = _ox_crossover(
                    parent1.chromosome, parent2.chromosome
                )
            else:
                c1_chrom = parent1.chromosome[:]
                c2_chrom = parent2.chromosome[:]

            c1_chrom = mutate(c1_chrom, config.mutation_rate)
            c2_chrom = mutate(c2_chrom, config.mutation_rate)

            f1 = compute_fitness(c1_chrom, points, dist_matrix)
            f2 = compute_fitness(c2_chrom, points, dist_matrix)

            new_population.append(Individual(chromosome=c1_chrom, fitness=f1))
            if len(new_population) < config.population_size:
                new_population.append(Individual(chromosome=c2_chrom, fitness=f2))

        population = sorted(new_population)

        best_f   = population[0].fitness
        avg_f    = sum(ind.fitness for ind in population) / len(population)
        best_fitness_hist.append(best_f)
        avg_fitness_hist.append(avg_f)

        if callback:
            callback(gen, best_f)

        if verbose and (gen % 10 == 0 or gen == config.n_generations - 1):
            print(f"  Gen {gen+1:>3}/{config.n_generations} | "
                  f"Melhor: {best_f:.2f} | Média: {avg_f:.2f}")

    elapsed = time.time() - start_time

    result = GAResult(
        best_individual=population[0],
        best_fitness_hist=best_fitness_hist,
        avg_fitness_hist=avg_fitness_hist,
        execution_time_s=round(elapsed, 3),
        config=config,
    )

    if verbose:
        print(result.summary())

    return result
