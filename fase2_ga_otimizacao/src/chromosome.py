"""
chromosome.py
Representação e decodificação do cromossomo para otimização de hiperparâmetros.

Cada indivíduo é um vetor de 6 genes reais em [0, 1].
A decodificação mapeia cada gene para um hiperparâmetro concreto do pipeline sklearn.

Genes:
  [0] scaler_gene      → tipo de scaler  (categorical)
  [1] use_pca_gene     → usar PCA?       (boolean)
  [2] pca_var_gene     → variância PCA   (float  0.80–0.99)
  [3] C_gene           → C do LogReg     (log10: 10^(-3) a 10^2)
  [4] solver_gene      → solver LogReg   (categorical)
  [5] max_iter_gene    → max_iter        (int 200–2000)
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Espaço de busca
# ---------------------------------------------------------------------------

SCALERS  = ["StandardScaler", "RobustScaler", "MinMaxScaler"]
SOLVERS  = ["lbfgs", "liblinear", "saga"]

PCA_VAR_MIN  = 0.80
PCA_VAR_MAX  = 0.99
C_LOG_MIN    = -3.0    # 10^-3 = 0.001
C_LOG_MAX    =  2.0    # 10^2  = 100
MAX_ITER_MIN = 200
MAX_ITER_MAX = 2000

N_GENES = 6


# ---------------------------------------------------------------------------
# Decodificação
# ---------------------------------------------------------------------------

def decode(genes: List[float]) -> Dict[str, Any]:
    """
    Converte o vetor de genes [0,1]^6 em hiperparâmetros concretos.

    Returns:
        dict com chaves: scaler, use_pca, pca_variance, C, solver, max_iter
    """
    scaler_idx  = int(genes[0] * len(SCALERS))
    scaler_idx  = min(scaler_idx, len(SCALERS) - 1)

    use_pca     = genes[1] >= 0.5

    pca_var     = PCA_VAR_MIN + genes[2] * (PCA_VAR_MAX - PCA_VAR_MIN)

    C_log       = C_LOG_MIN + genes[3] * (C_LOG_MAX - C_LOG_MIN)
    C           = 10 ** C_log

    solver_idx  = int(genes[4] * len(SOLVERS))
    solver_idx  = min(solver_idx, len(SOLVERS) - 1)

    max_iter    = int(MAX_ITER_MIN + genes[5] * (MAX_ITER_MAX - MAX_ITER_MIN))

    return {
        "scaler":       SCALERS[scaler_idx],
        "use_pca":      use_pca,
        "pca_variance": round(pca_var, 4),
        "C":            round(C, 6),
        "solver":       SOLVERS[solver_idx],
        "max_iter":     max_iter,
    }


def random_genes(rng: random.Random = None) -> List[float]:
    """Gera um cromossomo aleatório."""
    r = rng or random
    return [r.random() for _ in range(N_GENES)]


def baseline_genes() -> List[float]:
    """
    Genes que reproduzem o modelo vencedor da Fase 1:
      RobustScaler + PCA(0.90) + LogReg(C=1, solver=lbfgs, max_iter=1000)
    """
    scaler_gene   = SCALERS.index("RobustScaler") / len(SCALERS) + 0.01
    use_pca_gene  = 0.9          # True
    pca_var_gene  = (0.90 - PCA_VAR_MIN) / (PCA_VAR_MAX - PCA_VAR_MIN)
    C_gene        = (0.0  - C_LOG_MIN)   / (C_LOG_MAX   - C_LOG_MIN)   # C=1 → log10=0
    solver_gene   = SOLVERS.index("lbfgs") / len(SOLVERS) + 0.01
    max_iter_gene = (1000 - MAX_ITER_MIN) / (MAX_ITER_MAX - MAX_ITER_MIN)

    return [scaler_gene, use_pca_gene, pca_var_gene, C_gene, solver_gene, max_iter_gene]


def clip_genes(genes: List[float]) -> List[float]:
    """Garante que todos os genes estejam em [0, 1]."""
    return [max(0.0, min(1.0, g)) for g in genes]


def genes_to_str(genes: List[float]) -> str:
    """Representação legível do cromossomo decodificado."""
    hp = decode(genes)
    pca_str = f"PCA({hp['pca_variance']:.2f})" if hp["use_pca"] else "sem PCA"
    return (
        f"[{hp['scaler']} | {pca_str} | "
        f"C={hp['C']:.4f} | {hp['solver']} | iter={hp['max_iter']}]"
    )
