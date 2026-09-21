import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from FireflyAlgorithm import FireflyAlgorithm
from GeneticAlgorithm import GeneticAlgorithm
from HybridGAPSO import HybridGAPSO
from HybridPSOFA import HybridPSOFA
from HybridGAFA import HybridGAFA
from HybridAlgorithm import HybridAlgorithm
from PSO import PSO
from VRP import VRP

N_RUNS = 30
BASE_SEED = 1001
MAX_GENS =1000
EXPERIMENT_NAME="A-n32-k5"

fa_results = []
pso_results = []
ga_results = []
hybridGAPSO_results = []
HybridPSOFA_results = []
HybridGAFA_results = []
hybrid_results = []

for run in range(N_RUNS):

    seed = BASE_SEED + run
    print(f"Run {run+1}, Seed = {seed}")

    vrp = VRP("datasets/small/"+ EXPERIMENT_NAME +".vrp")

    fa = FireflyAlgorithm(vrp, max_gens =MAX_GENS, seed=seed,experiment_name=EXPERIMENT_NAME)
    fa_results.append(fa.run())

    pso = PSO(vrp, max_gens =MAX_GENS, seed=seed,experiment_name=EXPERIMENT_NAME)
    pso_results.append(pso.run())

    ga = GeneticAlgorithm(vrp, max_gens =MAX_GENS, seed=seed,experiment_name=EXPERIMENT_NAME)
    ga_results.append(ga.run())

    hybridGAPSO = HybridGAPSO(vrp, max_gens =MAX_GENS,seed=seed,experiment_name=EXPERIMENT_NAME)
    hybridGAPSO_results.append(hybridGAPSO.run())

    hybridPSOFA = HybridPSOFA(vrp, max_gens =MAX_GENS,seed=seed,experiment_name=EXPERIMENT_NAME)
    HybridPSOFA_results.append(hybridPSOFA.run())

    hybridGAFA = HybridGAFA(vrp, max_gens =MAX_GENS,seed=seed,experiment_name=EXPERIMENT_NAME)
    HybridGAFA_results.append(hybridGAFA.run())

    hybrid = HybridAlgorithm(vrp, max_gens =MAX_GENS,seed=seed,experiment_name=EXPERIMENT_NAME)
    hybrid_results.append(hybrid.run())

    plt.figure(figsize=(10,6))

    plt.plot(ga_results[run]["history"], label="GA")
    plt.plot(pso_results[run]["history"], label="PSO")
    plt.plot(fa_results[run]["history"], label="FA")
    plt.plot(hybrid_results[run]["history"], label="HYBRID")


    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"results/convergence_{EXPERIMENT_NAME}_run{run + 1}.png"
    plt.savefig(filename, dpi=300)
    plt.close()