import os
import random
import time
from copy import deepcopy
from typing import List, Dict, Optional, Tuple

import numpy as np

from VRP import VRP


class GeneticAlgorithm:
    def __init__(
            self,
            vrp_instance: VRP,
            n_pop: int = 100,
            max_gens: int = None,
            crossover_rate: float = 0.8,
            mutation_rate: float = 0.2,
            elitism: bool = True,
            seed: int = None,
            init_pop: Optional[List[List[int]]] = None,
            experiment_name: str = None,
            log_dir: str = "logs"
    ):
        self.vrp = vrp_instance
        self.n_pop = n_pop
        self.max_gens = max_gens
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.init_pop = deepcopy(init_pop) if init_pop is not None else None

        # RNG
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # State
        self.best_val = float('inf')
        self.best_perm: Optional[List[int]] = None
        self.history: List[float] = []
        self.eval_cache: Dict[Tuple[int, ...], float] = {}

        # Logging
        self.experiment_name = experiment_name
        self.seed = seed
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_filename = f"{log_dir}/GA_{experiment_name}_seed{seed}.txt"

    # --- helpers ---
    def _perm_to_key(self, perm: List[int]) -> Tuple[int, ...]:
        return tuple(perm)

    def eval_perm_cached(self, perm: List[int]) -> float:
        key = self._perm_to_key(perm)
        if key in self.eval_cache:
            return self.eval_cache[key]
        val, _ = self.vrp.evaluate_permutation(perm)
        self.eval_cache[key] = val
        return val

    def random_permutation(self, n: int, depot: int = 0) -> List[int]:
        perm = [i for i in range(n) if i != depot]
        self.rng.shuffle(perm)
        return perm

    def init_population(self, n_nodes, n_pop, depot=0, base_perm=None):
        population = []

        if base_perm is not None:
            for _ in range(n_pop):
                p = base_perm.copy()
                self.rng.shuffle(p)
                population.append(p)
        else:
            for _ in range(n_pop):
                population.append(self.random_permutation(n_nodes, depot))

        return population

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        # OX Crossover
        def ox(p1, p2):
            n = len(p1)
            a, b = sorted(self.rng.sample(range(n), 2))
            child = [-1] * n

            # 1. ve 2. adımlar: İlk ebeveynden segment kopyalanır
            child[a:b + 1] = p1[a:b + 1]

            # 3. ve 4. adımlar: İkinci ebeveynin ikinci kesim noktasından (b+1) başlanarak doldurulur
            p2_idx = (b + 1) % n
            child_idx = (b + 1) % n

            while -1 in child:
                if p2[p2_idx] not in child:
                    child[child_idx] = p2[p2_idx]
                    child_idx = (child_idx + 1) % n
                p2_idx = (p2_idx + 1) % n

            return child

        c1 = ox(parent1, parent2)
        c2 = ox(parent2, parent1)
        return c1, c2

    def mutate(self, perm: List[int]) -> List[int]:
        perm = perm.copy()
        if self.rng.random() < self.mutation_rate:
            i, j = self.rng.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    def tournament_selection(self, population, values, k=3):
        idxs = [self.rng.randrange(len(population)) for _ in range(k)]
        best = min(idxs, key=lambda i: values[i])
        return deepcopy(population[best])


    def step(self, population, values):
        new_pop = []
        if self.elitism:
            elite = int(np.argmin(values))
            new_pop.append(deepcopy(population[elite]))

        # Algoritma 3'teki döngü yapısı
        while len(new_pop) < len(population):
            p1 = self.tournament_selection(population, values, k=3)
            p2 = self.tournament_selection(population, values, k=3)

            if self.rng.random() < self.crossover_rate:
                c1, c2 = self.crossover(p1, p2)
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)

            c1 = self.mutate(c1)
            new_pop.append(c1)

            # Popülasyon sınırı aşılmaması için kontrol
            if len(new_pop) < len(population):
                c2 = self.mutate(c2)
                new_pop.append(c2)

        new_vals = [self.eval_perm_cached(p) for p in new_pop]
        best_idx = int(np.argmin(new_vals))
        return new_pop, new_vals, new_vals[best_idx], deepcopy(new_pop[best_idx])

    def run(self) -> Dict:

        start_time = time.perf_counter()
        n_nodes = len(self.vrp.coords)
        population = self.init_population(n_nodes, self.n_pop)
        values = [self.eval_perm_cached(p) for p in population]

        best_idx = int(np.argmin(values))
        self.best_val = values[best_idx]
        self.best_perm = deepcopy(population[best_idx])
        self.history.append(self.best_val)

        for gen in range(self.max_gens):
            population, values, gbest_val, gbest_perm = self.step(population, values)
            if gbest_val < self.best_val:
                self.best_val = gbest_val
                self.best_perm = deepcopy(gbest_perm)
            self.history.append(self.best_val)
            print(f"[GA] Iter {gen + 1}: Best Cost = {self.best_val :.2f}")

        best_cost, best_routes = self.vrp.evaluate_permutation(self.best_perm)
        total_time = time.perf_counter() - start_time

        with open(self.log_filename, "w") as f:
            f.write(f"Seed: {self.seed}\n")
            f.write(f"BestCost: {best_cost}\n")
            f.write(f"TotalElapsedTime(sec): {total_time:.4f}\n")
            f.write("History:\n")
            for v in self.history:
                f.write(f"{v}\n")

        return {
            "best_perm": self.best_perm,
            "best_routes": best_routes,
            "best_cost": best_cost,
            "history": self.history
        }
