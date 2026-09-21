import os
import random
import time
from copy import deepcopy
from typing import List, Optional

import numpy as np

from VRP import VRP


class FireflyAlgorithm:

    def __init__(
            self,
            vrp_instance: VRP,
            n_pop: int = 100,
            max_gens: int = 200,
            crossover_rate: float = 0.8,
            mutation_rate: float = 0.2,
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
        self.seed = seed

        self.init_pop = deepcopy(init_pop) if init_pop is not None else None

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # State
        self.best_val = float("inf")
        self.best_perm: Optional[List[int]] = None
        self.history: List[float] = []

        # Logging
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        self.log_filename = f"{log_dir}/FA_{experiment_name}_seed{seed}.txt"

    # ----------------------------
    # Helpers
    # ----------------------------
    def random_permutation(self, n_nodes, depot=0):
        perm = [i for i in range(n_nodes) if i != depot]
        self.rng.shuffle(perm)
        return perm

    def init_population_method(self):
        population = []
        if self.init_pop:
            for p in self.init_pop:
                population.append(deepcopy(p))
        else:
            for _ in range(self.n_pop):
                p = self.random_permutation(len(self.vrp.coords), self.vrp.depot)
                population.append(p)
        return population

    # ----------------------------
    # Light intensity
    # ----------------------------
    def light_intensity(self, cost):
        return 1.0 / (cost + 1e-6)

    def hamming_distance(self, a, b):
        return sum(1 for i in range(len(a)) if a[i] != b[i])

    # ----------------------------
    # PMX crossover
    # ----------------------------
    def pmx(self, p1, p2):
        n = len(p1)
        c1, c2 = sorted(self.rng.sample(range(n), 2))
        child = [-1] * n
        child[c1:c2] = p1[c1:c2]
        for i in range(c1, c2):
            if p2[i] not in child:
                pos = i
                while True:
                    val = p1[pos]
                    pos = p2.index(val)
                    if child[pos] == -1:
                        child[pos] = p2[i]
                        break
        for i in range(n):
            if child[i] == -1:
                child[i] = p2[i]
        return child

    # ----------------------------
    # Route-Based Mutation
    # ----------------------------
    def route_based_mutation(self, perm):
        cost, routes = self.vrp.evaluate_permutation(perm)
        mutated_routes = deepcopy(routes)

        # Rota-içi takas (Intra-route swap)
        if self.rng.random() < self.mutation_rate:
            valid_routes = [i for i, r in enumerate(mutated_routes) if len(r) >= 2]
            if valid_routes:
                r_idx = self.rng.choice(valid_routes)
                i, j = self.rng.sample(range(len(mutated_routes[r_idx])), 2)
                mutated_routes[r_idx][i], mutated_routes[r_idx][j] = mutated_routes[r_idx][j], mutated_routes[r_idx][i]

        # Rotalar-arası takas (Inter-route swap)
        if self.rng.random() < self.mutation_rate:
            if len(mutated_routes) >= 2:
                r1, r2 = self.rng.sample(range(len(mutated_routes)), 2)
                if len(mutated_routes[r1]) > 0 and len(mutated_routes[r2]) > 0:
                    i = self.rng.choice(range(len(mutated_routes[r1])))
                    j = self.rng.choice(range(len(mutated_routes[r2])))
                    mutated_routes[r1][i], mutated_routes[r2][j] = mutated_routes[r2][j], mutated_routes[r1][i]

        new_perm = []
        for r in mutated_routes:
            for node in r:
                if node != self.vrp.depot:
                    new_perm.append(node)

        return new_perm

    # ----------------------------
    # Local Search with Fitness Evaluation
    # ----------------------------
    def local_search(self, perm, best_perm, trials=3):
        if best_perm is None:
            return perm

        best_local_perm = perm.copy()
        best_local_val = self.vrp.evaluate_permutation(best_local_perm)[0]

        if self.hamming_distance(perm, best_perm) < len(perm) / 2:
            # Improved 2-opt (Kesin iyileştirme arar)
            for _ in range(trials):
                cand = best_local_perm.copy()
                i, j = sorted(self.rng.sample(range(len(cand)), 2))
                cand[i:j] = reversed(cand[i:j])

                cand_val = self.vrp.evaluate_permutation(cand)[0]
                if cand_val < best_local_val:
                    best_local_perm = cand
                    best_local_val = cand_val
        else:
            # 2-h-opt perturbasyonu (Kötüye gitmiyorsa kabul eder)
            cand = best_local_perm.copy()
            idx = self.rng.sample(range(len(cand)), 3)
            segment = [cand[i] for i in idx]
            self.rng.shuffle(segment)
            for i, v in zip(idx, segment):
                cand[i] = v

            cand_val = self.vrp.evaluate_permutation(cand)[0]
            if cand_val <= best_local_val:
                best_local_perm = cand

        return best_local_perm

    # ----------------------------
    # Step
    # ----------------------------
    def step(self, population, values):
        if self.best_perm is None:
            self.best_perm = deepcopy(population[np.argmin(values)])
            self.best_val = min(values)

        intensities = [self.light_intensity(v) for v in values]

        for i in range(self.n_pop):
            for j in range(i):
                if intensities[j] > intensities[i]:

                    if self.rng.random() < self.crossover_rate:

                        candidate = self.pmx(population[i], population[j])
                    else:
                        candidate = population[i][:]

                    # Maliyet kontrollü yerel arama
                    candidate = self.local_search(candidate, self.best_perm)

                    # Rota tabanlı kapasite korunumlu mutasyon
                    candidate = self.route_based_mutation(candidate)

                    val = self.vrp.evaluate_permutation(candidate)[0]

                    # Sadece mevcut çözümden daha iyiyse kabul et (Greedy Acceptance)
                    if val < values[i]:
                        population[i] = candidate
                        values[i] = val
                        intensities[i] = self.light_intensity(val)

        idxs = np.argsort(values)
        population = [population[k] for k in idxs]
        values = [values[k] for k in idxs]

        if values[0] < self.best_val:
            self.best_val = values[0]
            self.best_perm = deepcopy(population[0])

        return population, values, values[0], deepcopy(population[0])

    # ----------------------------
    # Run
    # ----------------------------
    def run(self):
        start_time = time.perf_counter()
        population = self.init_population_method()
        values = [self.vrp.evaluate_permutation(p)[0] for p in population]

        best_idx = np.argmin(values)
        self.best_perm = deepcopy(population[best_idx])
        self.best_val = values[best_idx]
        self.history.append(self.best_val)

        print(f"[FA] Initial Best = {self.best_val:.2f}")

        for gen in range(self.max_gens):
            population, values, val, perm = self.step(population, values)
            self.history.append(self.best_val)
            print(f"[FA] Iter {gen + 1:03d}: Best = {self.best_val:.2f}")

        best_cost, best_routes = self.vrp.evaluate_permutation(self.best_perm)
        elapsed = time.perf_counter() - start_time

        with open(self.log_filename, "w") as f:
            f.write(f"Seed: {self.seed}\n")
            f.write(f"BestCost: {best_cost}\n")
            f.write(f"TotalElapsedTime(sec): {elapsed:.4f}\n")
            f.write("History:\n")
            for v in self.history:
                f.write(f"{v}\n")

        return {
            "best_perm": self.best_perm,
            "best_routes": best_routes,
            "best_cost": best_cost,
            "history": self.history
        }