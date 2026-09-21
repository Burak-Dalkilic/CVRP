import os
import random
import time
from copy import deepcopy
from typing import List, Tuple, Optional, Dict

import numpy as np

from VRP import VRP


class PSO:

    def __init__(self,
                 vrp_instance: VRP,
                 n_particles: int = 100,
                 max_gens: int = None,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 max_ops: int = 5,
                 seed: int = None,
                 init_pop: Optional[List[List[int]]] = None,
                 two_opt: bool = False,
                 penalty_coef: float = 5000.0,
                 experiment_name: str = None,
                 log_dir: str = "logs"):

        self.vrp = vrp_instance
        self.n_particles = n_particles
        self.max_gens = max_gens
        self.w = float(np.clip(w, 0.0, 1.0))
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.max_ops = max(1, int(max_ops))
        self.seed = seed
        self.init_pop = deepcopy(init_pop) if init_pop is not None else None
        self.two_opt_enabled = bool(two_opt)
        self.penalty_coef = float(penalty_coef)

        # RNGs
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Logging
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_filename = f"{log_dir}/PSO_{experiment_name}_seed{seed}.txt"

        # Internal state
        self.n_nodes = len(self.vrp.coords)
        self.population = self._init_population(self.n_nodes, self.n_particles, self.vrp.depot)
        self.eval_cache: Dict[Tuple[int, ...], float] = {}
        self.values = [self._eval_cached(p) for p in self.population]

        self.pbest = deepcopy(self.population)
        self.pbest_vals = self.values.copy()

        gidx = int(np.argmin(self.values))
        self.gbest = deepcopy(self.population[gidx])
        self.gbest_val = float(self.values[gidx])

        self.velocities: List[List[Tuple[int, int]]] = [[] for _ in range(self.n_particles)]
        self.history: List[float] = [self.gbest_val]

    # -------------------------------------------------

    def _perm_key(self, perm: List[int]) -> Tuple[int, ...]:
        return tuple(perm)

    def _eval_cached(self, perm: List[int]) -> float:
        k = self._perm_key(perm)
        if k in self.eval_cache:
            return self.eval_cache[k]
        val, _ = self.vrp.evaluate_permutation(perm, penalty_coef=self.penalty_coef)
        self.eval_cache[k] = val
        return val

    def random_permutation(self, n: int, depot: int = 0) -> List[int]:
        perm = [i for i in range(n) if i != depot]
        self.rng.shuffle(perm)
        return perm

    def _init_population(self, n_nodes: int, n_pop: int, depot: int = 0):
        population = []

        if self.init_pop is not None:
            for p in self.init_pop:
                population.append(deepcopy(p))

            while len(population) < n_pop:
                base = self.init_pop[len(population) % len(self.init_pop)]
                p = base.copy()
                self.rng.shuffle(p)
                population.append(p)
        else:
            for _ in range(n_pop):
                population.append(self.random_permutation(n_nodes, depot))

        return population[:n_pop]

    def generate_velocity_fast(self, a, b, max_ops=None):
        if max_ops is None:
            max_ops = self.max_ops
        ops = []
        a_copy = a.copy()
        pos = {a_copy[i]: i for i in range(len(a_copy))}
        for i in range(len(a_copy)):
            if a_copy[i] != b[i]:
                j = pos[b[i]]
                ops.append((i, j))
                a_copy[i], a_copy[j] = a_copy[j], a_copy[i]
                pos[a_copy[i]] = i
                pos[a_copy[j]] = j
                if len(ops) >= max_ops:
                    break
        return ops

    @staticmethod
    def apply_swap_sequence(perm, swaps):
        p = perm.copy()
        for i, j in swaps:
            p[i], p[j] = p[j], p[i]
        return p

    def two_opt_first_improvement(self, perm):
        base_val = self._eval_cached(perm)
        n = len(perm)
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = perm[:i] + perm[i:j + 1][::-1] + perm[j + 1:]
                if self._eval_cached(cand) < base_val:
                    return cand
        return perm

    # -------------------------------------------------

    def step(self, population, values):
        self.population = deepcopy(population)
        self.values = values.copy()

        for i in range(len(self.population)):
            vel_inertia = [op for op in self.velocities[i] if self.rng.random() < self.w]

            # Bilişsel Hızın (Cognitive) formüle uygun olasılıksal kırpılması
            vel_cog = self.generate_velocity_fast(self.population[i], self.pbest[i])
            r1 = self.rng.random()
            # c1 maksimum 2.0 kabul edilerek oranlanıyor
            keep_cog = int(len(vel_cog) * (self.c1 / 2.0) * r1)
            vel_cog = vel_cog[:keep_cog]

            # Sosyal Hızın (Social) formüle uygun olasılıksal kırpılması
            vel_soc = self.generate_velocity_fast(self.population[i], self.gbest)
            r2 = self.rng.random()
            keep_soc = int(len(vel_soc) * (self.c2 / 2.0) * r2)
            vel_soc = vel_soc[:keep_soc]

            new_vel = (vel_inertia + vel_cog + vel_soc)[: self.max_ops * 2]
            self.velocities[i] = new_vel

            new_perm = self.apply_swap_sequence(self.population[i], new_vel)
            if self.two_opt_enabled:
                new_perm = self.two_opt_first_improvement(new_perm)

            new_val = self._eval_cached(new_perm)

            self.population[i] = new_perm
            self.values[i] = new_val

            if new_val < self.pbest_vals[i]:
                self.pbest[i] = deepcopy(new_perm)
                self.pbest_vals[i] = new_val
                if new_val < self.gbest_val:
                    self.gbest = deepcopy(new_perm)
                    self.gbest_val = new_val

        self.history.append(self.gbest_val)
        return deepcopy(self.population), self.values.copy(), self.gbest_val, deepcopy(self.gbest)

    # -------------------------------------------------

    def run(self, verbose: bool = True):

        start_time = time.perf_counter()
        for gen in range(self.max_gens):
            self.population, self.values, gval, _ = self.step(self.population, self.values)
            if verbose:
                print(f"[PSO] Iter {gen + 1}: Best = {gval:.2f}")

        best_cost, best_routes = self.vrp.evaluate_permutation(
            self.gbest, penalty_coef=self.penalty_coef
        )
        total_time = time.perf_counter() - start_time
        with open(self.log_filename, "w") as f:
            f.write(f"Seed: {self.seed}\n")
            f.write(f"BestCost: {best_cost}\n")
            f.write(f"TotalElapsedTime(sec): {total_time:.4f}\n")
            f.write("History:\n")
            for v in self.history:
                f.write(f"{v}\n")

        return {
            "best_perm": deepcopy(self.gbest),
            "best_routes": deepcopy(best_routes),
            "best_cost": float(best_cost),
            "history": deepcopy(self.history)
        }
