import random
import time
from copy import deepcopy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from FireflyAlgorithm import FireflyAlgorithm
from GeneticAlgorithm import GeneticAlgorithm
from PSO import PSO
from VRP import VRP

# ==========================================================
# INPUTS
# ==========================================================
diversity = ctrl.Antecedent(np.linspace(0, 1, 101), 'diversity')
iteration = ctrl.Antecedent(np.linspace(0, 1, 101), 'iteration')

# ==========================================================
# OUTPUTS
# ==========================================================
w_GA = ctrl.Consequent(np.linspace(0, 1, 101), 'w_GA')
w_PSO = ctrl.Consequent(np.linspace(0, 1, 101), 'w_PSO')
w_FA = ctrl.Consequent(np.linspace(0, 1, 101), 'w_FA')

# ==========================================================
# INPUT MEMBERSHIP FUNCTIONS (5×5)
# ==========================================================
diversity['VL'] = fuzz.trimf(diversity.universe, [0.0, 0.0, 0.2])
diversity['L'] = fuzz.trimf(diversity.universe, [0.0, 0.2, 0.4])
diversity['M'] = fuzz.trimf(diversity.universe, [0.3, 0.5, 0.7])
diversity['H'] = fuzz.trimf(diversity.universe, [0.6, 0.8, 1.0])
diversity['VH'] = fuzz.trimf(diversity.universe, [0.8, 1.0, 1.0])

iteration['VL'] = fuzz.trimf(iteration.universe, [0.0, 0.0, 0.2])
iteration['L'] = fuzz.trimf(iteration.universe, [0.0, 0.2, 0.4])
iteration['M'] = fuzz.trimf(iteration.universe, [0.3, 0.5, 0.7])
iteration['H'] = fuzz.trimf(iteration.universe, [0.6, 0.8, 1.0])
iteration['VH'] = fuzz.trimf(iteration.universe, [0.8, 1.0, 1.0])

# ==========================================================
# OUTPUT MEMBERSHIP FUNCTIONS (ORTAK)
# ==========================================================
for w in [w_GA, w_PSO, w_FA]:
    w['VL'] = fuzz.trimf(w.universe, [0.0, 0.0, 0.2])
    w['L'] = fuzz.trimf(w.universe, [0.1, 0.3, 0.5])
    w['M'] = fuzz.trimf(w.universe, [0.4, 0.6, 0.8])
    w['H'] = fuzz.trimf(w.universe, [0.7, 0.9, 1.0])
    w['VH'] = fuzz.trimf(w.universe, [0.8, 1.0, 1.0])

# ==========================================================
# FUZZY RULE BASE (5×5 = 25 RULES)
# Logic:
# Early/Low Div  -> GA Dominant (Restore Exploration)
# Mid/Mid Div    -> FA Dominant (Balanced Search)
# Late/High Div  -> PSO Dominant (Fast Exploitation)
# ==========================================================

rules = [

    # ================= VERY LOW DIVERSITY (Exploration - GA) =================
    ctrl.Rule(diversity['VL'] & iteration['VL'], (w_GA['VH'], w_PSO['VL'], w_FA['L'])),
    ctrl.Rule(diversity['VL'] & iteration['L'],  (w_GA['VH'], w_PSO['VL'], w_FA['L'])),
    ctrl.Rule(diversity['VL'] & iteration['M'],  (w_GA['H'],  w_PSO['VL'], w_FA['M'])),
    ctrl.Rule(diversity['VL'] & iteration['H'],  (w_GA['M'],  w_PSO['L'],  w_FA['H'])),
    ctrl.Rule(diversity['VL'] & iteration['VH'], (w_GA['M'],  w_PSO['M'],  w_FA['H'])),

    # ================= LOW DIVERSITY =================
    ctrl.Rule(diversity['L'] & iteration['VL'], (w_GA['H'], w_PSO['L'], w_FA['M'])),
    ctrl.Rule(diversity['L'] & iteration['L'],  (w_GA['H'], w_PSO['L'], w_FA['M'])),
    ctrl.Rule(diversity['L'] & iteration['M'],  (w_GA['M'], w_PSO['L'], w_FA['H'])),
    ctrl.Rule(diversity['L'] & iteration['H'],  (w_GA['M'], w_PSO['M'], w_FA['H'])),
    ctrl.Rule(diversity['L'] & iteration['VH'], (w_GA['L'], w_PSO['H'], w_FA['M'])),

    # ================= MEDIUM DIVERSITY (Balanced - FA) =================
    ctrl.Rule(diversity['M'] & iteration['VL'], (w_GA['M'],  w_PSO['M'], w_FA['VH'])),
    ctrl.Rule(diversity['M'] & iteration['L'],  (w_GA['M'],  w_PSO['M'], w_FA['VH'])),
    ctrl.Rule(diversity['M'] & iteration['M'],  (w_GA['L'],  w_PSO['M'], w_FA['VH'])),
    ctrl.Rule(diversity['M'] & iteration['H'],  (w_GA['VL'], w_PSO['H'], w_FA['H'])),
    ctrl.Rule(diversity['M'] & iteration['VH'], (w_GA['VL'], w_PSO['H'], w_FA['M'])),

    # ================= HIGH DIVERSITY =================
    ctrl.Rule(diversity['H'] & iteration['VL'], (w_GA['L'],  w_PSO['H'],  w_FA['M'])),
    ctrl.Rule(diversity['H'] & iteration['L'],  (w_GA['L'],  w_PSO['H'],  w_FA['M'])),
    ctrl.Rule(diversity['H'] & iteration['M'],  (w_GA['L'],  w_PSO['H'],  w_FA['M'])),
    ctrl.Rule(diversity['H'] & iteration['H'],  (w_GA['VL'], w_PSO['VH'], w_FA['L'])),
    ctrl.Rule(diversity['H'] & iteration['VH'], (w_GA['VL'], w_PSO['VH'], w_FA['VL'])),

    # ================= VERY HIGH DIVERSITY (Exploitation - PSO) =================
    ctrl.Rule(diversity['VH'] & iteration['VL'], (w_GA['VL'], w_PSO['VH'], w_FA['M'])),
    ctrl.Rule(diversity['VH'] & iteration['L'],  (w_GA['VL'], w_PSO['VH'], w_FA['M'])),
    ctrl.Rule(diversity['VH'] & iteration['M'],  (w_GA['VL'], w_PSO['H'],  w_FA['M'])),
    ctrl.Rule(diversity['VH'] & iteration['H'],  (w_GA['VL'], w_PSO['H'],  w_FA['L'])),
    ctrl.Rule(diversity['VH'] & iteration['VH'], (w_GA['VL'], w_PSO['VH'], w_FA['VL']))
]
# ==========================================================
# SYSTEM
# ==========================================================
weight_ctrl = ctrl.ControlSystem(rules)
weight_sim = ctrl.ControlSystemSimulation(weight_ctrl)

# ============================================================
# 🔹 HYBRID ALGORITHM
# ============================================================

import os


class HybridAlgorithm:

    def __init__(self, vrp: VRP, n_agents=100, max_gens: int = None,
                 fa_params=None, pso_params=None, ga_params=None,
                 seed: int = None, init_pop=None,
                 experiment_name=None,
                 log_dir="logs"):

        self.vrp = vrp
        self.n_agents = n_agents
        self.max_gens = max_gens
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------
        os.makedirs(log_dir, exist_ok=True)
        self.log_filename = f"{log_dir}/HYBRID_{experiment_name}_seed{seed}.txt"

        self.fa_params = fa_params or {"crossover_rate": 0.8, "mutation_rate": 0.2}
        self.pso_params = pso_params or {"w": 0.7, "c1": 1.5, "c2": 1.5}
        self.ga_params = ga_params or {"crossover_rate": 0.8, "mutation_rate": 0.2}

        self.population = deepcopy(init_pop) if init_pop else [
            self.random_permutation(len(vrp.coords), vrp.depot)
            for _ in range(n_agents)
        ]

        self.values = [vrp.evaluate_permutation(p)[0] for p in self.population]

        best_idx = np.argmin(self.values)
        self.gbest = deepcopy(self.population[best_idx])
        self.gbest_val = self.values[best_idx]

        self.fa = FireflyAlgorithm(
            vrp, n_agents, **self.fa_params, init_pop=deepcopy(self.population)
        )

        self.ga = GeneticAlgorithm(
            vrp, n_agents, **self.ga_params, init_pop=deepcopy(self.population)
        )
        self.pso = PSO(
            vrp, n_agents, **self.pso_params, init_pop=deepcopy(self.population)
        )

        self.history = [self.gbest_val]
        self.diversity_history = [self.population_diversity(self.population)]
        self.strategy_history = []
        self.stagnation_counter = 0
        self.stagnation_threshold = 10  # 10 iter iyileşme yoksa stagnation

        print(f"[HYBRID] Initial Best Cost = {self.gbest_val:.2f}")

    # -------------------------------------------------

    def random_permutation(self, n, depot=0):
        perm = [i for i in range(n) if i != depot]
        random.shuffle(perm)
        return perm

    # 🔹 DÜZELTİLDİ: Edge-Frequency tabanlı Shannon Entropisi (VRP için Q1 Standartı)
    def population_diversity(self, pop):
        """
        Edge-Frequency based Shannon Entropy.
        Measures diversity based on the uniqueness of edges (node transitions).
        Normalized to [0,1].
        """
        if len(pop) < 2:
            return 0.0

        route_len = len(pop[0])
        max_node = max(max(p) for p in pop) + 1

        edge_counts = np.zeros((max_node, max_node))

        for p in pop:
            for i in range(route_len - 1):
                edge_counts[p[i], p[i + 1]] += 1

            if hasattr(self.vrp, 'depot'):
                edge_counts[self.vrp.depot, p[0]] += 1
                edge_counts[p[-1], self.vrp.depot] += 1

        total_edges = np.sum(edge_counts)
        if total_edges == 0:
            return 0.0

        p_edges = edge_counts[edge_counts > 0] / total_edges
        entropy = -np.sum(p_edges * np.log(p_edges))

        max_possible_edges = max_node * (max_node - 1)
        max_entropy = np.log(max_possible_edges + 1e-12)

        return min(entropy / max_entropy, 1.0)

    # -------------------------------------------------

    def run(self):

        start_time = time.perf_counter()
        for gen in range(self.max_gens):

            diversity_val = max(self.population_diversity(self.population), 1e-6)

            # 🔹 FUZZY INPUT
            weight_sim.input['diversity'] = diversity_val
            weight_sim.input['iteration'] = gen / self.max_gens
            weight_sim.compute()

            weights = np.array([weight_sim.output['w_GA'], weight_sim.output['w_PSO'], weight_sim.output['w_FA']])
            weights /= weights.sum()

            # 🔹 RUN ALL ALGORITHMS

            pop_ga, vals_ga, val_ga, perm_ga = self.ga.step(deepcopy(self.population), deepcopy(self.values))
            pop_pso, vals_pso, val_pso, perm_pso = self.pso.step(deepcopy(self.population), deepcopy(self.values))
            pop_fa, vals_fa, val_fa, perm_fa = self.fa.step(deepcopy(self.population), deepcopy(self.values))

            div_ga = self.population_diversity(pop_ga)
            div_pso = self.population_diversity(pop_pso)
            div_fa = self.population_diversity(pop_fa)
            div_vals = np.array([div_ga, div_pso, div_fa])

            # KARAR MEKANİZMASI
            if self.stagnation_counter < self.stagnation_threshold:

                costs = np.array([val_ga, val_pso, val_fa])
                cost_diff = costs.max() - costs.min()

                if cost_diff == 0:
                    costs_norm = np.zeros(3)
                else:
                    costs_norm = (costs - costs.min()) / cost_diff
                fitness_score = 1.0 - costs_norm

                # Toplamsal Ağırlıklandırma: %70 Fuzzy Kararı, %30 Anlık Fitness
                beta = 0.7
                scores = (beta * weights) + ((1 - beta) * fitness_score)
                idx = np.argmax(scores)

            else:
                idx = np.argmax(div_vals)
                self.stagnation_counter = 0

            strategies = [
                ("GA", pop_ga, vals_ga, val_ga, perm_ga),
                ("PSO", pop_pso, vals_pso, val_pso, perm_pso),
                ("FA", pop_fa, vals_fa, val_fa, perm_fa)
            ]

            strategy, self.population, self.values, best_val, best_perm = strategies[idx]

            # 🔹 En iyi çözümü güncelle
            if best_val < self.gbest_val:
                self.gbest_val = best_val
                self.gbest = deepcopy(best_perm)
                # İyileşme oldu → stagnation sıfırla
                self.stagnation_counter = 0
            else:
                # İyileşme yok → stagnation artır
                self.stagnation_counter += 1

            self.history.append(self.gbest_val)
            self.diversity_history.append(diversity_val)
            self.strategy_history.append(strategy)

            print(
                f"[HYBRID] Iter {gen + 1:03d} | Cost: {self.gbest_val:.2f} | "
                f"GA:{weights[0]:.2f} PSO:{weights[1]:.2f} FA:{weights[2]:.2f} | "
                f"Strategy: {strategy} | Div: {diversity_val:.3f}"
            )

        best_cost, best_routes = self.vrp.evaluate_permutation(self.gbest)
        total_time = time.perf_counter() - start_time

        with open(self.log_filename, "w") as f:
            f.write(f"Seed: {self.seed}\n")
            f.write(f"BestCost: {best_cost}\n\n")
            f.write(f"TotalElapsedTime(sec): {total_time:.4f}\n")
            f.write("History:\n")
            for v in self.history:
                f.write(f"{v}\n")

        return {
            "best_perm": self.gbest,
            "best_routes": best_routes,
            "best_cost": best_cost,
            "history": self.history,
            "diversity_history": self.diversity_history,
            "strategy_history": self.strategy_history,
            "D": self.vrp.D,
            "demands": self.vrp.demands
        }
