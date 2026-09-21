import os
import random
import time
from copy import deepcopy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from FireflyAlgorithm import FireflyAlgorithm
from PSO import PSO
from VRP import VRP

# ============================================================
# 🔹 FUZZY SYSTEM (PSO – FA)
# ============================================================

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

diversity = ctrl.Antecedent(np.linspace(0, 1, 101), 'diversity')
iteration = ctrl.Antecedent(np.linspace(0, 1, 101), 'iteration')

w_PSO = ctrl.Consequent(np.linspace(0, 1, 101), 'w_PSO')
w_FA = ctrl.Consequent(np.linspace(0, 1, 101), 'w_FA')

for var in [diversity, iteration]:
    var['VL'] = fuzz.trapmf(var.universe, [0.0, 0.0, 0.1, 0.25])
    var['L'] = fuzz.trapmf(var.universe, [0.1, 0.25, 0.35, 0.5])
    var['M'] = fuzz.trapmf(var.universe, [0.35, 0.5, 0.6, 0.75])
    var['H'] = fuzz.trapmf(var.universe, [0.6, 0.75, 0.85, 1.0])
    var['VH'] = fuzz.trapmf(var.universe, [0.8, 0.9, 1.0, 1.0])

for w in [w_PSO, w_FA]:
    w['VL'] = fuzz.trapmf(w.universe, [0.0, 0.0, 0.1, 0.25])
    w['L'] = fuzz.trapmf(w.universe, [0.1, 0.25, 0.35, 0.5])
    w['M'] = fuzz.trapmf(w.universe, [0.35, 0.5, 0.6, 0.75])
    w['H'] = fuzz.trapmf(w.universe, [0.6, 0.75, 0.85, 1.0])
    w['VH'] = fuzz.trapmf(w.universe, [0.8, 0.9, 1.0, 1.0])

rules = [
    # =========================================================================
    # DIVERSITY 'VL' (Very Low Diversity)
    # =========================================================================
    # Early stages: Strongly favors FA for attraction-based intermediate search to prevent premature convergence.
    ctrl.Rule(diversity['VL'] & iteration['VL'], (w_PSO['L'], w_FA['VH'])),
    ctrl.Rule(diversity['VL'] & iteration['L'], (w_PSO['L'], w_FA['VH'])),
    # Mid stages: Starts balancing PSO and FA.
    ctrl.Rule(diversity['VL'] & iteration['M'], (w_PSO['M'], w_FA['H'])),
    # Late stages: Shifts priority to PSO for final convergence.
    ctrl.Rule(diversity['VL'] & iteration['H'], (w_PSO['H'], w_FA['M'])),
    ctrl.Rule(diversity['VL'] & iteration['VH'], (w_PSO['H'], w_FA['M'])),

    # =========================================================================
    # DIVERSITY 'L' (Low Diversity)
    # =========================================================================
    # Early stages: Favors FA for local search capabilities.
    ctrl.Rule(diversity['L'] & iteration['VL'], (w_PSO['L'], w_FA['H'])),
    ctrl.Rule(diversity['L'] & iteration['L'], (w_PSO['L'], w_FA['H'])),
    # Mid stages: Balanced approach between PSO and FA[cite: 1].
    ctrl.Rule(diversity['L'] & iteration['M'], (w_PSO['M'], w_FA['M'])),
    # Late stages: Favors PSO to converge on best known solutions.
    ctrl.Rule(diversity['L'] & iteration['H'], (w_PSO['H'], w_FA['M'])),
    ctrl.Rule(diversity['L'] & iteration['VH'], (w_PSO['H'], w_FA['M'])),

    # =========================================================================
    # DIVERSITY 'M' (Medium Diversity)
    # =========================================================================
    # Early stages: Balanced exploration and convergence.
    ctrl.Rule(diversity['M'] & iteration['VL'], (w_PSO['M'], w_FA['M'])),
    ctrl.Rule(diversity['M'] & iteration['L'], (w_PSO['M'], w_FA['M'])),
    # Mid stages: Starts favoring PSO[cite: 1].
    ctrl.Rule(diversity['M'] & iteration['M'], (w_PSO['H'], w_FA['M'])),
    # Late stages: Strongly favors PSO for exploitation[cite: 1].
    ctrl.Rule(diversity['M'] & iteration['H'], (w_PSO['H'], w_FA['L'])),
    ctrl.Rule(diversity['M'] & iteration['VH'], (w_PSO['H'], w_FA['L'])),

    # =========================================================================
    # DIVERSITY 'H' (High Diversity)
    # =========================================================================
    # Early stages: Favors PSO since the population is already structurally diverse.
    ctrl.Rule(diversity['H'] & iteration['VL'], (w_PSO['H'], w_FA['L'])),
    ctrl.Rule(diversity['H'] & iteration['L'], (w_PSO['H'], w_FA['L'])),
    # Mid stages: Strongly favors PSO, minimizes FA[cite: 1].
    ctrl.Rule(diversity['H'] & iteration['M'], (w_PSO['H'], w_FA['VL'])),
    # Late stages: Very strongly favors PSO for rapid convergence[cite: 1].
    ctrl.Rule(diversity['H'] & iteration['H'], (w_PSO['VH'], w_FA['VL'])),
    ctrl.Rule(diversity['H'] & iteration['VH'], (w_PSO['VH'], w_FA['VL'])),

    # =========================================================================
    # DIVERSITY 'VH' (Very High Diversity)
    # =========================================================================
    # Early/Mid stages: Strongly favors PSO to guide highly scattered solutions toward promising regions.
    ctrl.Rule(diversity['VH'] & iteration['VL'], (w_PSO['H'], w_FA['VL'])),
    ctrl.Rule(diversity['VH'] & iteration['L'], (w_PSO['H'], w_FA['VL'])),
    ctrl.Rule(diversity['VH'] & iteration['M'], (w_PSO['H'], w_FA['VL'])),
    # Late stages: Very strongly favors PSO to pull scattered particles toward optima.
    ctrl.Rule(diversity['VH'] & iteration['H'], (w_PSO['VH'], w_FA['VL'])),
    ctrl.Rule(diversity['VH'] & iteration['VH'], (w_PSO['VH'], w_FA['VL']))
]

weight_ctrl = ctrl.ControlSystem(rules)
weight_sim = ctrl.ControlSystemSimulation(weight_ctrl)


# ============================================================
# 🔹 HYBRID PSO–FA
# ============================================================

class HybridPSOFA:

    def __init__(self,
                 vrp: VRP,
                 n_agents=100,
                 max_gens=200,
                 pso_params=None,
                 fa_params=None,
                 seed=None,
                 init_pop=None,
                 log_dir="logs",
                 experiment_name="PSO_FA"):

        self.vrp = vrp
        self.n_agents = n_agents
        self.max_gens = max_gens
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        os.makedirs(log_dir, exist_ok=True)
        self.log_filename = f"{log_dir}/HYBRID_PSO_FA_{experiment_name}_seed{seed}.txt"

        # --- PSO ve FA nesneleri ---
        self.pso = PSO(vrp, n_agents, **(pso_params or {"w": 0.7, "c1": 1.5, "c2": 1.5}), init_pop=deepcopy(init_pop))

        self.fa = FireflyAlgorithm(vrp, n_agents, **(fa_params or {"crossover_rate": 0.8, "mutation_rate": 0.2}),
                                   init_pop=deepcopy(init_pop))

        # --- Popülasyon ---
        self.population = deepcopy(init_pop) if init_pop else [
            self.random_perm(len(vrp.coords), vrp.depot)
            for _ in range(n_agents)
        ]
        self.values = [vrp.evaluate_permutation(p)[0] for p in self.population]

        idx = np.argmin(self.values)
        self.gbest = deepcopy(self.population[idx])
        self.gbest_val = self.values[idx]

        self.history = [self.gbest_val]
        self.diversity_history = [self.population_diversity(self.population)]
        self.strategy_history = []
        self.stagnation_counter = 0
        self.stagnation_threshold = 10  # 10 iter iyileşme yoksa stagnation

    # -------------------------------------------------
    def random_perm(self, n, depot=0):
        perm = [i for i in range(n) if i != depot]
        random.shuffle(perm)
        return perm

    # 🔹 DÜZELTİLDİ: Edge-Frequency tabanlı Shannon Entropisi
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

        start = time.perf_counter()

        for gen in range(self.max_gens):

            # 🔒 SAFE DIVERSITY
            diversity_val = max(self.population_diversity(self.population), 1e-6)

            # 🔹 Fuzzy ağırlıkları hesapla
            weight_sim.input['diversity'] = diversity_val
            weight_sim.input['iteration'] = gen / self.max_gens
            weight_sim.compute()

            # 🔒 FAIL-SAFE FUZZY OUTPUT
            if ('w_PSO' not in weight_sim.output) or ('w_FA' not in weight_sim.output):
                w_pso, w_fa = 0.5, 0.5
            else:
                w_pso = weight_sim.output['w_PSO']
                w_fa = weight_sim.output['w_FA']

            weights = np.array([w_pso, w_fa])
            weights /= weights.sum()

            # -------- PSO adımı --------
            pop_pso, vals_pso, val_pso, perm_pso = self.pso.step(deepcopy(self.population), deepcopy(self.values))

            # -------- FA adımı --------
            pop_fa, vals_fa, val_fa, perm_fa = self.fa.step(deepcopy(self.population), deepcopy(self.values))

            # 🔹 YENİLİK: Her algoritmanın ürettiği popülasyonun çeşitliliğini ölç
            div_pso = self.population_diversity(pop_pso)
            div_fa = self.population_diversity(pop_fa)
            div_vals = np.array([div_pso, div_fa])

            # KARAR MEKANİZMASI
            if self.stagnation_counter < self.stagnation_threshold:
                costs = np.array([val_pso, val_fa])
                cost_diff = costs.max() - costs.min()

                if cost_diff == 0:
                    costs_norm = np.zeros(2)
                else:
                    costs_norm = (costs - costs.min()) / cost_diff

                fitness_score = 1.0 - costs_norm

                beta = 0.7
                scores = (beta * weights) + ((1 - beta) * fitness_score)
                idx = np.argmax(scores)

            else:
                # Eşiğe ulaşıldı (Stagnation) -> Sadece Çeşitliliği Maksimize Edeni Seç
                idx = np.argmax(div_vals)
                self.stagnation_counter = 0
                print("    [!] STAGNATION RESET: Diversity shock applied!")

            strategies = [
                ("PSO", pop_pso, vals_pso, val_pso, perm_pso),
                ("FA", pop_fa, vals_fa, val_fa, perm_fa)
            ]

            strategy, self.population, self.values, best_val, best_perm = strategies[idx]

            # 🔹 En iyi çözümü güncelle
            if best_val < self.gbest_val:
                self.gbest_val = best_val
                self.gbest = deepcopy(best_perm)
                # iyileşme oldu → stagnation sıfırla
                self.stagnation_counter = 0
            else:
                # iyileşme yok → stagnation artır
                self.stagnation_counter += 1

            self.history.append(self.gbest_val)
            self.diversity_history.append(diversity_val)
            self.strategy_history.append(strategy)

            print(
                f"[PSO–FA] Gen {gen + 1:03d} | Best {self.gbest_val:.2f} | Strategy: {strategy} | Div: {diversity_val:.3f}")

        total_time = time.perf_counter() - start
        best_cost, best_routes = self.vrp.evaluate_permutation(self.gbest)

        # 🔹 Log dosyası
        with open(self.log_filename, "w") as f:
            f.write(f"Seed: {self.seed}\n")
            f.write(f"BestCost: {best_cost}\n")
            f.write(f"ElapsedTime(sec): {total_time:.4f}\n")
            f.write("History:\n")
            for v in self.history:
                f.write(f"{v}\n")

        return {
            "best_cost": best_cost,
            "best_routes": best_routes,
            "history": self.history,
            "diversity": self.diversity_history,
            "strategy": self.strategy_history
        }
