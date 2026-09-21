import math
import random
from typing import List, Tuple, Optional

import numpy as np

# --- Random seed kontrolü ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class VRP:
    def __init__(self, filename: Optional[str] = None):
        self.coords: List[Tuple[float, float]] = []
        self.demands: List[int] = []
        self.capacity: Optional[int] = None
        self.depot: int = 0
        self.D: Optional[np.ndarray] = None

        if filename:
            self.read_vrp_file(filename)

    # --- TSPLIB formatında VRP dosyasını okur ---
    def read_vrp_file(self, filename: str):
        coords = []
        demands = []
        capacity = None
        depot = None
        section = None

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line == "" or line.startswith("COMMENT"):
                    continue
                if line.startswith("CAPACITY"):
                    capacity = int(line.split(":")[1])
                elif line.startswith("NODE_COORD_SECTION"):
                    section = "NODE_COORD_SECTION"
                elif line.startswith("DEMAND_SECTION"):
                    section = "DEMAND_SECTION"
                elif line.startswith("DEPOT_SECTION"):
                    section = "DEPOT_SECTION"
                elif line.startswith("EOF"):
                    break
                else:
                    if section == "NODE_COORD_SECTION":
                        parts = line.split()
                        if len(parts) >= 3:
                            coords.append((float(parts[1]), float(parts[2])))
                    elif section == "DEMAND_SECTION":
                        parts = line.split()
                        if len(parts) >= 2:
                            demands.append(int(parts[1]))
                    elif section == "DEPOT_SECTION":
                        parts = line.split()
                        if "-1" in parts:
                            continue
                        # Eğer tek bir ID varsa, onu depot olarak kullan
                        if len(parts) == 1:
                            depot = int(parts[0]) - 1
                        else:
                            # Koordinatlar gelmişse 0. noktayı depot olarak kabul et
                            depot = 0

        # Eksik alanları kontrol et
        if depot is None:
            depot = 0
        if demands[depot] != 0:
            demands[depot] = 0

        # Sınıfa ata
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.depot = depot
        self.D = self.euclidean_matrix()

    # --- Öklidyen mesafe matrisi ---
    def euclidean_matrix(self) -> np.ndarray:
        n = len(self.coords)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                xi, yi = self.coords[i]
                xj, yj = self.coords[j]
                D[i, j] = math.hypot(xi - xj, yi - yj)
        return D

    # --- Permütasyondan rotaları oluştur ---
    def split_routes_from_permutation(self, perm: List[int]) -> List[List[int]]:
        routes = []
        current_route = []
        current_load = 0

        for c in perm:
            d = self.demands[c]
            if current_load + d <= self.capacity:
                current_route.append(c)
                current_load += d
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_load = d

        if current_route:
            routes.append(current_route)

        return routes

    # --- Bir rota setinin toplam mesafesini hesapla ---
    def total_distance_of_routes(self, routes: List[List[int]]) -> float:
        total = 0.0
        for r in routes:
            if len(r) == 0:
                continue
            prev = self.depot
            for cust in r:
                total += self.D[prev, cust]
                prev = cust
            total += self.D[prev, self.depot]
        return total

    # --- Permütasyonu değerlendir ---
    def evaluate_permutation(self, perm: List[int], penalty_coef=1000.0) -> Tuple[float, List[List[int]]]:
        routes = self.split_routes_from_permutation(perm)
        dist = self.total_distance_of_routes(routes)
        penalty = 0.0
        for r in routes:
            load = sum(self.demands[c] for c in r)
            if load > self.capacity:
                penalty += (load - self.capacity) * penalty_coef
        return dist + penalty, routes

    # --- Bilgi yazdırma ---
    def summary(self):
        print(f"VRP Instance Summary")
        print(f"- # of customers: {len(self.coords) - 1}")
        print(f"- Vehicle capacity: {self.capacity}")
        print(f"- Depot index: {self.depot}")
        print(f"- Distance matrix size: {self.D.shape}")
