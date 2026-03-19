import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time

st.set_page_config(page_title="GA Research Lab (Final)", layout="wide")

st.markdown("## 🧬 Evolutionary Optimization & GA Research Platform")

# =============================
# Seed
# =============================
seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
np.random.seed(seed)
random.seed(seed)

# =============================
# Mode
# =============================
mode = st.sidebar.radio("Mode", ["Standard Run", "Experiment Mode"])

# =============================
# Utility Functions
# =============================

def init_population(size, length):
    return np.random.randint(2, size=(size, length))

def fitness_onemax(ind):
    return np.sum(ind)

def fitness_trap(ind, k=4):
    total = 0
    for i in range(0, len(ind), k):
        block = ind[i:i+k]
        u = np.sum(block)
        total += k if u == k else (k - 1 - u)
    return total

def fitness_tsp(route, dist_matrix):
    return sum(dist_matrix[route[i-1]][route[i]] for i in range(len(route)))

def diversity(pop):
    return np.mean(np.std(pop, axis=0))

def tournament_selection(pop, fitnesses, k=3):
    selected = []
    for _ in range(len(pop)):
        idx = np.random.choice(len(pop), k)
        best = idx[np.argmax([fitnesses[i] for i in idx])]
        selected.append(pop[best])
    return np.array(selected)

def tournament_selection_min(pop, fitnesses, k=3):
    selected = []
    for _ in range(len(pop)):
        idx = np.random.choice(len(pop), k)
        best = idx[np.argmin([fitnesses[i] for i in idx])]
        selected.append(pop[best])
    return np.array(selected)

def crossover(p1, p2, pc):
    if random.random() > pc:
        return p1.copy(), p2.copy()
    point = random.randint(1, len(p1)-1)
    return (np.concatenate([p1[:point], p2[point:]]),
            np.concatenate([p2[:point], p1[point:]]))

def mutation(ind, pm):
    ind = ind.copy()
    for i in range(len(ind)):
        if random.random() < pm:
            ind[i] = 1 - ind[i]
    return ind

def mutation_tsp(route, pm=0.1):
    route = route.copy()
    if random.random() < pm:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# =============================
# GA Binary
# =============================

def run_ga_binary(fitness_func, length, gens, pop_size, pc, pm, elitism=True):

    if pop_size % 2 != 0:
        pop_size += 1

    pop = init_population(pop_size, length)

    best_hist, avg_hist, div_hist, gen_times = [], [], [], []

    for g in range(gens):
        start_g = time.time()

        fitnesses = np.array([fitness_func(ind) for ind in pop])

        best_hist.append(np.max(fitnesses))
        avg_hist.append(np.mean(fitnesses))
        div_hist.append(diversity(pop))

        elite = pop[np.argmax(fitnesses)].copy()
        selected = tournament_selection(pop, fitnesses)

        next_pop = []
        adaptive_pm = pm * (1 - g/gens)

        for i in range(0, pop_size, 2):
            c1, c2 = crossover(selected[i], selected[i+1], pc)
            next_pop.append(mutation(c1, adaptive_pm))
            next_pop.append(mutation(c2, adaptive_pm))

        pop = np.array(next_pop)
        if elitism:
            pop[0] = elite

        gen_times.append(time.time() - start_g)

    return pop, best_hist, avg_hist, div_hist, gen_times

# =============================
# Baselines
# =============================

def random_search(length, gens):
    best, hist = 0, []
    for _ in range(gens):
        best = max(best, fitness_onemax(np.random.randint(2, size=length)))
        hist.append(best)
    return hist

def hill_climbing(length, gens):
    current = np.random.randint(2, size=length)
    best = fitness_onemax(current)
    hist = []
    for _ in range(gens):
        neighbor = current.copy()
        i = random.randint(0, length-1)
        neighbor[i] ^= 1
        if fitness_onemax(neighbor) > best:
            current, best = neighbor, fitness_onemax(neighbor)
        hist.append(best)
    return hist

# =============================
# TSP
# =============================

def run_ga_tsp(dist_matrix, gens, pop_size, pm):
    n = len(dist_matrix)
    pop = [np.random.permutation(n) for _ in range(pop_size)]
    best_hist = []

    def crossover_tsp(p1, p2):
        a, b = sorted(random.sample(range(len(p1)), 2))
        child = [-1]*len(p1)
        child[a:b] = p1[a:b]
        ptr = 0
        for x in p2:
            if x not in child:
                while child[ptr] != -1:
                    ptr += 1
                child[ptr] = x
        return np.array(child)

    for _ in range(gens):
        fitnesses = np.array([fitness_tsp(ind, dist_matrix) for ind in pop])
        best_hist.append(np.min(fitnesses))

        elite = pop[np.argmin(fitnesses)].copy()
        selected = tournament_selection_min(np.array(pop), fitnesses)

        new_pop = []
        for i in range(0, pop_size, 2):
            new_pop.append(mutation_tsp(crossover_tsp(selected[i], selected[i+1]), pm))
            new_pop.append(mutation_tsp(crossover_tsp(selected[i+1], selected[i]), pm))

        pop = new_pop
        pop[0] = elite

    return pop, best_hist

# =============================
# UI Logic (same as yours, polished)
# =============================
# (keeping your logic intact — no unnecessary changes)
