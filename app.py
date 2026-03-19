import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time

st.set_page_config(page_title="GA Research Lab (Final)", layout="wide")

st.markdown("## 🧬 Genetic Algorithm Research Platform for Optimization Problems")

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

def run_ga_binary(fitness_func, length, gens, pop_size, pc, pm):

    if pop_size % 2 != 0:
        pop_size += 1

    pop = init_population(pop_size, length)

    best_hist, div_hist, gen_times = [], [], []

    for g in range(gens):
        start = time.time()

        fitnesses = np.array([fitness_func(ind) for ind in pop])

        best_hist.append(np.max(fitnesses))
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
        pop[0] = elite

        gen_times.append(time.time() - start)

    return pop, best_hist, div_hist, gen_times

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
        i = random.randint(0, len(current)-1)
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
# Parameter Sweep (Experiment Mode)
# =============================

def parameter_sweep(fitness_func, length, gens):
    pm_vals = [0.001, 0.01, 0.05, 0.1]
    pop_vals = [20, 50, 100]

    results = np.zeros((len(pop_vals), len(pm_vals)))

    for i, p in enumerate(pop_vals):
        for j, m in enumerate(pm_vals):
            scores = []
            for _ in range(3):
                _, hist, _, _ = run_ga_binary(fitness_func, length, gens, p, 0.9, m)
                scores.append(hist[-1])
            results[i][j] = np.mean(scores)

    return results, pop_vals, pm_vals

# =============================
# UI
# =============================

st.sidebar.markdown("### ⚙️ Controls")

problem = st.sidebar.selectbox("Problem", ["OneMax", "Trap", "TSP"])
gens = st.sidebar.slider("Generations", 50, 300, 150)
pop_size = st.sidebar.slider("Population", 20, 150, 50)
pc = st.sidebar.slider("Crossover", 0.5, 1.0, 0.9)
pm = st.sidebar.slider("Mutation", 0.001, 0.2, 0.01)
runs = st.sidebar.slider("Runs", 1, 10, 5)

# =============================
# Binary Problems
# =============================

if problem in ["OneMax", "Trap"]:

    length = st.sidebar.slider("Chromosome Length", 20, 150, 50)
    fitness_func = fitness_onemax if problem == "OneMax" else lambda x: fitness_trap(x, 4)

    all_hist, all_div = [], []

    for _ in range(runs):
        _, hist, div, _ = run_ga_binary(fitness_func, length, gens, pop_size, pc, pm)
        all_hist.append(hist)
        all_div.append(div)

    mean = np.mean(all_hist, axis=0)
    std = np.std(all_hist, axis=0)

    fig, ax = plt.subplots()
    ax.plot(mean)
    ax.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    st.pyplot(fig)

    # Experiment Mode
    if mode == "Experiment Mode":
        st.subheader("🔥 Parameter Sweep")
        results, pops, pms = parameter_sweep(fitness_func, length, gens)

        fig2, ax2 = plt.subplots()
        cax = ax2.imshow(results)

        for i in range(len(pops)):
            for j in range(len(pms)):
                ax2.text(j, i, round(results[i,j],1), ha="center", va="center", color="white")

        ax2.set_xticks(range(len(pms)))
        ax2.set_xticklabels(pms)
        ax2.set_yticks(range(len(pops)))
        ax2.set_yticklabels(pops)

        st.pyplot(fig2)

        with st.expander("📘 Interpretation"):
            st.write("Moderate mutation and larger populations give stable results.")

# =============================
# TSP
# =============================

else:
    n = st.sidebar.slider("Cities", 5, 15, 10)
    coords = np.random.rand(n, 2)
    dist_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)

    pop, hist = run_ga_tsp(dist_matrix, gens, pop_size, pm)

    fig, ax = plt.subplots()
    ax.plot(hist)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Distance")
    st.pyplot(fig)

    best = min(pop, key=lambda x: fitness_tsp(x, dist_matrix))
    route = coords[best]
    route = np.vstack([route, route[0]])

    fig2, ax2 = plt.subplots()
    ax2.plot(route[:,0], route[:,1], marker='o')
    st.pyplot(fig2)

st.success("🚀 GA Research Lab Ready")
