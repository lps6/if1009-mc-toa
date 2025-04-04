import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random

# Load fault matrix and coverage matrix
def load_matrices(fault_matrix_path, coverage_matrix_path):
    fault_matrix = pd.read_csv(fault_matrix_path, index_col=0).astype(int)
    coverage_matrix = pd.read_csv(coverage_matrix_path, index_col=0).astype(int)
    return fault_matrix, coverage_matrix

# Calculate early fault detection score (higher is better)
def early_fault_score(test_subset, fault_matrix):
    subset = fault_matrix.iloc[test_subset]
    detection_order = subset.any(axis=0).idxmax(axis=0)
    score = 0
    for fault in fault_matrix.columns:
        for i, test_id in enumerate(test_subset):
            if fault_matrix.at[test_id, fault] == 1:
                score += len(test_subset) - i  # reward earlier detection
                break
    return score

# Objective function
def evaluate(individual, fault_matrix, coverage_matrix):
    selected = [i for i, bit in enumerate(individual) if bit]
    if len(selected) == 0:
        return (0, 0, 0, 0)
    
    faults_detected = fault_matrix.iloc[selected].any(axis=0).sum()
    coverage_achieved = coverage_matrix.iloc[selected].sum().sum()
    early_score = early_fault_score(selected, fault_matrix)
    size = len(selected)
    return (faults_detected, coverage_achieved, early_score, -size)  # Maximize all but size

# Create optimization setup
def run_mc_toa(fault_matrix, coverage_matrix, reduction_ratio=0.2, ngen=100, pop_size=100):
    num_tests = fault_matrix.shape[0]
    max_tests = int(num_tests * reduction_ratio)

    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: random.random() < reduction_ratio)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_tests)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def feasible(ind): return sum(ind) <= max_tests
    def distance(ind): return abs(sum(ind) - max_tests)

    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0.0, distance))
    toolbox.register("evaluate", evaluate, fault_matrix=fault_matrix, coverage_matrix=coverage_matrix)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    pop, _ = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=0.7, mutpb=0.2, ngen=ngen, verbose=True)

    pareto = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    best = max(pareto, key=lambda ind: ind.fitness.values[0])  # max fault detection
    selected_tests = [i for i, b in enumerate(best) if b]

    return selected_tests, best.fitness.values

