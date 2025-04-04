
import numpy as np
from pathlib import Path

def parse_killmap(killmap_log):
    """Parses killmap.log and returns a binary fault matrix."""
    with open(killmap_log, 'r') as f:
        lines = f.readlines()

    tests = set()
    mutants = set()
    kill_matrix = {}

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        test, mutant = parts
        tests.add(test)
        mutants.add(mutant)
        kill_matrix.setdefault(test, set()).add(mutant)

    sorted_tests = sorted(tests)
    sorted_mutants = sorted(mutants)

    matrix = np.zeros((len(sorted_tests), len(sorted_mutants)), dtype=int)
    for i, test in enumerate(sorted_tests):
        for j, mutant in enumerate(sorted_mutants):
            if mutant in kill_matrix.get(test, set()):
                matrix[i][j] = 1

    print(f"Fault matrix shape: {matrix.shape}")
    return matrix

def parse_coverage_matrix(coverage_path):
    """Parses the SIR coverage matrix (lines covered per test) and returns a binary coverage matrix."""
    with open(coverage_path, 'r') as f:
        lines = f.readlines()

    coverage_data = []
    for line in lines:
        row = [int(x) for x in line.strip().split()]
        coverage_data.append(row)

    matrix = np.array(coverage_data, dtype=int)
    print(f"✅ Coverage matrix shape: {matrix.shape}")
    return matrix

def main():
    killmap_log = "/Users/lucaspires/Downloads/TestProject/siena/killmap.log"
    coverage_path = "/Users/lucaspires/Downloads/TestProject/siena/siena-coverage-matrix.txt"
    output_dir = Path("./siena/MC-TOA")

    output_dir.mkdir(parents=True, exist_ok=True)

    fault_matrix = parse_killmap(killmap_log)
    np.save(output_dir / "fault_matrix.npy", fault_matrix)

    coverage_matrix = parse_coverage_matrix(coverage_path)
    np.save(output_dir / "coverage_matrix.npy", coverage_matrix)

    print("Matrices saved to:", output_dir)

if __name__ == "__main__":
    main()
