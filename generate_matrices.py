import numpy as np

def generate_fault_matrix(mutants_log_path, killmap_log_path, num_tests):
    with open(mutants_log_path, "r") as f:
        mutant_ids = [line.strip() for line in f if line.strip()]
    mutant_index_map = {mutant_id: idx for idx, mutant_id in enumerate(mutant_ids)}
    num_mutants = len(mutant_ids)

    # Step 2: Build fault matrix
    fault_matrix = np.zeros((num_mutants, num_tests), dtype=np.int32)
    with open(killmap_log_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            mutant_id, test_id_str = parts
            test_id = int(test_id_str)
            if mutant_id in mutant_index_map:
                m_idx = mutant_index_map[mutant_id]
                fault_matrix[m_idx, test_id] = 1
    return fault_matrix

def generate_coverage_matrix(coverage_matrix_path):
    with open(coverage_matrix_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    coverage_matrix = np.array([[int(x) for x in line.split()] for line in lines], dtype=np.int32)
    return coverage_matrix

def main():
    mutants_log = "./siena/mutants.log"
    killmap_log = "./siena/killmap.log"
    coverage_path = "./siena/siena-coverage-matrix.txt"

    coverage_matrix = generate_coverage_matrix(coverage_path)
    num_tests = coverage_matrix.shape[0]

    fault_matrix = generate_fault_matrix(mutants_log, killmap_log, num_tests)

    np.save("./siena/MC-TOA/fault_matrix.npy", fault_matrix)
    np.save("./siena/MC-TOA/coverage_matrix.npy", coverage_matrix)
    print("Saved fault_matrix.npy and coverage_matrix.npy to ./siena/MC-TOA/")

if __name__ == "__main__":
    main()
