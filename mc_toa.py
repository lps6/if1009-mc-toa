import numpy as np

def mc_toa_selection(coverage_matrix, reduction_ratio):
    num_tests = coverage_matrix.shape[0]
    target_size = max(1, int(num_tests * reduction_ratio))

    # Greedy coverage maximization escolhida como parametro pro projeto
    selected = []
    remaining = list(range(num_tests))
    covered = np.zeros(coverage_matrix.shape[1], dtype=bool)

    while len(selected) < target_size and remaining:
        best_test = None
        best_gain = -1
        for i in remaining:
            new_coverage = np.logical_or(covered, coverage_matrix[i])
            gain = np.sum(new_coverage) - np.sum(covered)
            if gain > best_gain:
                best_gain = gain
                best_test = i

        if best_test is None:
            break
        selected.append(best_test)
        covered = np.logical_or(covered, coverage_matrix[best_test])
        remaining.remove(best_test)

    return selected
