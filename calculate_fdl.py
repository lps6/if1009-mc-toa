import numpy as np

def calculate_fdl(fault_matrix, selected_indices):
    """
    Calculate the Fault Detection Loss (FDL) for a reduced test suite.

    Parameters:
    - fault_matrix (numpy.ndarray): Binary matrix [tests x faults]
    - selected_indices (list or numpy.ndarray): Indices of selected tests in reduced suite

    Returns:
    - fdl (float): Fault Detection Loss in percentage
    - fd_original (int): Number of faults detected by full suite
    - fd_reduced (int): Number of faults detected by reduced suite
    """
    # Faults detected by the full test suite
    fd_original = np.sum(np.any(fault_matrix, axis=0))

    # Faults detected by the reduced suite
    reduced_matrix = fault_matrix[selected_indices, :]
    fd_reduced = np.sum(np.any(reduced_matrix, axis=0))

    # Compute FDL
    fdl = ((fd_original - fd_reduced) / fd_original) * 100 if fd_original > 0 else 0.0

    return fdl, fd_original, fd_reduced
