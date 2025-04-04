import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mc_toa import mc_toa_selection  # Assumes your MC-TOA code is in mc_toa.py

def calculate_fdl(full_fault_matrix, reduced_fault_matrix):
    full_detected = np.any(full_fault_matrix, axis=0)
    reduced_detected = np.any(reduced_fault_matrix, axis=0)
    missed_faults = np.logical_and(full_detected, np.logical_not(reduced_detected))
    fdl = missed_faults.sum() / full_detected.sum()
    return fdl

def main(fault_matrix_path, coverage_matrix_path):
    fault_matrix = np.load(fault_matrix_path)
    coverage_matrix = np.load(coverage_matrix_path)

    reduction_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    results = []

    for ratio in reduction_ratios:
        selected_indices = mc_toa_selection(coverage_matrix, ratio)
        reduced_fault_matrix = fault_matrix[selected_indices, :]
        fdl = calculate_fdl(fault_matrix, reduced_fault_matrix)
        results.append({"Reduction Ratio": int(ratio * 100), "FDL": fdl})
        print(f"Reduction {int(ratio * 100)}% -> FDL: {fdl:.4f}")

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv("fdl_results.csv", index=False)
    print("✅ FDL results saved to fdl_results.csv")

    # Generate plot
    plt.figure(figsize=(6, 4))
    plt.bar(df["Reduction Ratio"], df["FDL"], color="skyblue")
    plt.xlabel("Test Suite Reduction (%)")
    plt.ylabel("Fault Detection Loss (FDL)")
    plt.title("FDL vs. Test Suite Reduction")
    plt.xticks(df["Reduction Ratio"])
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("fdl_plot.png")
    print("📊 FDL plot saved to fdl_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault_matrix", required=True, help="Path to fault_matrix.npy")
    parser.add_argument("--coverage_matrix", required=True, help="Path to coverage_matrix.npy")
    args = parser.parse_args()

    main(args.fault_matrix, args.coverage_matrix)
