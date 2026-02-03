#!/usr/bin/env python
"""
Example usage of the major_matrix module with custom solvers.

This demonstrates how to extract the ILP problem in matrix form
and solve it with your own custom solvers or heuristics.
"""

import numpy as np
from collections import defaultdict

from aldy.major_matrix import build_major_matrices, extract_solution, validate_solution
from aldy.major import _filter_alleles
from aldy.gene import Gene, Mutation
from aldy.profile import Profile
from aldy.coverage import Coverage
from aldy.solutions import CNSolution
from aldy.common import script_path


def greedy_heuristic(matrices):
    """
    Example custom heuristic solver: Greedy selection based on coverage fit.
    
    This is a simple demonstration - not optimal, but shows how to use the matrices.
    """
    print(f"\n=== Greedy Heuristic Solver ===")
    print(f"Problem size: {matrices.n_alleles} alleles, {matrices.n_mutations} mutations")
    
    # Initialize solution
    x = np.zeros(matrices.n_alleles)
    y = np.zeros(matrices.n_mutations)
    z = 0
    
    # Greedy: Select alleles to satisfy CN constraints
    for t in range(matrices.n_cn_configs):
        required = int(matrices.d[t])
        candidates = [j for j in range(matrices.n_alleles) if matrices.B[t, j] == 1]
        
        # Select first 'required' candidates (ordered by index)
        for i, j in enumerate(candidates[:required]):
            x[j] = 1
    
    # Compute error vector
    e = matrices.c - matrices.A @ x - y
    
    # Check if any mutations are not explained (could mark as novel)
    for i in range(matrices.n_mutations):
        contribution = sum(matrices.A[i, j] * x[j] for j in range(matrices.n_alleles))
        if contribution < 0.5 and matrices.mutation_vars[i].op != "_":
            y[i] = 1
            z = 1
    
    # Recompute error after marking novels
    e = matrices.c - matrices.A @ x - y
    
    # Validate and return
    validation = validate_solution(matrices, x, y, e, z)
    
    print(f"Solution valid: {validation['valid']}")
    print(f"Objective: {validation['objective']:.4f}")
    print(f"  Coverage fit: {validation['coverage_fit']:.4f}")
    print(f"  Novel penalty: {validation['novel_penalty']:.4f}")
    
    if not validation['valid']:
        print(f"Violations: {validation['violations']}")
    
    return x, y, e, z, validation


def main():
    """Example: Load data, extract matrices, solve with custom heuristic."""
    
    # Setup
    print("Loading gene and preparing data...")
    profile = Profile("test")
    profile.threshold = 0.5
    profile.min_coverage = 2.0
    profile.major_novel = 21.0
    
    gene = Gene(script_path("aldy.resources.genes/cyp2d6.yml"))
    cn_solution = CNSolution(gene, 0, ["1", "1"])
    
    # Create coverage data
    cov_data = defaultdict(dict)
    for pos in range(2549, 2850, 50):
        cov_data[pos]["_"] = [(60, 60)] * 15
    
    # Add a specific mutation
    if (2850, "2850C>T") in gene.mutations:
        cov_data[2850]["2850C>T"] = [(60, 60)] * 8
        cov_data[2850]["_"] = [(60, 60)] * 7
    
    coverage = Coverage(gene, profile, None, cov_data, None, {})
    
    # Filter alleles and get mutations (same as Aldy does)
    alleles, filtered_cov = _filter_alleles(gene, coverage, cn_solution)
    
    func_muts = {
        Mutation(*m)
        for m in gene.mutations
        if gene.is_functional(m) and filtered_cov[Mutation(*m)] > 0
    }
    func_muts |= {
        m
        for m in gene.random_mutations
        if gene.is_functional((m.pos, m.op)) and filtered_cov[m] > 0
    }
    
    print(f"Found {len(alleles)} candidate alleles")
    print(f"Found {len(func_muts)} functional mutations")
    
    # Extract matrices
    print("\nExtracting ILP matrices...")
    matrices = build_major_matrices(gene, filtered_cov, cn_solution, alleles, func_muts)
    
    print(f"\nMatrix dimensions:")
    print(f"  A: {matrices.A.shape} (mutation × allele incidence)")
    print(f"  B: {matrices.B.shape} (CN config × allele)")
    print(f"  c: {matrices.c.shape} (coverage vector)")
    print(f"  d: {matrices.d.shape} (CN requirements)")
    print(f"  Ordering constraints: {len(matrices.P_indices)}")
    print(f"  Position groups: {len(matrices.position_groups)}")
    
    # Display some matrix details
    print(f"\nAllele variables (first 5):")
    for i, (name, idx) in enumerate(matrices.allele_vars[:5]):
        print(f"  x[{i}] = *{name} copy {idx}")
    
    print(f"\nMutation variables (first 5):")
    for i, mut in enumerate(matrices.mutation_vars[:5]):
        print(f"  y[{i}] = {mut}")
    
    print(f"\nCN configuration requirements:")
    for i, cfg in enumerate(matrices.cn_configs):
        print(f"  {cfg}: {matrices.d[i]} copies required")
    
    # Solve with custom heuristic
    print("\n" + "="*60)
    x, y, e, z, validation = greedy_heuristic(matrices)
    
    # Extract solution
    selected_alleles, novel_muts = extract_solution(matrices, x, y)
    
    print(f"\n=== Solution ===")
    print(f"Selected alleles: {selected_alleles}")
    print(f"Novel mutations: {novel_muts}")
    
    print("\n" + "="*60)
    print("You can now use these matrices with any ILP solver:")
    print("  - CPLEX")
    print("  - Gurobi")
    print("  - SCIP")
    print("  - Custom branch-and-bound")
    print("  - Genetic algorithms")
    print("  - Simulated annealing")
    print("  - etc.")
    
    return matrices, x, y, e, z


if __name__ == "__main__":
    matrices, x, y, e, z = main()
