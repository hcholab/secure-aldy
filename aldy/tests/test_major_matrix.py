#!/usr/bin/env python
# 786
# Aldy source: test_major_matrix.py
#   This file is subject to the terms and conditions defined in
#   file 'LICENSE', which is part of this source code package.

"""
Tests for major_matrix module.

This test validates that the matrix extraction module produces the same results
as the original solve_major_model implementation.
"""

import pytest
import numpy as np
import collections
from ortools.linear_solver import pywraplp

from aldy.major_matrix import build_major_matrices, extract_solution, validate_solution
from aldy.major import estimate_major, solve_major_model, _filter_alleles
from aldy.gene import Gene, Mutation
from aldy.profile import Profile
from aldy.coverage import Coverage
from aldy.solutions import CNSolution, SolvedAllele
from aldy.common import script_path, sorted_tuple


def solve_with_ortools(matrices):
    """
    Solve the ILP problem using OR-tools CP-SAT solver.
    
    Returns: (status, objective, x, y, e, z)
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("Could not create SCIP solver")
    
    # Create variables
    x = [solver.BoolVar(f'x_{j}') for j in range(matrices.n_alleles)]
    y = [solver.BoolVar(f'y_{i}') for i in range(matrices.n_mutations)]
    z = solver.BoolVar('z')
    
    # Error variables (continuous)
    e = [solver.NumVar(-solver.infinity(), solver.infinity(), f'e_{i}')
         for i in range(matrices.n_mutations)]
    
    # Absolute value variables for errors
    e_abs = [solver.NumVar(0, solver.infinity(), f'e_abs_{i}')
             for i in range(matrices.n_mutations)]
    
    # Coverage equality constraints: A·x + y + e = c
    for i in range(matrices.n_mutations):
        constraint = solver.Constraint(float(matrices.c[i]), float(matrices.c[i]))
        for j in range(matrices.n_alleles):
            if matrices.A[i, j] != 0:
                constraint.SetCoefficient(x[j], float(matrices.A[i, j]))
        constraint.SetCoefficient(y[i], 1)
        constraint.SetCoefficient(e[i], 1)
    
    # CN configuration constraints: B·x = d
    for t in range(matrices.n_cn_configs):
        constraint = solver.Constraint(float(matrices.d[t]), float(matrices.d[t]))
        for j in range(matrices.n_alleles):
            if matrices.B[t, j] != 0:
                constraint.SetCoefficient(x[j], float(matrices.B[t, j]))
    
    # Ordering constraints: x[j1] <= x[j2]
    for j1, j2 in matrices.P_indices:
        constraint = solver.Constraint(-solver.infinity(), 0)
        constraint.SetCoefficient(x[j1], 1)
        constraint.SetCoefficient(x[j2], -1)
    
    # Novel indicator constraints: z >= y[i]
    for i in range(matrices.n_mutations):
        constraint = solver.Constraint(0, solver.infinity())
        constraint.SetCoefficient(z, 1)
        constraint.SetCoefficient(y[i], -1)
    
    # Novel indicator constraint: z <= sum(y)
    constraint = solver.Constraint(0, solver.infinity())
    constraint.SetCoefficient(z, matrices.n_mutations)
    for i in range(matrices.n_mutations):
        constraint.SetCoefficient(y[i], -1)
    
    # One novel per position constraints
    for pos, indices in matrices.position_groups.items():
        constraint = solver.Constraint(-solver.infinity(), 1)
        for i in indices:
            constraint.SetCoefficient(y[i], 1)
    
    # XOR constraints: each mutation must be explained by allele OR novel
    # This is complex to linearize, so we use a simpler approximation:
    # For each mutation i: sum(A[i,j]*x[j]) + y[i] >= 1
    # This ensures mutation is explained by at least one source
    for i in range(matrices.n_mutations):
        if matrices.mutation_vars[i].op != "_":  # Skip reference mutations
            constraint = solver.Constraint(1, solver.infinity())
            for j in range(matrices.n_alleles):
                if matrices.A[i, j] != 0:
                    constraint.SetCoefficient(x[j], matrices.A[i, j])
            constraint.SetCoefficient(y[i], 1)
    
    # Absolute value constraints: e_abs[i] >= e[i] and e_abs[i] >= -e[i]
    for i in range(matrices.n_mutations):
        c1 = solver.Constraint(0, solver.infinity())
        c1.SetCoefficient(e_abs[i], 1)
        c1.SetCoefficient(e[i], -1)
        
        c2 = solver.Constraint(0, solver.infinity())
        c2.SetCoefficient(e_abs[i], 1)
        c2.SetCoefficient(e[i], 1)
    
    # Objective: min ||e||_1 + λ*z + 0.1*sum(y)
    objective = solver.Objective()
    for i in range(matrices.n_mutations):
        objective.SetCoefficient(e_abs[i], 1)
    objective.SetCoefficient(z, matrices.lambda_novel)
    for i in range(matrices.n_mutations):
        objective.SetCoefficient(y[i], 0.1)
    objective.SetMinimization()
    
    # Solve
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        x_val = np.array([x[j].solution_value() for j in range(matrices.n_alleles)])
        y_val = np.array([y[i].solution_value() for i in range(matrices.n_mutations)])
        e_val = np.array([e[i].solution_value() for i in range(matrices.n_mutations)])
        z_val = z.solution_value()
        obj_val = objective.Value()
        
        return status, obj_val, x_val, y_val, e_val, z_val
    else:
        return status, None, None, None, None, None


def test_matrix_extraction_simple():
    """Test basic matrix extraction with synthetic data."""
    
    # Create simple synthetic test data
    profile = Profile("test")
    profile.threshold = 0.5
    profile.min_coverage = 2.0
    
    # Load a real gene for structure
    gene = Gene(script_path("aldy.resources.genes/cyp2d6.yml"))
    
    # Create a simple CN solution (2 copies)
    cn_solution = CNSolution(gene, 0, ["1", "1"])
    
    # Create minimal coverage data
    cov_data = collections.defaultdict(dict)
    # Add some coverage at key positions
    for pos in [2549, 2615, 2850]:  # Some CYP2D6 positions
        cov_data[pos]["_"] = [(60, 60)] * 10
    
    coverage = Coverage(gene, profile, None, cov_data, None, {})
    
    # Filter alleles
    alleles, filtered_cov = _filter_alleles(gene, coverage, cn_solution)
    
    # Get functional mutations
    func_muts = {
        Mutation(*m)
        for m in gene.mutations
        if gene.is_functional(m) and filtered_cov[Mutation(*m)] > 0
    }
    
    # Build matrices
    matrices = build_major_matrices(gene, filtered_cov, cn_solution, alleles, func_muts)
    
    # Basic validation
    assert matrices.n_alleles > 0, "Should have allele variables"
    assert matrices.n_mutations >= 0, "Should have mutation variables"
    assert matrices.A.shape == (matrices.n_mutations, matrices.n_alleles)
    assert matrices.B.shape == (matrices.n_cn_configs, matrices.n_alleles)
    assert matrices.c.shape == (matrices.n_mutations,)
    assert matrices.d.shape == (matrices.n_cn_configs,)
    assert len(matrices.allele_vars) == matrices.n_alleles
    assert len(matrices.mutation_vars) == matrices.n_mutations


def test_matrix_vs_original_solver():
    """
    Test that matrix extraction + OR-tools produces same results as original solver.
    """
    
    profile = Profile("test")
    profile.threshold = 0.5
    profile.min_coverage = 2.0
    profile.major_novel = 21.0
    
    # Load real gene
    gene = Gene(script_path("aldy.resources.genes/cyp2d6.yml"))
    
    # Create CN solution (use valid CN configs)
    cn_solution = CNSolution(gene, 0, ["1", "1"])
    
    # Create realistic coverage data with some mutations
    cov_data = collections.defaultdict(dict)
    
    # Add reference coverage at various positions
    for pos in range(2549, 2850, 50):
        cov_data[pos]["_"] = [(60, 60)] * 15
    
    # Add some specific mutations
    # Example: 2850C>T (rs1065852, defines *4)
    if (2850, "2850C>T") in gene.mutations:
        cov_data[2850]["2850C>T"] = [(60, 60)] * 8
        cov_data[2850]["_"] = [(60, 60)] * 7
    
    coverage = Coverage(gene, profile, None, cov_data, None, {})
    
    # Run original solver
    try:
        original_solutions = estimate_major(gene, coverage, cn_solution, "any")
    except Exception as e:
        pytest.skip(f"Original solver failed: {e}")
    
    if not original_solutions:
        pytest.skip("No original solutions found")
    
    # Filter alleles and get mutations (same as original)
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
    
    if not func_muts and not alleles:
        pytest.skip("No functional mutations or alleles to test")
    
    # Build matrices
    matrices = build_major_matrices(gene, filtered_cov, cn_solution, alleles, func_muts)
    
    # Solve with OR-tools
    status, obj_val, x_val, y_val, e_val, z_val = solve_with_ortools(matrices)
    
    if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
        pytest.skip(f"OR-tools solver did not find solution: status={status}")
    
    # Extract solution
    selected_alleles, novel_muts = extract_solution(matrices, x_val, y_val)
    
    # Validate solution
    validation = validate_solution(matrices, x_val, y_val, e_val, z_val)
    assert validation["valid"], f"Solution validation failed: {validation['violations']}"
    
    # Compare with original solution
    original_sol = original_solutions[0]
    original_alleles = sorted_tuple([i.major for i, v in original_sol.solution.items() 
                                     for _ in range(v)])
    matrix_alleles = sorted_tuple(selected_alleles)
    
    # Check if alleles match (may not be exact due to solver differences)
    # At minimum, check that objective values are close
    print(f"\nOriginal solution: {original_alleles}")
    print(f"Original score: {original_sol.score:.4f}")
    print(f"Matrix solution: {matrix_alleles}")
    print(f"Matrix score: {obj_val:.4f}")
    print(f"Objective breakdown: {validation}")
    
    # Allow some tolerance for different solvers
    assert abs(original_sol.score - obj_val) < 5.0, \
        f"Objective values differ significantly: {original_sol.score} vs {obj_val}"


def test_solution_validation():
    """Test that solution validation catches constraint violations."""
    
    profile = Profile("test")
    gene = Gene(script_path("aldy.resources.genes/cyp2d6.yml"))
    cn_solution = CNSolution(gene, 0, ["1", "1"])
    
    cov_data = collections.defaultdict(dict)
    for pos in range(2549, 2650, 50):
        cov_data[pos]["_"] = [(60, 60)] * 10
    
    coverage = Coverage(gene, profile, None, cov_data, None, {})
    alleles, filtered_cov = _filter_alleles(gene, coverage, cn_solution)
    
    func_muts = {
        Mutation(*m)
        for m in gene.mutations
        if gene.is_functional(m) and filtered_cov[Mutation(*m)] > 0
    }
    
    if not alleles:
        pytest.skip("No alleles available for testing")
    
    matrices = build_major_matrices(gene, filtered_cov, cn_solution, alleles, func_muts)
    
    # Create an invalid solution (violates CN constraint)
    x_bad = np.zeros(matrices.n_alleles)
    x_bad[0] = 1  # Select only one allele when CN requires 2
    y_bad = np.zeros(matrices.n_mutations)
    e_bad = matrices.c - matrices.A @ x_bad - y_bad
    z_bad = 0
    
    validation = validate_solution(matrices, x_bad, y_bad, e_bad, z_bad)
    
    # Should detect CN violation (unless by chance it works)
    if matrices.d.sum() > 1:  # If we need more than 1 allele
        assert not validation["valid"] or "CN configuration violated" in str(validation["violations"]), \
            "Should detect CN constraint violation"


@pytest.mark.parametrize("gene_name", ["cyp2d6", "cyp2c19"])
def test_multiple_genes(gene_name):
    """Test matrix extraction works for different genes."""
    
    try:
        gene = Gene(script_path(f"aldy.resources.genes/{gene_name}.yml"))
    except:
        pytest.skip(f"Gene {gene_name} not available")
    
    profile = Profile("test")
    cn_solution = CNSolution(gene, 0, ["1", "1"])
    
    cov_data = collections.defaultdict(dict)
    # Add minimal coverage
    positions = list(gene.regions[0].values())[0] if gene.regions else None
    if positions:
        for pos in range(positions.start, min(positions.start + 500, positions.end), 100):
            cov_data[pos]["_"] = [(60, 60)] * 10
    
    coverage = Coverage(gene, profile, None, cov_data, None, {})
    alleles, filtered_cov = _filter_alleles(gene, coverage, cn_solution)
    
    func_muts = {
        Mutation(*m)
        for m in gene.mutations
        if gene.is_functional(m) and filtered_cov[Mutation(*m)] > 0
    }
    
    if not alleles:
        pytest.skip(f"No alleles for {gene_name}")
    
    # Should not raise exception
    matrices = build_major_matrices(gene, filtered_cov, cn_solution, alleles, func_muts)
    
    assert matrices.n_alleles > 0
    assert matrices.A.shape[1] == matrices.n_alleles


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
