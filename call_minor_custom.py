#!/usr/bin/env python
"""
Run minor allele calling using matrix extraction + custom solver.

Usage:
    python call_minor_custom.py [BAM_FILE GENE PROFILE SOLVER]

Solvers:
    ortools    - OR-Tools SCIP (exact ILP solver)
    sa         - Simulated Annealing (heuristic)
    hc         - Hill Climbing (local search)
    abc        - Artificial Bee Colony (swarm intelligence)
    all        - Compare all available solvers

Example:
    python call_minor_custom.py ../data/NA07000.bam CYP2D6 illumina ortools
    python call_minor_custom.py ../data/NA07000.bam CYP2D6 illumina all
"""

import sys
import time
from typing import List, Set

import numpy as np
from ortools.linear_solver import pywraplp

from aldy.gene import Gene, Mutation
from aldy.profile import Profile
from aldy.coverage import Coverage
from aldy.solutions import CNSolution, MajorSolution, SolvedAllele
from aldy.common import script_path
from aldy.sam import Sample
from aldy.major import estimate_major
from aldy.minor import _print_candidates
from aldy.minor_matrix import build_minor_matrices


def solve_minor_matrices_with_ortools(matrices):
    """Solve minor matrices with OR-Tools SCIP."""
    start_time = time.time()
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("Could not create SCIP solver")

    vars = []
    for i, (name, vtype) in enumerate(zip(matrices.var_names, matrices.var_types)):
        lb = matrices.var_lb[i]
        ub = matrices.var_ub[i]
        if vtype == "B":
            v = solver.BoolVar(name)
        else:
            v = solver.NumVar(lb if lb != float("-inf") else -solver.infinity(),
                              ub if ub != float("inf") else solver.infinity(),
                              name)
        vars.append(v)

    # Equality constraints
    for coeffs, rhs in zip(matrices.A_eq, matrices.b_eq):
        ct = solver.Constraint(rhs, rhs)
        for idx, coef in coeffs.items():
            ct.SetCoefficient(vars[idx], float(coef))

    # Inequality constraints
    for coeffs, rhs in zip(matrices.A_ineq, matrices.b_ineq):
        ct = solver.Constraint(-solver.infinity(), rhs)
        for idx, coef in coeffs.items():
            ct.SetCoefficient(vars[idx], float(coef))

    # Objective
    objective = solver.Objective()
    for idx, coef in matrices.objective.items():
        objective.SetCoefficient(vars[idx], float(coef))
    objective.SetMinimization()

    status = solver.Solve()
    runtime = time.time() - start_time

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None, None, runtime

    values = np.array([v.solution_value() for v in vars])
    return objective.Value(), values, runtime


def _compute_objective_minor(matrices, values):
    """Compute objective value for given solution."""
    obj = 0.0
    for idx, coef in matrices.objective.items():
        obj += coef * values[idx]
    return obj


def _is_feasible_minor(matrices, values):
    """Check if solution satisfies all constraints."""
    # Check equality constraints
    for coeffs, rhs in zip(matrices.A_eq, matrices.b_eq):
        lhs = sum(coeffs.get(idx, 0) * values[idx] for idx in coeffs)
        if abs(lhs - rhs) > 1e-6:
            return False
    
    # Check inequality constraints
    for coeffs, rhs in zip(matrices.A_ineq, matrices.b_ineq):
        lhs = sum(coeffs.get(idx, 0) * values[idx] for idx in coeffs)
        if lhs > rhs + 1e-6:
            return False
    
    return True


def solve_minor_matrices_with_simulated_annealing(matrices, max_iterations=5000, initial_temp=10.0, cooling_rate=0.99):
    """Solve minor matrices with Simulated Annealing heuristic."""
    start_time = time.time()
    
    n_vars = len(matrices.var_names)
    
    def generate_solution():
        """Generate random feasible solution."""
        values = np.zeros(n_vars)
        for i in range(n_vars):
            if matrices.var_types[i] == "B":
                values[i] = np.random.choice([0, 1])
            else:
                values[i] = matrices.var_lb[i]
        return values
    
    def neighbor(values):
        """Generate neighbor solution."""
        values_new = values.copy()
        # Flip a random binary variable or perturb continuous
        idx = np.random.randint(n_vars)
        if matrices.var_types[idx] == "B":
            values_new[idx] = 1 - values_new[idx]
        else:
            values_new[idx] = np.clip(
                values_new[idx] + np.random.randn() * 0.1,
                matrices.var_lb[idx],
                matrices.var_ub[idx]
            )
        return values_new
    
    # Initialize
    values_current = generate_solution()
    obj_current = _compute_objective_minor(matrices, values_current)
    
    values_best = values_current.copy()
    obj_best = obj_current
    
    temperature = initial_temp
    accepted = 0
    improved = 0
    
    for iteration in range(max_iterations):
        values_new = neighbor(values_current)
        
        if not _is_feasible_minor(matrices, values_new):
            continue
        
        obj_new = _compute_objective_minor(matrices, values_new)
        delta = obj_new - obj_current
        
        if delta < 0 or np.random.random() < np.exp(-delta / temperature):
            values_current = values_new
            obj_current = obj_new
            accepted += 1
            
            if obj_new < obj_best:
                values_best = values_new.copy()
                obj_best = obj_new
                improved += 1
        
        temperature *= cooling_rate
    
    runtime = time.time() - start_time
    return obj_best, values_best, runtime


def solve_minor_matrices_with_hill_climbing(matrices, max_iterations=3000, restarts=5):
    """Solve minor matrices with Hill Climbing heuristic."""
    start_time = time.time()
    
    n_vars = len(matrices.var_names)
    
    def generate_solution():
        """Generate random feasible solution."""
        values = np.zeros(n_vars)
        for i in range(n_vars):
            if matrices.var_types[i] == "B":
                values[i] = np.random.choice([0, 1])
            else:
                values[i] = matrices.var_lb[i]
        return values
    
    def get_neighbors(values):
        """Generate neighborhood."""
        neighbors = []
        for idx in range(n_vars):
            values_new = values.copy()
            if matrices.var_types[idx] == "B":
                values_new[idx] = 1 - values_new[idx]
                neighbors.append(values_new)
            else:
                for delta in [-0.5, 0.5]:
                    values_new = values.copy()
                    values_new[idx] = np.clip(
                        values_new[idx] + delta,
                        matrices.var_lb[idx],
                        matrices.var_ub[idx]
                    )
                    neighbors.append(values_new)
        return neighbors
    
    # Global best
    obj_best_global = float('inf')
    values_best_global = None
    
    for restart in range(restarts):
        values_current = generate_solution()
        obj_current = _compute_objective_minor(matrices, values_current)
        
        for iteration in range(max_iterations):
            neighbors = get_neighbors(values_current)
            
            best_neighbor = None
            best_neighbor_obj = obj_current
            
            for values_new in neighbors:
                if _is_feasible_minor(matrices, values_new):
                    obj_new = _compute_objective_minor(matrices, values_new)
                    if obj_new < best_neighbor_obj:
                        best_neighbor = values_new
                        best_neighbor_obj = obj_new
            
            if best_neighbor is not None and best_neighbor_obj < obj_current:
                values_current = best_neighbor
                obj_current = best_neighbor_obj
            else:
                break
        
        if obj_current < obj_best_global:
            values_best_global = values_current.copy()
            obj_best_global = obj_current
    
    runtime = time.time() - start_time
    return obj_best_global, values_best_global, runtime


def solve_minor_matrices_with_abc(matrices, colony_size=15, max_cycles=50, limit=5):
    """Solve minor matrices with Artificial Bee Colony heuristic."""
    start_time = time.time()
    
    n_vars = len(matrices.var_names)
    
    def generate_solution():
        """Generate random feasible solution."""
        values = np.zeros(n_vars)
        for i in range(n_vars):
            if matrices.var_types[i] == "B":
                values[i] = np.random.choice([0, 1])
            else:
                values[i] = matrices.var_lb[i]
        return values
    
    def modify_solution(values, values_partner):
        """Modify solution using partner."""
        values_new = values.copy()
        idx = np.random.randint(n_vars)
        
        if matrices.var_types[idx] == "B":
            values_new[idx] = values_partner[idx] if np.random.random() < 0.5 else 1 - values_new[idx]
        else:
            values_new[idx] = np.clip(
                values_new[idx] + 0.1 * (values_partner[idx] - values_new[idx]),
                matrices.var_lb[idx],
                matrices.var_ub[idx]
            )
        
        return values_new
    
    # Initialize colony
    food_sources = []
    fitness = []
    trial_counters = []
    
    for _ in range(colony_size):
        values = generate_solution()
        obj = _compute_objective_minor(matrices, values)
        food_sources.append(values)
        fitness.append(1.0 / (1.0 + obj))
        trial_counters.append(0)
    
    best_idx = np.argmax(fitness)
    values_best = food_sources[best_idx].copy()
    obj_best = 1.0 / fitness[best_idx] - 1.0
    
    improvements = 0
    
    for cycle in range(max_cycles):
        # Employed bees
        for i in range(colony_size):
            partner_idx = np.random.choice([j for j in range(colony_size) if j != i])
            values_new = modify_solution(food_sources[i], food_sources[partner_idx])
            
            if _is_feasible_minor(matrices, values_new):
                obj_new = _compute_objective_minor(matrices, values_new)
                fitness_new = 1.0 / (1.0 + obj_new)
                
                if fitness_new > fitness[i]:
                    food_sources[i] = values_new
                    fitness[i] = fitness_new
                    trial_counters[i] = 0
                    improvements += 1
                    
                    if obj_new < obj_best:
                        values_best = values_new.copy()
                        obj_best = obj_new
                else:
                    trial_counters[i] += 1
        
        # Onlooker bees
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness if total_fitness > 0 else 1.0/colony_size for f in fitness]
        
        for i in range(colony_size):
            selected_idx = np.random.choice(range(colony_size), p=probabilities)
            partner_idx = np.random.choice([j for j in range(colony_size) if j != selected_idx])
            values_new = modify_solution(food_sources[selected_idx], food_sources[partner_idx])
            
            if _is_feasible_minor(matrices, values_new):
                obj_new = _compute_objective_minor(matrices, values_new)
                fitness_new = 1.0 / (1.0 + obj_new)
                
                if fitness_new > fitness[selected_idx]:
                    food_sources[selected_idx] = values_new
                    fitness[selected_idx] = fitness_new
                    trial_counters[selected_idx] = 0
                    improvements += 1
                    
                    if obj_new < obj_best:
                        values_best = values_new.copy()
                        obj_best = obj_new
                else:
                    trial_counters[selected_idx] += 1
        
        # Scout bees
        for i in range(colony_size):
            if trial_counters[i] > limit:
                values_new = generate_solution()
                food_sources[i] = values_new
                obj_new = _compute_objective_minor(matrices, values_new)
                fitness[i] = 1.0 / (1.0 + obj_new)
                trial_counters[i] = 0
    
    runtime = time.time() - start_time
    return obj_best, values_best, runtime



def _build_minor_inputs(
    gene: Gene,
    coverage: Coverage,
    major_sols: List[MajorSolution],
):
    """Mirror minor.py candidate allele/mutation extraction."""
    alleles: List[SolvedAllele] = []
    mutations: Set[Mutation] = set()
    for major_sol in major_sols:
        for sa in major_sol.solution:
            alleles += [
                SolvedAllele(gene, sa.major, mi) for mi in gene.alleles[sa.major].minors
            ]
            mutations |= set(gene.alleles[sa.major].func_muts)
            for minor in gene.alleles[sa.major].minors.values():
                mutations |= set(minor.neutral_muts)
        mutations |= set(major_sol.added)

    def default_filter_fn(cov, mut):
        r = gene.region_at(mut.pos)
        if mut.op != "_" and not (
            mut in mutations
            or (r and r[1][0] == "e")
            or (r and r[1] in ["utr3", "utr5", "up"])
        ):
            return False
        cond = cov.basic_filter(mut, cn=coverage.profile.cn_max)
        if mut.op != "_":
            cond = cond and cov.basic_filter(
                mut, cn=major_sols[0].cn_solution.position_cn(mut.pos) + 0.5
            )
        return cond

    cov = coverage.filtered(Coverage.quality_filter)
    cov = cov.filtered(default_filter_fn)
    mutations |= {m for m in gene.random_mutations if cov[m] > 0}

    return alleles, mutations, cov


def main(bam_file: str, gene_name: str, profile_name: str, solver_type: str) -> None:
    print("=" * 70)
    print("MINOR STAR-ALLELE CALLING USING MATRIX EXTRACTION")
    print("=" * 70)

    gene = Gene(script_path(f"aldy.resources.genes/{gene_name.lower()}.yml"))
    profile = Profile.load(gene, profile_name)
    sample = Sample(gene, profile, bam_file, None, None)

    cn_solution = CNSolution(gene, 0, ["1", "1"])

    print(f"\nInput: BAM={bam_file}, Gene={gene.name}, Profile={profile_name}")
    print(f"CN configuration: {dict(cn_solution.solution)}")

    print("\nRunning major calling (cbc)...")
    major_sols = estimate_major(gene, sample.coverage, cn_solution, solver="cbc")
    if not major_sols:
        print("\n✗ No major solutions found")
        return

    alleles, mutations, cov = _build_minor_inputs(gene, sample.coverage, major_sols)
    _print_candidates(gene, alleles, cn_solution, cov, mutations)

    # Use best major solution
    major_sol = major_sols[0]

    print("\nBuilding minor matrices...")
    matrices = build_minor_matrices(gene, cov, major_sol, alleles, mutations)
    print(f"  Vars: {len(matrices.var_names)}")
    print(f"  Equalities: {len(matrices.A_eq)}")
    print(f"  Inequalities: {len(matrices.A_ineq)}")

    if solver_type not in {"ortools", "sa", "hc", "abc", "all"}:
        raise ValueError("Solver must be 'ortools', 'sa', 'hc', 'abc', or 'all'.")

    print("\nSolving minor matrices...")

    results = []
    
    # OR-Tools SCIP
    if solver_type in {"ortools", "all"}:
        print("  Running OR-Tools SCIP...")
        obj, values, runtime = solve_minor_matrices_with_ortools(matrices)
        if values is not None:
            results.append(("OR-Tools SCIP", obj, runtime, values))
        else:
            print("    ✗ Failed")
    
    # Simulated Annealing
    if solver_type in {"sa", "all"}:
        print("  Running Simulated Annealing...")
        obj, values, runtime = solve_minor_matrices_with_simulated_annealing(matrices)
        if values is not None:
            results.append(("Simulated Annealing", obj, runtime, values))
        else:
            print("    ✗ Failed")
    
    # Hill Climbing
    if solver_type in {"hc", "all"}:
        print("  Running Hill Climbing...")
        obj, values, runtime = solve_minor_matrices_with_hill_climbing(matrices)
        if values is not None:
            results.append(("Hill Climbing", obj, runtime, values))
        else:
            print("    ✗ Failed")
    
    # ABC
    if solver_type in {"abc", "all"}:
        print("  Running Artificial Bee Colony...")
        obj, values, runtime = solve_minor_matrices_with_abc(matrices)
        if values is not None:
            results.append(("ABC", obj, runtime, values))
        else:
            print("    ✗ Failed")

    if not results:
        print("\n✗ All solvers failed")
        return

    print("\n" + "=" * 70)
    print("SOLVER COMPARISON")
    print("=" * 70)
    print(f"\n{'Solver':<25} {'Objective':>12} {'Runtime':>10}")
    print(f"{'-'*48}")
    for name, obj, runtime, _ in results:
        print(f"{name:<25} {obj:>12.4f} {runtime:>9.2f}s")

    # Use the best solution
    best_idx = np.argmin([r[1] for r in results])
    best_name, best_obj, best_runtime, best_values = results[best_idx]

    print(f"\n✓ Best solution: {best_name} (objective: {best_obj:.4f})")
    
    if len(results) > 1:
        # Check agreement
        all_allele_sets = []
        for name, obj, runtime, values in results:
            selected = []
            for var_name, val in zip(matrices.var_names, values):
                if var_name.startswith("x_") and val > 0.5:
                    selected.append(var_name)
            all_allele_sets.append(set(selected))
        
        if len(set(frozenset(s) for s in all_allele_sets)) == 1:
            print(f"✓ All solvers found the SAME alleles!")
        else:
            print(f"⚠ Solvers found DIFFERENT alleles")

    # Report selected allele copies
    selected = []
    for name, val in zip(matrices.var_names, best_values):
        if name.startswith("x_") and val > 0.5:
            selected.append(name)

    print("\nSelected allele copies:")
    for s in selected:
        print(f"  {s}")


if __name__ == "__main__":
    bam_file = "../data/NA07000.bam"
    gene_name = "CYP2D6"
    profile_name = "illumina"
    solver_type = "ortools"

    if len(sys.argv) >= 5:
        bam_file = sys.argv[1]
        gene_name = sys.argv[2]
        profile_name = sys.argv[3]
        solver_type = sys.argv[4]
    elif len(sys.argv) >= 4:
        bam_file = sys.argv[1]
        gene_name = sys.argv[2]
        profile_name = sys.argv[3]
    elif len(sys.argv) >= 2:
        solver_type = sys.argv[1]
    
    if solver_type not in {'ortools', 'sa', 'hc', 'abc', 'all'}:
        print(f"Usage: {sys.argv[0]} [BAM_FILE GENE PROFILE SOLVER]")
        print(f"   or: {sys.argv[0]} [SOLVER]")
        print(f"\nSOLVER options:")
        print(f"  ortools - OR-Tools SCIP solver (exact)")
        print(f"  sa      - Simulated Annealing (heuristic)")
        print(f"  hc      - Hill Climbing (local search)")
        print(f"  abc     - Artificial Bee Colony (swarm)")
        print(f"  all     - Compare all solvers (default)")
        sys.exit(1)

    print(f"Selected solver(s): {solver_type.upper()}")

    try:
        main(bam_file, gene_name, profile_name, solver_type)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
