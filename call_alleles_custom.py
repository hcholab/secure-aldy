#!/usr/bin/env python
"""
Call CYP2D6 star alleles on NA07000.bam using matrix extraction + custom solver.

Usage:
    python call_alleles_custom.py [ortools|sa|hc|abc|exhaustive|all]

Solvers:
    ortools    - Use OR-Tools SCIP (exact ILP solver, guaranteed optimal)
    sa         - Use Simulated Annealing (heuristic, escapes local optima)
    hc         - Use Hill Climbing (simple local search, fast)
    abc        - Use Artificial Bee Colony (swarm intelligence)
    exhaustive - Use Exhaustive Search (tries all feasible combinations)
    all        - Compare all solvers (default)

Examples:
    python call_alleles_custom.py ortools      # Fast exact solution
    python call_alleles_custom.py sa           # Simulated annealing
    python call_alleles_custom.py hc           # Hill climbing
    python call_alleles_custom.py abc          # Bee colony optimization
    python call_alleles_custom.py exhaustive   # Exhaustive search
    python call_alleles_custom.py all          # Compare all methods
"""

import numpy as np
import sys
import time
from ortools.linear_solver import pywraplp
from collections import defaultdict
from itertools import combinations

from aldy.major_matrix import build_major_matrices, extract_solution, validate_solution
from aldy.major import _filter_alleles
from aldy.gene import Gene, Mutation
from aldy.profile import Profile
from aldy.coverage import Coverage
from aldy.solutions import CNSolution
from aldy.common import script_path
from aldy.sam import Sample


def solve_with_ortools_custom(matrices):
    """Custom ILP solver using OR-tools."""
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"CUSTOM ILP SOLVER (OR-Tools SCIP)")
    print(f"{'='*70}")
    print(f"Problem size:")
    print(f"  - {matrices.n_alleles} allele copy variables")
    print(f"  - {matrices.n_mutations} mutation variables")
    print(f"  - {matrices.n_cn_configs} CN configurations")
    print(f"  - {len(matrices.P_indices)} ordering constraints")
    
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("Could not create SCIP solver")
    
    # Create variables
    x = [solver.BoolVar(f'x_{j}') for j in range(matrices.n_alleles)]
    y = [solver.BoolVar(f'y_{i}') for i in range(matrices.n_mutations)]
    z = solver.BoolVar('z')
    e = [solver.NumVar(-solver.infinity(), solver.infinity(), f'e_{i}')
         for i in range(matrices.n_mutations)]
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
    
    # Ordering constraints
    for j1, j2 in matrices.P_indices:
        constraint = solver.Constraint(-solver.infinity(), 0)
        constraint.SetCoefficient(x[j1], 1)
        constraint.SetCoefficient(x[j2], -1)
    
    # Novel indicator constraints
    for i in range(matrices.n_mutations):
        constraint = solver.Constraint(0, solver.infinity())
        constraint.SetCoefficient(z, 1)
        constraint.SetCoefficient(y[i], -1)
    
    constraint = solver.Constraint(0, solver.infinity())
    constraint.SetCoefficient(z, matrices.n_mutations)
    for i in range(matrices.n_mutations):
        constraint.SetCoefficient(y[i], -1)
    
    # One novel per position
    for pos, indices in matrices.position_groups.items():
        constraint = solver.Constraint(-solver.infinity(), 1)
        for i in indices:
            constraint.SetCoefficient(y[i], 1)
    
    # XOR-like constraint: mutation must be explained
    for i in range(matrices.n_mutations):
        if matrices.mutation_vars[i].op != "_":
            constraint = solver.Constraint(1, solver.infinity())
            for j in range(matrices.n_alleles):
                if matrices.A[i, j] != 0:
                    constraint.SetCoefficient(x[j], float(matrices.A[i, j]))
            constraint.SetCoefficient(y[i], 1)
    
    # Absolute value constraints
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
    print(f"\nSolving ILP...")
    solver.SetTimeLimit(60000)  # 60 second timeout
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        print(f"✓ Found OPTIMAL solution")
    elif status == pywraplp.Solver.FEASIBLE:
        print(f"✓ Found FEASIBLE solution")
    else:
        print(f"✗ Solver failed with status: {status}")
        return None, None, None, None, None, None
    
    x_val = np.array([x[j].solution_value() for j in range(matrices.n_alleles)])
    y_val = np.array([y[i].solution_value() for i in range(matrices.n_mutations)])
    e_val = np.array([e[i].solution_value() for i in range(matrices.n_mutations)])
    z_val = z.solution_value()
    obj_val = objective.Value()
    
    runtime = time.time() - start_time
    
    print(f"\nObjective value: {obj_val:.4f}")
    print(f"  Coverage fit error: {np.sum(np.abs(e_val)):.4f}")
    print(f"  Novel mutation penalty: {matrices.lambda_novel * z_val:.4f}")
    print(f"  Mutation count penalty: {0.1 * np.sum(y_val):.4f}")
    print(f"\nRuntime: {runtime:.2f} seconds")
    
    return status, obj_val, x_val, y_val, e_val, z_val, runtime


def solve_with_simulated_annealing(matrices, max_iterations=10000, initial_temp=100.0, cooling_rate=0.995):
    """
    Custom heuristic solver using Simulated Annealing.
    
    This is a metaheuristic that explores the solution space by occasionally
    accepting worse solutions to escape local optima.
    """
    
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"CUSTOM HEURISTIC SOLVER (Simulated Annealing)")
    print(f"{'='*70}")
    print(f"Problem size:")
    print(f"  - {matrices.n_alleles} allele copy variables")
    print(f"  - {matrices.n_mutations} mutation variables")
    print(f"  - {matrices.n_cn_configs} CN configurations")
    print(f"  - {len(matrices.P_indices)} ordering constraints")
    
    print(f"\nSimulated Annealing parameters:")
    print(f"  - Max iterations: {max_iterations}")
    print(f"  - Initial temperature: {initial_temp}")
    print(f"  - Cooling rate: {cooling_rate}")
    
    def compute_objective(x, y, z):
        """Compute objective value for given solution."""
        e = matrices.c - matrices.A @ x - y
        return np.sum(np.abs(e)) + matrices.lambda_novel * z + 0.1 * np.sum(y)
    
    def is_feasible(x, y):
        """Check if solution satisfies hard constraints."""
        # CN configuration must match
        for t in range(matrices.n_cn_configs):
            if abs(matrices.B[t, :] @ x - matrices.d[t]) > 0.5:
                return False
        
        # Ordering constraints
        for j1, j2 in matrices.P_indices:
            if x[j1] > x[j2] + 0.5:
                return False
        
        # One novel per position
        for pos, indices in matrices.position_groups.items():
            if sum(y[i] for i in indices) > 1.5:
                return False
        
        return True
    
    def generate_initial_solution():
        """Generate a feasible initial solution greedily."""
        x = np.zeros(matrices.n_alleles)
        y = np.zeros(matrices.n_mutations)
        
        # Select alleles to satisfy CN constraints
        for t in range(matrices.n_cn_configs):
            required = int(matrices.d[t])
            candidates = [j for j in range(matrices.n_alleles) if matrices.B[t, j] == 1]
            for j in candidates[:required]:
                x[j] = 1
        
        # Mark mutations as novel if not covered
        for i in range(matrices.n_mutations):
            if matrices.mutation_vars[i].op != "_":
                coverage_from_alleles = matrices.A[i, :] @ x
                if coverage_from_alleles < 0.5:
                    y[i] = 1
        
        z = 1 if np.sum(y) > 0.5 else 0
        return x, y, z
    
    def neighbor(x, y, z):
        """Generate a neighboring solution."""
        x_new = x.copy()
        y_new = y.copy()
        
        action = np.random.choice(['swap_allele', 'flip_novel', 'swap_allele_pair'])
        
        if action == 'swap_allele':
            # Try to swap one allele for another with same CN config
            selected = [j for j in range(matrices.n_alleles) if x[j] > 0.5]
            unselected = [j for j in range(matrices.n_alleles) if x[j] < 0.5]
            
            if selected and unselected:
                j_out = np.random.choice(selected)
                # Find unselected allele with same CN config
                t_out = [t for t in range(matrices.n_cn_configs) if matrices.B[t, j_out] > 0.5][0]
                candidates = [j for j in unselected if matrices.B[t_out, j] > 0.5]
                if candidates:
                    j_in = np.random.choice(candidates)
                    x_new[j_out] = 0
                    x_new[j_in] = 1
        
        elif action == 'flip_novel':
            # Flip a novel mutation indicator
            if matrices.n_mutations > 0:
                i = np.random.randint(matrices.n_mutations)
                if matrices.mutation_vars[i].op != "_":
                    y_new[i] = 1 - y_new[i]
        
        else:  # swap_allele_pair
            # Swap two alleles with same CN config
            selected = [j for j in range(matrices.n_alleles) if x[j] > 0.5]
            if len(selected) >= 2:
                j1, j2 = np.random.choice(selected, 2, replace=False)
                # Find candidates with their CN configs
                t1 = [t for t in range(matrices.n_cn_configs) if matrices.B[t, j1] > 0.5][0]
                t2 = [t for t in range(matrices.n_cn_configs) if matrices.B[t, j2] > 0.5][0]
                unselected = [j for j in range(matrices.n_alleles) if x[j] < 0.5]
                c1 = [j for j in unselected if matrices.B[t1, j] > 0.5]
                c2 = [j for j in unselected if matrices.B[t2, j] > 0.5]
                if c1 and c2:
                    x_new[j1] = 0
                    x_new[j2] = 0
                    x_new[np.random.choice(c1)] = 1
                    x_new[np.random.choice(c2)] = 1
        
        z_new = 1 if np.sum(y_new) > 0.5 else 0
        return x_new, y_new, z_new
    
    # Initialize
    print(f"\nGenerating initial solution...")
    x_current, y_current, z_current = generate_initial_solution()
    
    if not is_feasible(x_current, y_current):
        print(f"Warning: Initial solution is infeasible, adjusting...")
        # Try to fix feasibility
        for _ in range(100):
            x_current, y_current, z_current = generate_initial_solution()
            if is_feasible(x_current, y_current):
                break
    
    obj_current = compute_objective(x_current, y_current, z_current)
    x_best, y_best, z_best = x_current.copy(), y_current.copy(), z_current
    obj_best = obj_current
    
    print(f"Initial objective: {obj_current:.4f}")
    print(f"\nRunning simulated annealing...")
    
    temperature = initial_temp
    accepted = 0
    improved = 0
    
    for iteration in range(max_iterations):
        # Generate neighbor
        x_new, y_new, z_new = neighbor(x_current, y_current, z_current)
        
        if not is_feasible(x_new, y_new):
            continue  # Skip infeasible solutions
        
        obj_new = compute_objective(x_new, y_new, z_new)
        delta = obj_new - obj_current
        
        # Accept or reject
        if delta < 0 or np.random.random() < np.exp(-delta / temperature):
            x_current, y_current, z_current = x_new, y_new, z_new
            obj_current = obj_new
            accepted += 1
            
            if obj_new < obj_best:
                x_best, y_best, z_best = x_new.copy(), y_new.copy(), z_new
                obj_best = obj_new
                improved += 1
        
        # Cool down
        temperature *= cooling_rate
        
        # Progress report
        if (iteration + 1) % 1000 == 0:
            print(f"  Iteration {iteration+1}/{max_iterations}: "
                  f"Best={obj_best:.4f}, Current={obj_current:.4f}, "
                  f"Temp={temperature:.4f}, Accepted={accepted}")
    
    print(f"\n✓ Completed simulated annealing")
    print(f"\nFinal statistics:")
    print(f"  Total accepted moves: {accepted}/{max_iterations} ({100*accepted/max_iterations:.1f}%)")
    print(f"  Improvements found: {improved}")
    print(f"  Final temperature: {temperature:.6f}")
    
    # Compute final error vector
    e_best = matrices.c - matrices.A @ x_best - y_best
    
    print(f"\nBest objective value: {obj_best:.4f}")
    print(f"  Coverage fit error: {np.sum(np.abs(e_best)):.4f}")
    print(f"  Novel mutation penalty: {matrices.lambda_novel * z_best:.4f}")
    print(f"  Mutation count penalty: {0.1 * np.sum(y_best):.4f}")
    
    runtime = time.time() - start_time
    print(f"\nRuntime: {runtime:.2f} seconds")
    
    return "HEURISTIC", obj_best, x_best, y_best, e_best, z_best, runtime


def solve_with_hill_climbing(matrices, max_iterations=5000, restarts=10):
    """
    Custom heuristic solver using Hill Climbing with random restarts.
    
    Hill climbing is a simple local search that always moves to better neighbors.
    Multiple restarts help escape local optima.
    """
    
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"CUSTOM HEURISTIC SOLVER (Hill Climbing)")
    print(f"{'='*70}")
    print(f"Problem size:")
    print(f"  - {matrices.n_alleles} allele copy variables")
    print(f"  - {matrices.n_mutations} mutation variables")
    print(f"  - {matrices.n_cn_configs} CN configurations")
    print(f"  - {len(matrices.P_indices)} ordering constraints")
    
    print(f"\nHill Climbing parameters:")
    print(f"  - Max iterations per restart: {max_iterations}")
    print(f"  - Number of restarts: {restarts}")
    
    def compute_objective(x, y, z):
        """Compute objective value for given solution."""
        e = matrices.c - matrices.A @ x - y
        return np.sum(np.abs(e)) + matrices.lambda_novel * z + 0.1 * np.sum(y)
    
    def is_feasible(x, y):
        """Check if solution satisfies hard constraints."""
        # CN configuration must match
        for t in range(matrices.n_cn_configs):
            if abs(matrices.B[t, :] @ x - matrices.d[t]) > 0.5:
                return False
        
        # Ordering constraints
        for j1, j2 in matrices.P_indices:
            if x[j1] > x[j2] + 0.5:
                return False
        
        # One novel per position
        for pos, indices in matrices.position_groups.items():
            if sum(y[i] for i in indices) > 1.5:
                return False
        
        return True
    
    def generate_initial_solution():
        """Generate a feasible initial solution greedily."""
        x = np.zeros(matrices.n_alleles)
        y = np.zeros(matrices.n_mutations)
        
        # Select alleles to satisfy CN constraints
        for t in range(matrices.n_cn_configs):
            required = int(matrices.d[t])
            candidates = [j for j in range(matrices.n_alleles) if matrices.B[t, j] == 1]
            selected = np.random.choice(candidates, min(required, len(candidates)), replace=False)
            for j in selected:
                x[j] = 1
        
        # Mark mutations as novel if not covered
        for i in range(matrices.n_mutations):
            if matrices.mutation_vars[i].op != "_":
                coverage_from_alleles = matrices.A[i, :] @ x
                if coverage_from_alleles < 0.5:
                    y[i] = 1
        
        z = 1 if np.sum(y) > 0.5 else 0
        return x, y, z
    
    def get_neighbors(x, y, z):
        """Generate all feasible neighboring solutions."""
        neighbors = []
        
        # Try swapping each selected allele with an unselected one (same CN config)
        selected = [j for j in range(matrices.n_alleles) if x[j] > 0.5]
        unselected = [j for j in range(matrices.n_alleles) if x[j] < 0.5]
        
        for j_out in selected:
            t_out = [t for t in range(matrices.n_cn_configs) if matrices.B[t, j_out] > 0.5][0]
            for j_in in unselected:
                if matrices.B[t_out, j_in] > 0.5:
                    x_new = x.copy()
                    x_new[j_out] = 0
                    x_new[j_in] = 1
                    y_new = y.copy()
                    z_new = 1 if np.sum(y_new) > 0.5 else 0
                    if is_feasible(x_new, y_new):
                        neighbors.append((x_new, y_new, z_new))
        
        # Try flipping novel mutation indicators
        for i in range(matrices.n_mutations):
            if matrices.mutation_vars[i].op != "_":
                y_new = y.copy()
                y_new[i] = 1 - y_new[i]
                z_new = 1 if np.sum(y_new) > 0.5 else 0
                if is_feasible(x, y_new):
                    neighbors.append((x.copy(), y_new, z_new))
        
        return neighbors
    
    # Global best
    x_best_global, y_best_global, z_best_global = None, None, None
    obj_best_global = float('inf')
    
    print(f"\nRunning hill climbing with {restarts} restarts...")
    
    total_iterations = 0
    total_improvements = 0
    
    for restart in range(restarts):
        # Generate initial solution
        x_current, y_current, z_current = generate_initial_solution()
        obj_current = compute_objective(x_current, y_current, z_current)
        
        improved_this_restart = 0
        iterations_this_restart = 0
        
        for iteration in range(max_iterations):
            iterations_this_restart += 1
            total_iterations += 1
            
            # Generate all neighbors
            neighbors = get_neighbors(x_current, y_current, z_current)
            
            if not neighbors:
                break  # No neighbors, stuck
            
            # Find best neighbor
            best_neighbor = None
            best_neighbor_obj = obj_current
            
            for x_new, y_new, z_new in neighbors:
                obj_new = compute_objective(x_new, y_new, z_new)
                if obj_new < best_neighbor_obj:
                    best_neighbor = (x_new, y_new, z_new)
                    best_neighbor_obj = obj_new
            
            # Move to best neighbor if better
            if best_neighbor is not None and best_neighbor_obj < obj_current:
                x_current, y_current, z_current = best_neighbor
                obj_current = best_neighbor_obj
                improved_this_restart += 1
                total_improvements += 1
            else:
                # Local optimum reached
                break
        
        # Update global best
        if obj_current < obj_best_global:
            x_best_global = x_current.copy()
            y_best_global = y_current.copy()
            z_best_global = z_current
            obj_best_global = obj_current
        
        print(f"  Restart {restart+1}/{restarts}: obj={obj_current:.4f} "
              f"(iterations={iterations_this_restart}, improvements={improved_this_restart})")
    
    print(f"\n✓ Completed hill climbing")
    print(f"\nFinal statistics:")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Total improvements: {total_improvements}")
    print(f"  Best objective: {obj_best_global:.4f}")
    
    # Compute final error vector
    e_best = matrices.c - matrices.A @ x_best_global - y_best_global
    
    print(f"\nBest objective value: {obj_best_global:.4f}")
    print(f"  Coverage fit error: {np.sum(np.abs(e_best)):.4f}")
    print(f"  Novel mutation penalty: {matrices.lambda_novel * z_best_global:.4f}")
    print(f"  Mutation count penalty: {0.1 * np.sum(y_best_global):.4f}")
    
    runtime = time.time() - start_time
    print(f"\nRuntime: {runtime:.2f} seconds")
    
    return "HEURISTIC", obj_best_global, x_best_global, y_best_global, e_best, z_best_global, runtime


def solve_with_abc(matrices, colony_size=20, max_cycles=100, limit=10):
    """
    Custom heuristic solver using Artificial Bee Colony optimization.
    
    ABC simulates the foraging behavior of honey bees with three types:
    - Employed bees: Exploit known food sources
    - Onlooker bees: Choose sources based on quality
    - Scout bees: Explore new random sources
    """
    
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"CUSTOM HEURISTIC SOLVER (Artificial Bee Colony)")
    print(f"{'='*70}")
    print(f"Problem size:")
    print(f"  - {matrices.n_alleles} allele copy variables")
    print(f"  - {matrices.n_mutations} mutation variables")
    print(f"  - {matrices.n_cn_configs} CN configurations")
    print(f"  - {len(matrices.P_indices)} ordering constraints")
    
    print(f"\nABC parameters:")
    print(f"  - Colony size: {colony_size}")
    print(f"  - Max cycles: {max_cycles}")
    print(f"  - Abandonment limit: {limit}")
    
    def compute_objective(x, y, z):
        """Compute objective value for given solution."""
        e = matrices.c - matrices.A @ x - y
        return np.sum(np.abs(e)) + matrices.lambda_novel * z + 0.1 * np.sum(y)
    
    def is_feasible(x, y):
        """Check if solution satisfies hard constraints."""
        # CN configuration must match
        for t in range(matrices.n_cn_configs):
            if abs(matrices.B[t, :] @ x - matrices.d[t]) > 0.5:
                return False
        
        # Ordering constraints
        for j1, j2 in matrices.P_indices:
            if x[j1] > x[j2] + 0.5:
                return False
        
        # One novel per position
        for pos, indices in matrices.position_groups.items():
            if sum(y[i] for i in indices) > 1.5:
                return False
        
        return True
    
    def generate_solution():
        """Generate a random feasible solution."""
        x = np.zeros(matrices.n_alleles)
        y = np.zeros(matrices.n_mutations)
        
        # Select alleles to satisfy CN constraints
        for t in range(matrices.n_cn_configs):
            required = int(matrices.d[t])
            candidates = [j for j in range(matrices.n_alleles) if matrices.B[t, j] == 1]
            if len(candidates) >= required:
                selected = np.random.choice(candidates, required, replace=False)
                for j in selected:
                    x[j] = 1
        
        # Randomly mark some mutations as novel
        for i in range(matrices.n_mutations):
            if matrices.mutation_vars[i].op != "_":
                if np.random.random() < 0.2:  # 20% chance
                    y[i] = 1
        
        z = 1 if np.sum(y) > 0.5 else 0
        return x, y, z
    
    def modify_solution(x, y, z, x_partner, y_partner):
        """Generate new solution by modifying current one with partner."""
        x_new = x.copy()
        y_new = y.copy()
        
        # Randomly choose modification strategy
        if np.random.random() < 0.5 and matrices.n_alleles > 0:
            # Swap an allele
            selected = [j for j in range(matrices.n_alleles) if x[j] > 0.5]
            unselected = [j for j in range(matrices.n_alleles) if x[j] < 0.5]
            
            if selected and unselected:
                j_out = np.random.choice(selected)
                t_out = [t for t in range(matrices.n_cn_configs) if matrices.B[t, j_out] > 0.5][0]
                candidates = [j for j in unselected if matrices.B[t_out, j] > 0.5]
                
                if candidates:
                    j_in = np.random.choice(candidates)
                    x_new[j_out] = 0
                    x_new[j_in] = 1
        else:
            # Modify novel mutations
            if matrices.n_mutations > 0:
                i = np.random.randint(matrices.n_mutations)
                if matrices.mutation_vars[i].op != "_":
                    # Blend with partner
                    if np.random.random() < 0.5:
                        y_new[i] = y_partner[i]
                    else:
                        y_new[i] = 1 - y_new[i]
        
        z_new = 1 if np.sum(y_new) > 0.5 else 0
        
        if is_feasible(x_new, y_new):
            return x_new, y_new, z_new
        else:
            return x.copy(), y.copy(), z  # Return original if infeasible
    
    # Initialize food sources (solutions)
    print(f"\nInitializing colony of {colony_size} bees...")
    food_sources = []
    fitness = []
    trial_counters = []
    
    for _ in range(colony_size):
        x, y, z = generate_solution()
        obj = compute_objective(x, y, z)
        food_sources.append((x, y, z))
        fitness.append(1.0 / (1.0 + obj))  # Higher fitness for lower objective
        trial_counters.append(0)
    
    best_idx = np.argmax(fitness)
    x_best = food_sources[best_idx][0].copy()
    y_best = food_sources[best_idx][1].copy()
    z_best = food_sources[best_idx][2]
    obj_best = 1.0 / fitness[best_idx] - 1.0
    
    print(f"Initial best objective: {obj_best:.4f}")
    print(f"\nRunning ABC optimization...")
    
    improvements = 0
    
    for cycle in range(max_cycles):
        # EMPLOYED BEE PHASE
        for i in range(colony_size):
            # Choose a random partner
            partner_idx = np.random.choice([j for j in range(colony_size) if j != i])
            x_partner, y_partner, z_partner = food_sources[partner_idx]
            
            # Generate new solution
            x, y, z = food_sources[i]
            x_new, y_new, z_new = modify_solution(x, y, z, x_partner, y_partner)
            obj_new = compute_objective(x_new, y_new, z_new)
            fitness_new = 1.0 / (1.0 + obj_new)
            
            # Greedy selection
            if fitness_new > fitness[i]:
                food_sources[i] = (x_new, y_new, z_new)
                fitness[i] = fitness_new
                trial_counters[i] = 0
                improvements += 1
                
                if obj_new < obj_best:
                    x_best, y_best, z_best = x_new.copy(), y_new.copy(), z_new
                    obj_best = obj_new
            else:
                trial_counters[i] += 1
        
        # ONLOOKER BEE PHASE
        # Calculate selection probabilities
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness if total_fitness > 0 else 1.0/colony_size 
                        for f in fitness]
        
        for i in range(colony_size):
            # Roulette wheel selection
            selected_idx = np.random.choice(range(colony_size), p=probabilities)
            
            # Choose a random partner
            partner_idx = np.random.choice([j for j in range(colony_size) if j != selected_idx])
            x_partner, y_partner, z_partner = food_sources[partner_idx]
            
            # Generate new solution
            x, y, z = food_sources[selected_idx]
            x_new, y_new, z_new = modify_solution(x, y, z, x_partner, y_partner)
            obj_new = compute_objective(x_new, y_new, z_new)
            fitness_new = 1.0 / (1.0 + obj_new)
            
            # Greedy selection
            if fitness_new > fitness[selected_idx]:
                food_sources[selected_idx] = (x_new, y_new, z_new)
                fitness[selected_idx] = fitness_new
                trial_counters[selected_idx] = 0
                improvements += 1
                
                if obj_new < obj_best:
                    x_best, y_best, z_best = x_new.copy(), y_new.copy(), z_new
                    obj_best = obj_new
            else:
                trial_counters[selected_idx] += 1
        
        # SCOUT BEE PHASE
        # Abandon sources that haven't improved for 'limit' trials
        for i in range(colony_size):
            if trial_counters[i] > limit:
                # Generate new random solution
                x_new, y_new, z_new = generate_solution()
                obj_new = compute_objective(x_new, y_new, z_new)
                food_sources[i] = (x_new, y_new, z_new)
                fitness[i] = 1.0 / (1.0 + obj_new)
                trial_counters[i] = 0
        
        # Progress report
        if (cycle + 1) % 20 == 0:
            avg_fitness = np.mean(fitness)
            print(f"  Cycle {cycle+1}/{max_cycles}: Best={obj_best:.4f}, "
                  f"Avg_fitness={avg_fitness:.4f}, Improvements={improvements}")
    
    print(f"\n✓ Completed ABC optimization")
    print(f"\nFinal statistics:")
    print(f"  Total improvements: {improvements}")
    print(f"  Best objective: {obj_best:.4f}")
    
    # Compute final error vector
    e_best = matrices.c - matrices.A @ x_best - y_best
    
    print(f"\nBest objective value: {obj_best:.4f}")
    print(f"  Coverage fit error: {np.sum(np.abs(e_best)):.4f}")
    print(f"  Novel mutation penalty: {matrices.lambda_novel * z_best:.4f}")
    print(f"  Mutation count penalty: {0.1 * np.sum(y_best):.4f}")
    
    runtime = time.time() - start_time
    print(f"\nRuntime: {runtime:.2f} seconds")
    
    return "HEURISTIC", obj_best, x_best, y_best, e_best, z_best, runtime


def solve_with_exhaustive_search(matrices, max_combos=10000):
    """
    Exhaustive search solver - tries all feasible combinations of allele selections.
    
    For small problems, this can find the optimal solution by brute force.
    Limited by max_combos to prevent runtime explosion on large problems.
    """
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"CUSTOM EXHAUSTIVE SEARCH SOLVER")
    print(f"{'='*70}")
    print(f"Problem size:")
    print(f"  - {matrices.n_alleles} allele copy variables")
    print(f"  - {matrices.n_mutations} mutation variables")
    print(f"  - {matrices.n_cn_configs} CN configurations")
    print(f"  - {len(matrices.P_indices)} ordering constraints")
    
    def compute_objective(x, y, z):
        """Compute objective value for given solution."""
        e = matrices.c - matrices.A @ x - y
        return np.sum(np.abs(e)) + matrices.lambda_novel * z + 0.1 * np.sum(y)
    
    def is_feasible(x, y):
        """Check if solution satisfies hard constraints."""
        # CN configuration must match
        for t in range(matrices.n_cn_configs):
            if abs(matrices.B[t, :] @ x - matrices.d[t]) > 0.5:
                return False
        
        # Ordering constraints
        for j1, j2 in matrices.P_indices:
            if x[j1] > x[j2] + 0.5:
                return False
        
        # One novel per position
        for pos, indices in matrices.position_groups.items():
            if sum(y[i] for i in indices) > 1.5:
                return False
        
        return True
    
    print(f"\nGenerating all feasible x combinations...")
    
    # Generate all feasible x combinations (respecting CN constraints and ordering)
    feasible_x_combos = []
    
    # For each CN config, generate all valid combinations
    cn_config_alleles = [[] for _ in range(matrices.n_cn_configs)]
    for j in range(matrices.n_alleles):
        for t in range(matrices.n_cn_configs):
            if matrices.B[t, j] > 0.5:
                cn_config_alleles[t].append(j)
    
    # Generate combinations respecting ordering constraints
    def generate_ordered_combos(config_idx, current_x):
        if config_idx == matrices.n_cn_configs:
            return [current_x.copy()]
        
        required = int(matrices.d[config_idx])
        candidates = cn_config_alleles[config_idx]
        
        result = []
        for combo in combinations(candidates, required):
            x_new = current_x.copy()
            for j in combo:
                x_new[j] = 1
            result.extend(generate_ordered_combos(config_idx + 1, x_new))
        
        return result
    
    feasible_x_combos = generate_ordered_combos(0, np.zeros(matrices.n_alleles))
    print(f"  Found {len(feasible_x_combos)} feasible x combinations")
    
    if len(feasible_x_combos) > max_combos:
        print(f"  Limiting to first {max_combos} combinations (too many to evaluate all)")
        feasible_x_combos = feasible_x_combos[:max_combos]
    
    # For each feasible x, try all feasible y combinations
    print(f"\nSearching through all feasible solutions...")
    
    x_best = None
    y_best = None
    z_best = 0
    obj_best = float('inf')
    solutions_evaluated = 0
    
    for x in feasible_x_combos:
        # Try y = all zeros first
        y = np.zeros(matrices.n_mutations)
        z = 0
        
        if is_feasible(x, y):
            obj = compute_objective(x, y, z)
            solutions_evaluated += 1
            if obj < obj_best:
                x_best = x.copy()
                y_best = y.copy()
                z_best = z
                obj_best = obj
        
        # Try minimal novel mutations
        if matrices.position_groups:
            for pos, indices in matrices.position_groups.items():
                for i in indices:
                    y_test = np.zeros(matrices.n_mutations)
                    y_test[i] = 1
                    z_test = 1
                    
                    if is_feasible(x, y_test):
                        obj = compute_objective(x, y_test, z_test)
                        solutions_evaluated += 1
                        if obj < obj_best:
                            x_best = x.copy()
                            y_best = y_test.copy()
                            z_best = z_test
                            obj_best = obj
    
    runtime = time.time() - start_time
    
    # Compute final error vector
    e_best = matrices.c - matrices.A @ x_best - y_best
    
    print(f"\n✓ Completed exhaustive search")
    print(f"  Solutions evaluated: {solutions_evaluated}")
    print(f"\nBest objective value: {obj_best:.4f}")
    print(f"  Coverage fit error: {np.sum(np.abs(e_best)):.4f}")
    print(f"  Novel mutation penalty: {matrices.lambda_novel * z_best:.4f}")
    print(f"  Mutation count penalty: {0.1 * np.sum(y_best):.4f}")
    print(f"\nRuntime: {runtime:.2f} seconds")
    
    return "HEURISTIC", obj_best, x_best, y_best, e_best, z_best, runtime


def main(solver_type='all', bam_file="../data/NA07000.bam", gene_name="CYP2D6", profile_name="illumina"):
    """Load BAM file and call alleles using custom solver."""
    
    print("="*70)
    print("CYP2D6 STAR-ALLELE CALLING USING CUSTOM ILP SOLVER")
    print("="*70)
    
    print(f"\nInput:")
    print(f"  BAM file: {bam_file}")
    print(f"  Gene: {gene_name}")
    print(f"  Profile: {profile_name}")
    
    # Load gene and profile
    print(f"\nLoading gene database...")
    gene = Gene(script_path(f"aldy.resources.genes/{gene_name.lower()}.yml"))
    profile = Profile.load(gene, profile_name)
    
    print(f"  Gene: {gene.name}")
    print(f"  Genome: {gene.genome}")
    print(f"  Known alleles: {len(gene.alleles)}")
    print(f"  Functional mutations: {len([m for m in gene.mutations if gene.is_functional(m)])}")
    
    # Load sample from BAM
    print(f"\nLoading sample from BAM file...")
    sample = Sample(gene, profile, bam_file, None, None)
    
    # Calculate average coverage manually
    total_cov = sum(
        sum(len(reads) for reads in pos_cov.values())
        for pos_cov in sample.coverage._coverage.values()
    )
    num_positions = len(sample.coverage._coverage)
    avg_cov = total_cov / num_positions if num_positions > 0 else 0
    print(f"  Average coverage: {avg_cov:.2f}x")
    
    # Get CN solution (we'll use the one from the sample's CN calling)
    # For simplicity, assume diploid (2 copies)
    cn_solution = CNSolution(gene, 0, ["1", "1"])
    print(f"\nCN configuration: {dict(cn_solution.solution)}")
    
    # Filter alleles and get functional mutations
    print(f"\nFiltering alleles and mutations...")
    alleles, filtered_cov = _filter_alleles(gene, sample.coverage, cn_solution)
    
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
    
    print(f"  Candidate alleles: {len(alleles)}")
    print(f"  Functional mutations observed: {len(func_muts)}")
    
    if not alleles:
        print("\n✗ No candidate alleles found!")
        return
    
    # Show top candidate alleles
    print(f"\n  Top candidate alleles:")
    for i, allele_name in enumerate(sorted(alleles.keys())[:10]):
        allele = alleles[allele_name]
        print(f"    *{allele_name:15} (CN config: *{allele.cn_config}, "
              f"{len(allele.func_muts)} functional muts)")
    
    # Extract matrices
    print(f"\n{'='*70}")
    print(f"EXTRACTING ILP MATRICES")
    print(f"{'='*70}")
    matrices = build_major_matrices(gene, filtered_cov, cn_solution, alleles, func_muts)
    
    print(f"\nMatrix dimensions:")
    print(f"  A: {matrices.A.shape[0]} × {matrices.A.shape[1]} (mutations × alleles)")
    print(f"  B: {matrices.B.shape[0]} × {matrices.B.shape[1]} (CN configs × alleles)")
    print(f"  c: {len(matrices.c)} (coverage observations)")
    print(f"  d: {len(matrices.d)} (CN requirements)")
    
    # Solve with selected solver
    if solver_type == 'ortools':
        status, obj_val, x_val, y_val, e_val, z_val, runtime = solve_with_ortools_custom(matrices)
    elif solver_type == 'sa':
        status, obj_val, x_val, y_val, e_val, z_val, runtime = solve_with_simulated_annealing(matrices)
    elif solver_type == 'hc':
        status, obj_val, x_val, y_val, e_val, z_val, runtime = solve_with_hill_climbing(matrices)
    elif solver_type == 'abc':
        status, obj_val, x_val, y_val, e_val, z_val, runtime = solve_with_abc(matrices)
    elif solver_type == 'exhaustive':
        status, obj_val, x_val, y_val, e_val, z_val, runtime = solve_with_exhaustive_search(matrices)
    elif solver_type == 'all':
        print(f"\n{'='*70}")
        print(f"COMPARING ALL SOLVERS")
        print(f"{'='*70}")
        
        results = []
        
        # Solve with OR-tools
        print(f"\n[1/5] Running OR-Tools SCIP...")
        status1, obj1, x1, y1, e1, z1, rt1 = solve_with_ortools_custom(matrices)
        if x1 is not None:
            alleles1, muts1 = extract_solution(matrices, x1, y1)
            results.append(('OR-Tools SCIP', obj1, alleles1, len(muts1), rt1))
        
        # Solve with Simulated Annealing
        print(f"\n[2/5] Running Simulated Annealing...")
        status2, obj2, x2, y2, e2, z2, rt2 = solve_with_simulated_annealing(matrices)
        if x2 is not None:
            alleles2, muts2 = extract_solution(matrices, x2, y2)
            results.append(('Simulated Annealing', obj2, alleles2, len(muts2), rt2))
        
        # Solve with Hill Climbing
        print(f"\n[3/5] Running Hill Climbing...")
        status3, obj3, x3, y3, e3, z3, rt3 = solve_with_hill_climbing(matrices)
        if x3 is not None:
            alleles3, muts3 = extract_solution(matrices, x3, y3)
            results.append(('Hill Climbing', obj3, alleles3, len(muts3), rt3))
        
        # Solve with ABC
        print(f"\n[4/5] Running Artificial Bee Colony...")
        status4, obj4, x4, y4, e4, z4, rt4 = solve_with_abc(matrices)
        if x4 is not None:
            alleles4, muts4 = extract_solution(matrices, x4, y4)
            results.append(('ABC', obj4, alleles4, len(muts4), rt4))
        
        # Solve with Exhaustive Search
        print(f"\n[5/5] Running Exhaustive Search...")
        status5, obj5, x5, y5, e5, z5, rt5 = solve_with_exhaustive_search(matrices)
        if x5 is not None:
            alleles5, muts5 = extract_solution(matrices, x5, y5)
            results.append(('Exhaustive Search', obj5, alleles5, len(muts5), rt5))

        
        # Compare results
        print(f"\n{'='*70}")
        print(f"SOLVER COMPARISON")
        print(f"{'='*70}")
        
        print(f"\n{'Solver':<25} {'Objective':>12} {'Runtime':>10} {'Alleles':<20} {'Novel':>6}")
        print(f"{'-'*85}")
        for name, obj, alleles, n_novel, runtime in results:
        
            allele_str = str(alleles) if len(str(alleles)) < 20 else str(alleles)[:17] + "..."
            print(f"{name:<25} {obj:>12.4f} {runtime:>9.2f}s {allele_str:<20} {n_novel:>6}")
        # Find best solution
        best_idx = np.argmin([r[1] for r in results])
        best_name = results[best_idx][0]
        best_obj = results[best_idx][1]
        
        print(f"\n✓ Best solution: {best_name} (objective: {best_obj:.4f})")
        
        # Check agreement
        all_alleles = [set(r[2]) for r in results]
        if len(set(tuple(sorted(a)) for a in all_alleles)) == 1:
            print(f"✓ All solvers found the SAME alleles!")
        else:
            print(f"⚠ Solvers found DIFFERENT alleles")
            unique_solutions = {}
            for name, obj, alleles, n_novel, runtime in results:
                key = tuple(sorted(alleles))
                if key not in unique_solutions:
                    unique_solutions[key] = []
                unique_solutions[key].append((name, obj, runtime))
            
            print(f"\nUnique solutions found:")
            for i, (alleles_key, solvers) in enumerate(unique_solutions.items(), 1):
                print(f"  Solution {i}: {list(alleles_key)}")
                for solver, obj, runtime in solvers:
                    print(f"    - {solver}: {obj:.4f} ({runtime:.2f}s)")
        
        # Use the best solution
        if best_name == 'OR-Tools SCIP':
            status, obj_val, x_val, y_val, e_val, z_val, runtime = status1, obj1, x1, y1, e1, z1, rt1
        elif best_name == 'Simulated Annealing':
            status, obj_val, x_val, y_val, e_val, z_val, runtime = status2, obj2, x2, y2, e2, z2, rt2
        elif best_name == 'Hill Climbing':
            status, obj_val, x_val, y_val, e_val, z_val, runtime = status3, obj3, x3, y3, e3, z3, rt3
        else:  # ABC
            if best_name == 'ABC':
                status, obj_val, x_val, y_val, e_val, z_val, runtime = status4, obj4, x4, y4, e4, z4, rt4
            else:  # Exhaustive Search
                status, obj_val, x_val, y_val, e_val, z_val, runtime = status5, obj5, x5, y5, e5, z5, rt5
        
        print(f"\n→ Using {best_name} solution")
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. Use 'ortools', 'sa', 'hc', 'abc', 'exhaustive', or 'all'")
    
    if x_val is None:
        print("\n✗ Solver failed!")
        return
    
    # Validate solution
    print(f"\n{'='*70}")
    print(f"VALIDATING SOLUTION")
    print(f"{'='*70}")
    validation = validate_solution(matrices, x_val, y_val, e_val, z_val)
    
    if validation["valid"]:
        print(f"✓ Solution is VALID - all constraints satisfied")
    else:
        print(f"✗ Solution has VIOLATIONS:")
        for v in validation["violations"]:
            print(f"    - {v}")
    
    # Extract human-readable solution
    selected_alleles, novel_muts = extract_solution(matrices, x_val, y_val)
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULT")
    print(f"{'='*70}")
    
    # Count alleles
    from collections import Counter
    allele_counts = Counter(selected_alleles)
    
    print(f"\nGenotype: {gene_name} ", end="")
    genotype_str = "/".join([f"*{a}" for a in sorted(allele_counts.keys())])
    print(genotype_str)
    
    print(f"\nSelected alleles:")
    for allele, count in sorted(allele_counts.items()):
        print(f"  *{allele:15} × {count}")
    
    if novel_muts:
        print(f"\nNovel mutations detected:")
        for mut in novel_muts:
            print(f"  {mut}")
    else:
        print(f"\n✓ No novel mutations (all explained by known alleles)")
    
    print(f"\nScore: {obj_val:.4f}")
    
    print(f"\n{'='*70}")
    print(f"SUCCESS - Allele calling completed using custom ILP solver!")
    print(f"{'='*70}\n")
    
    return matrices, x_val, y_val, e_val, z_val, selected_alleles


if __name__ == "__main__":
    import sys
    
    # Parse command-line arguments
    solver_type = 'all'  # default
    bam_file = "../data/NA07000.bam"
    gene_name = "CYP2D6"
    profile_name = "illumina"
    
    if len(sys.argv) >= 5:
        bam_file = sys.argv[1]
        gene_name = sys.argv[2]
        profile_name = sys.argv[3]
        solver_type = sys.argv[4]
    elif len(sys.argv) >= 2:
        solver_type = sys.argv[1]
    
    if solver_type not in ['ortools', 'sa', 'hc', 'abc', 'all']:
        print(f"Usage: {sys.argv[0]} [BAM_FILE GENE PROFILE SOLVER]")
        print(f"   or: {sys.argv[0]} [SOLVER]")
        print(f"\nSOLVER options:")
        print(f"  ortools: Use OR-Tools SCIP solver (exact)")
        print(f"  sa:      Use Simulated Annealing (heuristic)")
        print(f"  hc:      Use Hill Climbing (local search)")
        print(f"  abc:     Use Artificial Bee Colony (swarm intelligence)")
        print(f"  all:     Compare all solvers (default)")
        sys.exit(1)
    
    print(f"Selected solver: {solver_type.upper()}")
    
    try:
        result = main(solver_type, bam_file, gene_name, profile_name)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
