# Major Allele Matrix Extraction Module

This module (`aldy/major_matrix.py`) extracts the major star-allele detection problem into standard Integer Linear Programming (ILP) matrix form, allowing you to use custom solvers and heuristics.

## Overview

The major star-allele detection problem is formulated as an ILP that selects alleles explaining observed sequencing coverage while minimizing coverage errors and penalizing novel mutations.

## Matrix Formulation

### Problem Data

**Matrices:**
- `A` (m × n): Allele-mutation incidence matrix. `A[i,j] = 1` if allele copy j contains mutation i
- `B` (k × n): CN configuration matrix. `B[t,j] = 1` if allele copy j has CN configuration t
- `c` (m,): Normalized coverage vector
- `d` (k,): CN configuration requirements

**Decision Variables:**
- `x` ∈ {0,1}^n: Binary allele copy selection
- `y` ∈ {0,1}^m: Binary novel mutation indicators
- `e` ∈ ℝ^m: Coverage error vector
- `z` ∈ {0,1}: Any novel mutation indicator

### Constraints

1. **Coverage equality**: `A·x + y + e = c`
2. **CN configuration**: `B·x = d`
3. **Ordering**: `x[j+1] ≤ x[j]` for same allele (prevents duplicates)
4. **XOR logic**: Each mutation explained by allele OR marked novel (not both)
5. **Novel indicator**: `z ≥ y[i]` for all i, and `z ≤ Σy[i]`
6. **One novel per position**: `Σ(y[i] : pos(i)=p) ≤ 1` for each position p

### Objective

Minimize: `||e||₁ + λ·z + 0.1·Σy[i]`

Where:
- `||e||₁`: Sum of absolute coverage errors
- `λ·z`: Large penalty (≈21) if ANY novel mutation
- `0.1·Σy[i]`: Small penalty per novel mutation

## Usage

### Basic Usage

```python
from aldy.major_matrix import build_major_matrices, extract_solution, validate_solution
from aldy.major import _filter_alleles
from aldy.gene import Gene, Mutation
from aldy.profile import Profile
from aldy.coverage import Coverage
from aldy.solutions import CNSolution
from aldy.common import script_path

# Load gene and prepare data (same as Aldy does internally)
gene = Gene(script_path("aldy.resources.genes/cyp2d6.yml"))
profile = Profile("illumina")
cn_solution = CNSolution(gene, 0, ["1", "1"])

# Get coverage from BAM file (see Aldy docs)
coverage = Coverage(...)

# Filter alleles and get functional mutations
alleles, filtered_cov = _filter_alleles(gene, coverage, cn_solution)
func_muts = {
    Mutation(*m)
    for m in gene.mutations
    if gene.is_functional(m) and filtered_cov[Mutation(*m)] > 0
}

# Extract matrices
matrices = build_major_matrices(gene, filtered_cov, cn_solution, alleles, func_muts)

# Now use matrices with your solver
# matrices.A, matrices.B, matrices.c, matrices.d, etc.
```

### Solving with Custom Solver

```python
# Example: Simple greedy heuristic
import numpy as np

x = np.zeros(matrices.n_alleles)
y = np.zeros(matrices.n_mutations)

# Select alleles to satisfy CN constraints
for t in range(matrices.n_cn_configs):
    required = int(matrices.d[t])
    candidates = [j for j in range(matrices.n_alleles) if matrices.B[t, j] == 1]
    for j in candidates[:required]:
        x[j] = 1

# Compute errors
e = matrices.c - matrices.A @ x - y
z = 0

# Extract human-readable solution
selected_alleles, novel_muts = extract_solution(matrices, x, y)
print(f"Selected alleles: {selected_alleles}")
```

### Solving with OR-Tools (MIP)

```python
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver('SCIP')

# Create variables
x = [solver.BoolVar(f'x_{j}') for j in range(matrices.n_alleles)]
y = [solver.BoolVar(f'y_{i}') for i in range(matrices.n_mutations)]
z = solver.BoolVar('z')
e = [solver.NumVar(-solver.infinity(), solver.infinity(), f'e_{i}')
     for i in range(matrices.n_mutations)]

# Add constraints: A·x + y + e = c
for i in range(matrices.n_mutations):
    constraint = solver.Constraint(float(matrices.c[i]), float(matrices.c[i]))
    for j in range(matrices.n_alleles):
        if matrices.A[i, j] != 0:
            constraint.SetCoefficient(x[j], float(matrices.A[i, j]))
    constraint.SetCoefficient(y[i], 1)
    constraint.SetCoefficient(e[i], 1)

# Add CN constraints: B·x = d
for t in range(matrices.n_cn_configs):
    constraint = solver.Constraint(float(matrices.d[t]), float(matrices.d[t]))
    for j in range(matrices.n_alleles):
        if matrices.B[t, j] != 0:
            constraint.SetCoefficient(x[j], float(matrices.B[t, j]))

# Add objective and solve...
```

### Solution Validation

```python
# Validate solution satisfies all constraints
validation = validate_solution(matrices, x, y, e, z)

print(f"Valid: {validation['valid']}")
print(f"Objective: {validation['objective']:.4f}")
print(f"Coverage fit: {validation['coverage_fit']:.4f}")
print(f"Novel penalty: {validation['novel_penalty']:.4f}")

if not validation['valid']:
    print(f"Violations: {validation['violations']}")
```

## API Reference

### `build_major_matrices(gene, coverage, cn_solution, allele_dict, func_muts)`

Extracts ILP matrices from Aldy's gene/coverage data.

**Returns:** `MajorILPMatrices` object containing:
- `A`: Allele-mutation incidence matrix
- `B`: CN configuration matrix  
- `c`: Normalized coverage vector
- `d`: CN configuration counts
- `P_indices`: Ordering constraint pairs
- `lambda_novel`: Novel mutation penalty
- Metadata for interpreting solutions

### `extract_solution(matrices, x, y)`

Converts binary variable values to human-readable allele names and mutations.

**Returns:** `(selected_alleles, novel_mutations)`

### `validate_solution(matrices, x, y, e, z)`

Validates solution and computes objective value.

**Returns:** Dict with `valid`, `violations`, `objective`, and component scores

## Testing

Run tests to verify correctness:

```bash
pytest aldy/tests/test_major_matrix.py -v
```

Tests validate that matrix extraction produces identical results to the original Aldy solver.

## Example

See `example_matrix_usage.py` for a complete working example with a custom greedy heuristic.

```bash
python example_matrix_usage.py
```

## Compatible Solvers

The matrix form can be used with any ILP/MIP solver:

- **Commercial**: CPLEX, Gurobi, Xpress
- **Open Source**: SCIP, CBC, GLPK, HiGHS
- **OR-Tools**: Google's optimization toolkit
- **Custom**: Branch-and-bound, cutting planes, column generation
- **Heuristics**: Genetic algorithms, simulated annealing, local search

## Notes

- The XOR constraints (mutation must be explained by allele OR novel) require careful linearization in MIP solvers
- The absolute value in the objective (`||e||₁`) requires introducing auxiliary variables
- Ordering constraints prevent symmetric solutions (e.g., copy 0 and 2 selected but not copy 1)
- Reference mutations (marked with "_") represent positions matching the reference genome

## Citation

If you use this module, please cite the Aldy paper:

```
Aldy: A tool for allelic decomposition and exact genotyping of highly polymorphic genes
Nature Communications, 2018
```
