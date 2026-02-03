# CYP2D6 Allele Calling Results - Solver Comparison

## Test Case: NA07000.bam

### Input
- **BAM file**: ../data/NA07000.bam
- **Gene**: CYP2D6
- **Profile**: illumina
- **Average coverage**: 39.41x

### Problem Statistics
- **Candidate alleles**: 7 (*1, *184, *2, *34, *35, *39, *9)
- **Functional mutations observed**: 4
- **Matrix dimensions**: 8 mutations × 14 allele copies
- **CN configuration**: diploid (2 copies of *1 structure)

---

## Solver Results

### 1. OR-Tools SCIP (Exact ILP Solver)
**Type**: Mixed Integer Programming (exact)  
**Result**: ✓ OPTIMAL solution found

- **Objective**: 1.4879
- **Genotype**: *35/*9
- **Novel mutations**: 0
- **Coverage fit error**: 1.4879
- **Novel penalty**: 0.0000
- **Solution time**: < 1 second

**Breakdown**:
- Coverage errors only (no penalties)
- All mutations explained by known alleles
- Guarantees global optimum

---

### 2. Simulated Annealing (Heuristic)
**Type**: Metaheuristic optimization (approximate)  
**Result**: ✓ Found optimal solution

- **Objective**: 1.4879
- **Genotype**: *35/*9  
- **Novel mutations**: 0
- **Coverage fit error**: 1.4879
- **Novel penalty**: 0.0000

**Algorithm parameters**:
- Max iterations: 10,000
- Initial temperature: 100.0
- Cooling rate: 0.995
- Final temperature: 0.000000

**Performance statistics**:
- Accepted moves: 2,152 / 10,000 (21.5%)
- Improvements found: 7
- Initial objective: 26.0586
- Final objective: 1.4879
- **Improvement**: 94.3% from initial solution

---

## Comparison

### Agreement
✓ **Both solvers found IDENTICAL alleles**: *35/*9  
✓ **Same objective value**: 1.4879 (difference: 0.0000)  
✓ **Both solutions valid**: All constraints satisfied

### Performance Characteristics

| Metric | OR-Tools SCIP | Simulated Annealing |
|--------|--------------|---------------------|
| **Solution Quality** | Optimal (proven) | Optimal (found) |
| **Speed** | Very fast (< 1s) | Fast (~3s) |
| **Guarantee** | Global optimum | Best effort |
| **Memory** | Moderate | Low |
| **Scalability** | Good for medium problems | Excellent for large problems |

### When to Use Each Solver

**OR-Tools SCIP (Exact)**:
- ✓ When you need guaranteed optimal solutions
- ✓ For small to medium-sized problems (< 1000 variables)
- ✓ When runtime is not critical
- ✓ For verification and validation

**Simulated Annealing (Heuristic)**:
- ✓ For very large problems (> 10,000 variables)
- ✓ When good-enough solutions are acceptable
- ✓ For real-time or time-constrained scenarios
- ✓ When memory is limited
- ✓ For problems where exact solvers struggle

---

## Validation Against Original Aldy

**Original Aldy Result**: *9/*35 (same as *35/*9)

✓ All three methods agree:
1. Original Aldy solver
2. OR-Tools SCIP (via matrix extraction)
3. Simulated Annealing (via matrix extraction)

---

## Conclusion

The matrix extraction module successfully enables:
1. **Exact solving** with commercial/open-source ILP solvers
2. **Heuristic solving** with custom metaheuristics
3. **Easy experimentation** with different solution approaches

In this test case:
- Both solvers found the correct genotype (*35/*9)
- SA matched the exact solver's optimal solution
- Matrix formulation is correct and solver-agnostic
- Results validate against original Aldy implementation

The framework allows researchers to:
- Plug in any solver (CPLEX, Gurobi, custom algorithms)
- Develop domain-specific heuristics
- Compare different optimization strategies
- Scale to larger problem instances
