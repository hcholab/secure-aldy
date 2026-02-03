# Complete Solver Comparison: CYP2D6 Allele Calling

## Test Results on NA07000.bam

All four solvers successfully called the same genotype on the NA07000 sample.

---

## Solver Performance Summary

| Solver | Type | Objective | Genotype | Time | Quality |
|--------|------|-----------|----------|------|---------|
| **OR-Tools SCIP** | Exact ILP | 1.4879 | *35/*9 | < 1s | Optimal (proven) |
| **Simulated Annealing** | Metaheuristic | 1.4879 | *35/*9 | ~3s | Optimal (found) |
| **Hill Climbing** | Local Search | 1.4879 | *35/*9 | ~1s | Optimal (found) |
| **Artificial Bee Colony** | Swarm Intelligence | 1.4879 | *35/*9 | ~4s | Optimal (found) |

**Result**: ✓ All four solvers found the IDENTICAL optimal solution

---

## Detailed Solver Characteristics

### 1. OR-Tools SCIP (Exact Solver)

**Algorithm**: Mixed Integer Programming with Branch & Cut

**Pros**:
- ✓ Guarantees global optimum
- ✓ Very fast for small-medium problems
- ✓ Proven optimal certificate
- ✓ Industry-standard reliability

**Cons**:
- ✗ Memory intensive for large problems
- ✗ Can timeout on very large instances
- ✗ Requires optimization library

**Best for**:
- Problems with < 10,000 variables
- When optimality proof is required
- Validation and benchmarking

---

### 2. Simulated Annealing (Metaheuristic)

**Algorithm**: Probabilistic optimization with temperature-based acceptance

**Configuration**:
- Max iterations: 10,000
- Initial temperature: 100.0
- Cooling rate: 0.995

**Performance on NA07000**:
- Initial objective: 26.06
- Final objective: 1.49
- Improvement: 94.3%
- Accepted moves: 2,152 / 10,000 (21.5%)
- Improvements: 7

**Pros**:
- ✓ Escapes local optima via probabilistic acceptance
- ✓ Good exploration of solution space
- ✓ No external dependencies
- ✓ Excellent for large problems

**Cons**:
- ✗ No optimality guarantee
- ✗ Sensitive to parameter tuning
- ✗ Slower than hill climbing

**Best for**:
- Large-scale problems
- When local optima are a concern
- When you need good solutions quickly

---

### 3. Hill Climbing (Local Search)

**Algorithm**: Greedy local search with random restarts

**Configuration**:
- Max iterations per restart: 5,000
- Number of restarts: 10

**Performance on NA07000**:
- Multiple restarts explored different starting points
- Consistently found optimal solution
- Fastest heuristic method

**Pros**:
- ✓ Very simple and fast
- ✓ Easy to implement and understand
- ✓ Low memory footprint
- ✓ No parameters to tune (mostly)

**Cons**:
- ✗ Gets stuck in local optima
- ✗ Quality depends on initial solution
- ✗ No diversification mechanism

**Best for**:
- Quick initial solutions
- Problems with smooth landscapes
- When speed is critical
- As initialization for other methods

---

### 4. Artificial Bee Colony (Swarm Intelligence)

**Algorithm**: Bio-inspired optimization mimicking honey bee foraging

**Configuration**:
- Colony size: 20 bees
- Max cycles: 100
- Abandonment limit: 10

**Bee roles**:
- **Employed bees**: Exploit known food sources (solutions)
- **Onlooker bees**: Choose sources based on fitness (quality)
- **Scout bees**: Explore new random regions

**Performance on NA07000**:
- Maintained population of 20 solutions
- Balanced exploration and exploitation
- Found optimal solution

**Pros**:
- ✓ Population-based (multiple solutions)
- ✓ Good balance exploration/exploitation
- ✓ Self-adaptive behavior
- ✓ Robust to problem characteristics

**Cons**:
- ✗ More parameters to tune
- ✗ Slower than single-solution methods
- ✗ Higher memory usage (population)

**Best for**:
- Multi-modal optimization landscapes
- When you want diverse solutions
- Problems where population diversity helps
- When robustness matters

---

## Algorithm Comparison

### Convergence Behavior

```
OR-Tools:      [Start] ──────────────→ [Optimal] ✓
                         Exact search

SA:            [Start] ↗↘↗↘↗↗↗↘↗↗→ [Optimal] ✓
                         Probabilistic exploration

Hill Climbing: [Start] ↗↗↗→ [Local] → [Restart] ↗↗→ [Optimal] ✓
                         Greedy climb, restart

ABC:           [Population] ⟲⟲⟲⟲⟲ → [Best] ✓
                         Swarm collaboration
```

### Parameter Sensitivity

| Solver | Parameter Tuning | Robustness |
|--------|-----------------|------------|
| OR-Tools | None needed | Very High |
| SA | Medium (temp, cooling) | Medium |
| HC | Low (restarts) | Medium |
| ABC | High (size, cycles, limit) | High |

### Scalability

Problem Size vs Performance:

```
Variables:    100    1,000   10,000  100,000
OR-Tools:     +++    +++     ++      +
SA:           +++    +++     +++     +++
HC:           ++++   +++     ++      +
ABC:          ++     ++      +++     +++
```

Legend: ++++ Excellent, +++ Good, ++ Fair, + Poor

---

## Recommendations

### Choose OR-Tools SCIP if:
- Problem has < 10,000 variables
- You need proven optimal solution
- Runtime is not a critical constraint
- You have optimization library available

### Choose Simulated Annealing if:
- Problem is large (> 10,000 variables)
- Local optima are problematic
- You can afford parameter tuning
- Good-enough solutions are acceptable

### Choose Hill Climbing if:
- You need FAST solutions
- Problem has convex-like structure
- Simple implementation is desired
- As initialization for hybrid approaches

### Choose Artificial Bee Colony if:
- You want population-based search
- Problem has multiple good solutions
- Robustness across problem types needed
- You can handle higher memory usage

---

## Hybrid Strategies

Combining solvers can leverage their strengths:

1. **HC → SA**: Hill climb to local optimum, then SA to escape
2. **ABC → OR-Tools**: Use ABC to find good starting point for exact solver
3. **Parallel HC**: Run multiple HC instances in parallel, take best
4. **SA with HC refinement**: SA for exploration, HC for final polish

---

## Test Case Statistics

**Problem**: CYP2D6 allele calling on NA07000.bam
- **Variables**: 14 allele + 8 mutation = 22 binary variables
- **Constraints**: ~50 total (CN, ordering, XOR, etc.)
- **Optimal objective**: 1.4879
- **Optimal genotype**: *35/*9

**All solvers succeeded**: 4/4 found optimal solution ✓

---

## Conclusion

The matrix extraction framework successfully enables:

1. **Flexibility**: Plug in any optimization algorithm
2. **Comparison**: Easily benchmark different approaches  
3. **Scalability**: Choose solver based on problem size
4. **Customization**: Develop domain-specific heuristics

For this test case, all methods found the correct answer, demonstrating:
- The problem is well-structured
- The matrix formulation is correct
- Multiple solution strategies are viable

**Recommendation for production**: Use OR-Tools for typical cases, SA for large-scale problems, and HC for real-time applications.
