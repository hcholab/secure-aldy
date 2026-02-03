#!/usr/bin/env python
# 786
# Aldy source: minor_matrix.py
#   This file is subject to the terms and conditions defined in
#   file 'LICENSE', which is part of this source code package.

"""
Matrix extraction for the minor star-allele optimization problem.

This mirrors the ILP in aldy/minor.py but exports it in a matrix-friendly,
solver-agnostic form (sparse row constraints with variable indices).

Formulation (high level):
    min  ||e||_1 + minor_miss * (sum(|M_a| x_a - sum k^x_{a,m}))
         + minor_add * sum n_{a,m} + phase penalties (optional)
    s.t. coverage balance for each mutation (including reference "_")
         major->minor count consistency
         ordering constraints for allele copies
         compatibility & functional constraints
         per-locus single-mutation constraints
         coverage feasibility constraints

Variables (binary unless noted):
    x_a            = allele copy selected
    k_{a,m}        = keep mutation m on allele a (m in allele definition)
    n_{a,m}        = add mutation m to allele a (m not in definition)
    k^x_{a,m}      = x_a * k_{a,m} (linearized)
    n^x_{a,m}      = x_a * n_{a,m} (linearized)
    e^+_m, e^-_m   = positive/negative error for L1 coverage fit

Constraints are returned as sparse rows:
    A_eq * v = b_eq
    A_ineq * v <= b_ineq
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Any
import numpy as np
from natsort import natsorted

from .gene import Gene, Mutation
from .coverage import Coverage
from .solutions import MajorSolution, SolvedAllele


@dataclass
class MinorILPMatrices:
    """Sparse matrix representation of the minor ILP."""

    var_names: List[str]
    var_types: List[str]
    var_lb: List[float]
    var_ub: List[float]
    objective: Dict[int, float]
    A_eq: List[Dict[int, float]]
    b_eq: List[float]
    A_ineq: List[Dict[int, float]]
    b_ineq: List[float]

    allele_vars: List[Tuple[SolvedAllele, int]]
    mutation_vars: List[Mutation]
    ref_mutations: List[Mutation]


class _VarIndex:
    def __init__(self):
        self.names: List[str] = []
        self.types: List[str] = []
        self.lb: List[float] = []
        self.ub: List[float] = []
        self.index: Dict[str, int] = {}

    def add(
        self,
        name: str,
        vtype: str = "B",
        lb: float = 0.0,
        ub: float = 1.0,
    ) -> int:
        if name in self.index:
            return self.index[name]
        idx = len(self.names)
        self.index[name] = idx
        self.names.append(name)
        self.types.append(vtype)
        self.lb.append(lb)
        self.ub.append(ub)
        return idx


def build_minor_matrices(
    gene: Gene,
    coverage: Coverage,
    major_sol: MajorSolution,
    alleles_list: List[SolvedAllele],
    mutations: Set[Mutation],
) -> MinorILPMatrices:
    """
    Build a sparse matrix representation of the minor ILP.

    :param gene: Gene instance
    :param coverage: Coverage instance
    :param major_sol: Major allele solution
    :param alleles_list: Candidate minor alleles
    :param mutations: Mutations to consider
    :return: MinorILPMatrices with sparse constraints
    """

    # Expand allele copies (same as minor.py)
    alleles: Dict[Tuple[SolvedAllele, int], Set[Mutation]] = {
        (a, 0): set(gene.alleles[a.major].func_muts)
        | set(gene.alleles[a.major].minors[a.minor].neutral_muts)
        for a in alleles_list
    }
    for a, _ in list(alleles):
        max_cn = major_sol.solution[SolvedAllele(gene, a.major, "", a.added, a.missing)]
        for cnt in range(1, max_cn):
            alleles[a, cnt] = alleles[a, 0]

    allele_keys = list(alleles.keys())

    # Build mutation list + reference mutations per locus
    def _normalize_mutations(ms: Set[Any]) -> List[Mutation]:
        out: List[Mutation] = []
        for m in ms:
            if isinstance(m, Mutation):
                out.append(m)
            elif isinstance(m, tuple) and len(m) == 2:
                out.append(Mutation(*m))
        return out

    mut_list = sorted(_normalize_mutations(mutations))
    ref_muts = [Mutation(pos, "_") for pos in sorted({m.pos for m in mut_list})]

    vars = _VarIndex()
    objective: Dict[int, float] = {}

    # Variable indices
    x_idx: Dict[Tuple[SolvedAllele, int], int] = {}
    k_idx: Dict[Tuple[Tuple[SolvedAllele, int], Mutation], int] = {}
    n_idx: Dict[Tuple[Tuple[SolvedAllele, int], Mutation], int] = {}
    k_mul_idx: Dict[Tuple[Tuple[SolvedAllele, int], Mutation], int] = {}
    n_mul_idx: Dict[Tuple[Tuple[SolvedAllele, int], Mutation], int] = {}
    e_pos_idx: Dict[Mutation, int] = {}
    e_neg_idx: Dict[Mutation, int] = {}

    # x variables
    for a in allele_keys:
        x_idx[a] = vars.add(f"x_{a[0].major}_{a[0].minor}_{a[1]}", "B", 0.0, 1.0)

    # k, n variables and linearization variables
    for a in allele_keys:
        for m in alleles[a]:
            k_idx[a, m] = vars.add(
                f"k_{m.pos}_{m.op}_{a[0].major}_{a[0].minor}_{a[1]}", "B", 0.0, 1.0
            )
            k_mul_idx[a, m] = vars.add(
                f"k_mul_{m.pos}_{m.op}_{a[0].major}_{a[0].minor}_{a[1]}", "B"
            )
        for m in mut_list:
            if gene.has_coverage(a[0].major, m.pos) and m not in alleles[a]:
                n_idx[a, m] = vars.add(
                    f"n_{m.pos}_{m.op}_{a[0].major}_{a[0].minor}_{a[1]}", "B"
                )
                n_mul_idx[a, m] = vars.add(
                    f"n_mul_{m.pos}_{m.op}_{a[0].major}_{a[0].minor}_{a[1]}", "B"
                )

    # Error variables (L1 via e_pos/e_neg)
    for m in mut_list + ref_muts:
        e_pos_idx[m] = vars.add(f"e_pos_{m.pos}_{m.op}", "C", 0.0, float("inf"))
        e_neg_idx[m] = vars.add(f"e_neg_{m.pos}_{m.op}", "C", 0.0, float("inf"))
        # Objective: 1 * (e_pos + e_neg)
        objective[e_pos_idx[m]] = 1.0
        objective[e_neg_idx[m]] = 1.0

    A_eq: List[Dict[int, float]] = []
    b_eq: List[float] = []
    A_ineq: List[Dict[int, float]] = []
    b_ineq: List[float] = []

    def add_eq(coeffs: Dict[int, float], rhs: float) -> None:
        A_eq.append(coeffs)
        b_eq.append(rhs)

    def add_ineq(coeffs: Dict[int, float], rhs: float) -> None:
        A_ineq.append(coeffs)
        b_ineq.append(rhs)

    # Major -> minor counts
    for sa, cnt in major_sol.solution.items():
        coeffs = {}
        for (vs, _), xi in x_idx.items():
            if (vs.major, vs.added, vs.missing) == (sa.major, sa.added, sa.missing):
                coeffs[xi] = coeffs.get(xi, 0.0) + 1.0
        add_eq(coeffs, float(cnt))

    # Ordering constraints
    by_minor: Dict[Tuple[str, str, Tuple[Mutation, ...], Tuple[Mutation, ...]], List[Tuple[int, int]]]
    by_minor = {}
    for a in allele_keys:
        key = (a[0].major, a[0].minor, tuple(a[0].added), tuple(a[0].missing))
        by_minor.setdefault(key, []).append((a[1], x_idx[a]))
    for _, items in by_minor.items():
        items = sorted(items, key=lambda x: x[0])
        for i in range(1, len(items)):
            _, xi = items[i]
            _, xprev = items[i - 1]
            add_ineq({xi: 1.0, xprev: -1.0}, 0.0)

    # Linearization constraints for k_mul and n_mul
    for (a, m), km in k_mul_idx.items():
        xi = x_idx[a]
        ki = k_idx[a, m]
        add_ineq({km: 1.0, xi: -1.0}, 0.0)  # km <= x
        add_ineq({km: 1.0, ki: -1.0}, 0.0)  # km <= k
        add_ineq({km: -1.0, xi: 1.0, ki: 1.0}, 1.0)  # km >= x + k - 1

    for (a, m), nm in n_mul_idx.items():
        xi = x_idx[a]
        ni = n_idx[a, m]
        add_ineq({nm: 1.0, xi: -1.0}, 0.0)  # nm <= x
        add_ineq({nm: 1.0, ni: -1.0}, 0.0)  # nm <= n
        add_ineq({nm: -1.0, xi: 1.0, ni: 1.0}, 1.0)  # nm >= x + n - 1

    # Compatibility and functional constraints
    for (a, m), ki in k_idx.items():
        add_ineq({ki: 1.0, x_idx[a]: -1.0}, 0.0)  # k <= x
        if gene.is_functional(m):
            add_ineq({x_idx[a]: 1.0, ki: -1.0}, 0.0)  # k >= x -> x - k <= 0

    for (a, m), ni in n_idx.items():
        add_ineq({ni: 1.0, x_idx[a]: -1.0}, 0.0)  # n <= x

    # Single mutation per locus and per allele
    for pos in {m.pos for m in mut_list}:
        for a in allele_keys:
            km = [k_mul_idx[a, m] for m in alleles[a] if m.pos == pos and m.op[:3] != "ins"]
            nm = [n_mul_idx[a, m] for (aa, m) in n_mul_idx if aa == a and m.pos == pos and m.op[:3] != "ins"]
            if len(km) + len(nm) > 1:
                coeffs = {i: 1.0 for i in km + nm}
                add_ineq(coeffs, 1.0)
            nvars = [n_idx[a, m] for (aa, m) in n_idx if aa == a and m.pos == pos and m.op[:3] != "ins"]
            if len(nvars) > 1:
                add_ineq({i: 1.0 for i in nvars}, 1.0)

    # Coverage equality for mutations
    for m in mut_list:
        coeffs = {}
        for (a, mm), km in k_mul_idx.items():
            if mm == m:
                coeffs[km] = coeffs.get(km, 0.0) + 1.0
        for (a, mm), nm in n_mul_idx.items():
            if mm == m:
                coeffs[nm] = coeffs.get(nm, 0.0) + 1.0
        coeffs[e_pos_idx[m]] = coeffs.get(e_pos_idx[m], 0.0) + 1.0
        coeffs[e_neg_idx[m]] = coeffs.get(e_neg_idx[m], 0.0) - 1.0
        scov = coverage.single_copy(m, major_sol.cn_solution)
        cov = coverage[m] / scov if scov > 0 else 0.0
        add_eq(coeffs, float(cov))

    # Coverage equality for reference mutations
    for ref_m in ref_muts:
        coeffs = {}
        pos = ref_m.pos
        for a in allele_keys:
            if not gene.has_coverage(a[0].major, pos):
                continue
            present = [m for m in alleles[a] if m.pos == pos and m.op[:3] != "ins"]
            if len(present) == 1:
                coeffs[x_idx[a]] = coeffs.get(x_idx[a], 0.0) + 1.0
                coeffs[k_mul_idx[a, present[0]]] = coeffs.get(k_mul_idx[a, present[0]], 0.0) - 1.0
            else:
                coeffs[x_idx[a]] = coeffs.get(x_idx[a], 0.0) + 1.0
                for (aa, mm), nm in n_mul_idx.items():
                    if aa == a and mm.pos == pos and mm.op[:3] != "ins":
                        coeffs[nm] = coeffs.get(nm, 0.0) - 1.0
        coeffs[e_pos_idx[ref_m]] = coeffs.get(e_pos_idx[ref_m], 0.0) + 1.0
        coeffs[e_neg_idx[ref_m]] = coeffs.get(e_neg_idx[ref_m], 0.0) - 1.0
        scov = coverage.single_copy(ref_m, major_sol.cn_solution)
        cov = coverage[ref_m] / scov if scov > 0 else 0.0
        add_eq(coeffs, float(cov))

    # Objective: minor_miss and minor_add (linear)
    miss = coverage.profile.minor_miss
    add = coverage.profile.minor_add
    for a in allele_keys:
        objective[x_idx[a]] = objective.get(x_idx[a], 0.0) + miss * len(alleles[a])
        for m in alleles[a]:
            objective[k_mul_idx[a, m]] = objective.get(k_mul_idx[a, m], 0.0) - miss
    for (a, m), ni in n_idx.items():
        objective[ni] = objective.get(ni, 0.0) + add

    return MinorILPMatrices(
        var_names=vars.names,
        var_types=vars.types,
        var_lb=vars.lb,
        var_ub=vars.ub,
        objective=objective,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        allele_vars=[(a[0], a[1]) for a in allele_keys],
        mutation_vars=mut_list,
        ref_mutations=ref_muts,
    )
