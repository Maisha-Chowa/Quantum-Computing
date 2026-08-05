"""Solve a small weighted Max-Cut problem as a QUBO with D-Wave Ocean."""

from collections import defaultdict

import dimod
from dwave.samplers import SimulatedAnnealingSampler


# Each tuple is (first vertex, second vertex, edge weight).
# The graph is a square A-B-C-D-A with one diagonal A-C.
EDGES = [
    ("A", "B", 1.0),
    ("B", "C", 1.0),
    ("C", "D", 1.0),
    ("D", "A", 1.0),
    ("A", "C", 1.0),
]


def build_max_cut_qubo(edges):
    """Return Q for min(-cut_weight).

    For an edge (u, v) with weight w, the edge is cut when
    x_u + x_v - 2*x_u*x_v = 1. Negating this expression gives
    the minimization coefficients -w, -w, and +2w.
    """
    qubo = defaultdict(float)

    for u, v, weight in edges:
        qubo[(u, u)] -= weight
        qubo[(v, v)] -= weight
        qubo[(u, v)] += 2.0 * weight

    return dict(qubo)


def cut_details(sample, edges):
    """Calculate the two partitions, cut edges, and total cut weight."""
    left = sorted(vertex for vertex, side in sample.items() if side == 0)
    right = sorted(vertex for vertex, side in sample.items() if side == 1)
    cut_edges = []
    cut_weight = 0.0

    for u, v, weight in edges:
        if sample[u] != sample[v]:
            cut_edges.append((u, v, weight))
            cut_weight += weight

    return left, right, cut_edges, cut_weight


def main():
    qubo = build_max_cut_qubo(EDGES)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

    print("QUBO coefficients:")
    for variables, coefficient in sorted(qubo.items()):
        print(f"  {variables}: {coefficient:+g}")

    # This is a local, classical simulation; no D-Wave API token is required.
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(
        bqm,
        num_reads=100,
        num_sweeps=1000,
        seed=42,
    ).aggregate()

    best = sampleset.first
    # Convert NumPy integer values returned by Ocean to ordinary Python ints.
    sample = {vertex: int(side) for vertex, side in best.sample.items()}
    left, right, cut_edges, cut_weight = cut_details(sample, EDGES)

    print("\nBest simulated-annealing solution:")
    print(f"  Assignment: {sample}")
    print(f"  Partition 0: {left}")
    print(f"  Partition 1: {right}")
    print(f"  Cut edges: {cut_edges}")
    print(f"  Cut weight: {cut_weight:g}")
    print(f"  QUBO energy: {best.energy:g}")

    # Exhaustive search is practical here because there are only 2^4 states.
    exact = dimod.ExactSolver().sample(bqm)
    exact_weight = -exact.first.energy
    print(f"\nExact maximum cut weight: {exact_weight:g}")

    assert abs(best.energy + cut_weight) < 1e-9
    assert abs(cut_weight - exact_weight) < 1e-9
    print("Verification passed: simulated annealing found a maximum cut.")


if __name__ == "__main__":
    main()
