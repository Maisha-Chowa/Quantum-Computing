# Max-Cut as a QUBO with D-Wave Ocean

This example solves a small weighted Max-Cut problem using Ocean's local
simulated-annealing sampler. It does not require a D-Wave API token.

For an interactive walkthrough, open `max-cut-qubo.ipynb` in Jupyter.

For the same signed weighted example as the brute-force tutorial running on a
real D-Wave QPU, open `MaxCut-QUBO-DWave-Ocean.ipynb`.

For an explanation of embedding chains, chain length, chain breaks, and their
relationship to gate-based systems, read
`Qubit-Chains-and-Chain-Breaks.md`.

## 1. Max-Cut problem

Max-Cut divides the vertices of a graph into two groups and maximizes the total
weight of edges whose endpoints are in different groups.

Assign one binary variable to every vertex:

- `x_v = 0`: vertex `v` is in partition 0
- `x_v = 1`: vertex `v` is in partition 1

For an edge `(u, v)` of weight `w`, the expression

```text
w * (x_u + x_v - 2*x_u*x_v)
```

is `w` when the endpoints are separated and `0` when they are together.
Max-Cut is therefore

```text
maximize sum_edges w * (x_u + x_v - 2*x_u*x_v)
```

Ocean minimizes energy, so negate the objective:

```text
minimize sum_edges [-w*x_u - w*x_v + 2*w*x_u*x_v]
```

This is already a QUBO. No constraint penalty is necessary because every
binary assignment defines a valid partition.

For every edge `(u, v, w)`, add these QUBO coefficients:

```text
Q[u, u] += -w
Q[v, v] += -w
Q[u, v] += 2w
```

The QUBO energy is the negative of the cut weight, so the lowest energy gives
the largest cut.

## 2. Install Ocean

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install dwave-ocean-sdk
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

## 3. Run

```bash
python solve-qubo.py
```

The example graph is a square with one diagonal. Its maximum cut has weight
`4`: vertices `A` and `C` are on one side and `B` and `D` are on the other
(or the complementary assignment).

The program solves it twice:

1. `SimulatedAnnealingSampler` performs local classical simulated annealing.
2. `dimod.ExactSolver` checks every one of the `2^4 = 16` assignments and
   verifies the answer.

`SimulatedAnnealingSampler` is a classical heuristic, not a simulation of a
quantum computer. It is nevertheless useful for learning and testing the same
QUBO/BQM formulation accepted by other Ocean samplers.

## 4. Try another graph

Edit `EDGES` in `solve-qubo.py`. For example:

```python
EDGES = [
    ("A", "B", 2.0),
    ("A", "C", 1.0),
    ("B", "C", 3.0),
]
```

Weights should normally be nonnegative for the standard weighted Max-Cut
problem. For larger graphs, increase `num_reads` and optionally `num_sweeps`.

ssh -T git@github-personal

git remote set-url origin git@github-personal:Maisha-Chowa/Quantum-Computing

git remote set-url origin git@github-personal:Maisha-Chowa/qubo-learning

https://github.com/anikaislam16/qubo-learning
