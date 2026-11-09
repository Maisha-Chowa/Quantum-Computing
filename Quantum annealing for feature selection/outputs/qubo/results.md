# Pima correlation-QUBO experiment results

Dataset: 768 records; training=614, test=154; seed=42.
Training-only five-fold selector tuning chose k=4, alpha=0.5; penalty A=1.199366.
Classifier: median imputation + standardization + logistic regression (C=1; threshold=0.5).
Objective: Pearson relevance–redundancy surrogate, not the reference paper's MI/CMI objective.

| Method | Accuracy | Positive-class F1 | Features | Final-run wall time (s) |
|---|---:|---:|---:|---:|
| All features | 0.7078 | 0.5455 | 8 | 0.003504 |
| Top correlation (CV k) | 0.7078 | 0.5455 | 8 | 0.005990 |
| Top correlation (matched k) | 0.7143 | 0.5686 | 4 | 0.005689 |
| Exact QUBO (CPU) | 0.6948 | 0.5347 | 4 | 0.005979 |
| Simulated annealing (CPU) | 0.6948 | 0.5347 | 4 | 0.067701 |
| D-Wave QPU | 0.6948 | 0.5347 | 4 | 5.028757 |
| Majority-class reference | 0.6494 | 0.0000 | 0 | 0.000179 |

Parameter-search wall time: 0.833909 s, reported separately from final runs.
Final-run time includes selection, fit, and prediction; imports, file I/O, plots, and verification are excluded.

## Recorded hardware result

Solver: Advantage_system6;graph_id=01dae5a273; problem ID: efc6b6f3-0166-4eda-876d-19006bc1fb8c.
Reads: 1000; physical qubits: 12.
Selected features: Pregnancies, Glucose, BMI, DiabetesPedigreeFunction.
Feature reduction versus all predictors: 50.0%.
Accuracy difference versus all features: -0.0130.
F1 difference versus all features: -0.0108.
Best feasible QUBO energy gap from the exact optimum: 5.5511151e-17.
Observed exact-optimum-read fraction: 0.0370.
Same subset as the exact solver: True.
Original QPU final-run wall time: 5.028757 s; CPU exact-QUBO final-run wall time: 0.005979 s.
This notebook loaded cached hardware samples: True.
Reported QPU access time: 0.139398 s; this excludes cloud/client overhead.
The QPU did not provide an end-to-end runtime advantage over exact CPU selection in this run.

## Interpretation

Smaller subsets and competitive predictive scores are benefits of feature selection; they are not automatically quantum-specific benefits.
This eight-variable experiment demonstrates implementation and, when completed, real hardware execution. It does not establish quantum advantage.
One reused held-out split and one QPU job provide descriptive results only; do not infer significance, clinical validity, or scalability.

References: [D-Wave sampling](https://docs.dwavequantum.com/en/latest/ocean/api_ref_system/samplers.html), [D-Wave timing](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html), [preventing leakage](https://scikit-learn.org/stable/common_pitfalls.html).
