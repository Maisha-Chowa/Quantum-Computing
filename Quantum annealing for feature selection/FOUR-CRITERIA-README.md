# Pima practice: four criteria, four features, Random Forest

Open **03-Pima-Four-Criteria-Random-Forest.ipynb** for the executed walkthrough.
The reusable implementation is **pima_four_criteria.py**. This is a separate
experiment from the earlier logistic-regression notebooks.

From the repository root:

```bash
.venv/bin/python -m pip install -r "Quantum annealing for feature selection/requirements.txt"
.venv/bin/python "Quantum annealing for feature selection/pima_four_criteria.py"
```

Set `DWAVE_API_TOKEN` in your environment or the repository's ignored `.env`.
It is read privately and never included in results. The default command uses a
direct **Advantage2 production QPU**, with 1,000 reads and 20 microseconds of
annealing per read. It discovers an accessible online solver of that generation
and chooses by current load; system numbers and working-graph identifiers do not
order hardware generations. The actual solver identity and graph are recorded.
If no matching QPU is accessible, execution fails explicitly.

Identical reruns reuse the completed hardware response in
`outputs/four_criteria/qpu_runs/`; they preserve its original measured timings.
`--fresh` requests a new hardware job. A pending submission record prevents
automatic resubmission after an interrupted job; inspect its problem ID in Leap
before removing that record. `--classical-only` runs the mathematical checks and
CPU baselines without requesting hardware. Its results have a separate filename.

## The four criteria

The supplied paper, *Quantum Annealing based Feature Selection in Machine
Learning*, uses **mutual information and conditional mutual information**, with
Kerberos hybrid solving (pages 2–4, equations 7 and 11). It does not define three
independent Pearson-based criteria. This practice formulation is an explicit
extension inspired by that work, using correlation scores and a direct QPU.
It is also a surrogate objective rather than Hall's ratio-based CFS merit.

For feature-selection bits `x_i`, compute these scores from **training rows only**:

| Criterion | Definition | Effect when minimizing |
|---|---|---|
| Relevance `r_i` | `abs(corr(X_i, y))` | Reward informative individual features |
| Redundancy `R_ij` | `abs(corr(X_i, X_j))`, zero diagonal | Penalize correlated feature pairs |
| Interaction `S_ij` | `abs(corr(z_i*z_j, y))`, where `z_i` is a training-standardized feature | Reward informative pair products |
| Cost `c_i` | Illustrative acquisition-effort cost divided by the maximum cost | Penalize expensive features |

The product interaction is a simple **statistical proxy**: it can detect a
centered XOR/product relationship that individual Pearson scores miss. It is
neither conditional mutual information nor a measure of pure causal synergy,
and can overlap with main effects. It does not detect every nonlinear interaction.
The Random Forest receives the selected **original columns**, not extra products.

Pima has no measured acquisition costs. Editable example units are
`[1, 4, 1, 2, 5, 1, 2, 1]` in the standard predictor order:
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age. These are practice assumptions, not healthcare prices.
Equal costs would make the cost contribution constant for every size-four subset.

## QUBO and exactly k = 4

With `C = k(k-1)/2 = 6`, minimize

```text
E(x) = -alpha/k * sum_i r_i*x_i
       +beta/C * sum_{i<j} R_ij*x_i*x_j
       -gamma/C * sum_{i<j} S_ij*x_i*x_j
       +delta/k * sum_i c_i*x_i
       +A*(sum_i x_i - 4)^2
```

The fixed practice weights are `alpha=1, beta=0.5, gamma=0.5, delta=0.25`.
Feature sums and pair sums are normalized separately so their numbers of terms
are comparable at `k=4`. They are not tuned on the held-out test set.

The **upper triangular** QUBO entries are

```text
Q_ii = (-alpha*r_i + delta*c_i)/k + A*(1 - 2*k)
Q_ij = (beta*R_ij - gamma*S_ij)/C + 2*A,  i < j
offset = A*k^2
```

Count each pair once. A symmetric display would need half the off-diagonal
coefficients for the same `x.T @ Q @ x` energy. The BQM retains the constant offset.

Write the unpenalized coefficients as `a_i` and `b_ij`. The implementation chooses
`A = max_i(abs(a_i) + sum_{j != i} abs(b_ij)) + 0.1`.
Flipping a bit toward size four decreases the penalty by at least `A`, more than
the maximum possible increase in the unpenalized objective. Thus every ideal
global minimum has four features. Every one of the 256 binary assignments is
checked against the original expression before requesting hardware.
Hardware is probabilistic: only returned size-four samples are eligible, and the
lowest-energy eligible sample is selected without consulting test performance.
The code never repairs an invalid sample or replaces it with an exact CPU answer.

## Data and Random Forest

Use the existing checksum-verified 768-row, eight-feature Pima CSV. Preserve all
rows, convert impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin,
and BMI to missing values, and retain zero pregnancies. Split 80/20 with
stratification and seed 42 (614 training, 154 test rows). Fit median imputation
and all selection scores only on training rows.

Compare all eight features, the top four individual relevance scores, the exact
QUBO subset, and the QPU subset. Each uses an identical Random Forest with
300 trees, `min_samples_leaf=2`, seed 42, and a 0.5 probability threshold. Report
accuracy, balanced accuracy, precision, recall, F1, ROC AUC, and confusion matrices.
These are descriptive results on one previously explored split; they do not
establish clinical utility or quantum speed advantage.

## QPU time and overhead

All raw D-Wave `sampleset.info['timing']` fields are saved in **microseconds**.
Wall clocks use Python `perf_counter()` in seconds; the sample set is resolved
before stopping the timer.

| Measurement | Definition |
|---|---|
| QPU access | `qpu_access_time / 1e6`: programming plus sampling charged to the job |
| QPU sampling | `qpu_sampling_time / 1e6`: all anneal/read/delay cycles |
| Annealing only | `qpu_anneal_time_per_sample * returned_reads / 1e6` |
| D-Wave access overhead | `qpu_access_overhead_time / 1e6`, a device-side initialization field |
| Sample-call wall time | From composite `sample()` through resolved samples, including embedding |
| Sample-call overhead | Sample-call wall time minus reported QPU access |
| Total solver wall time | Solver discovery/setup through resolved samples, before client teardown |
| Total overhead | Total solver wall time minus reported QPU access |

The final residual includes setup, embedding, network, queue waits, decoding, and
device-side access overhead. It does not isolate those components individually.
Do not add D-Wave's access-overhead field to this residual; it is already included.
Data preparation, exact verification, and Random Forest fit/prediction times are
separate from the solver wall interval. Cached runs show **original job timings**.

Results and the hardware job ID are in `outputs/four_criteria/results.json`, with
the classifier comparison in `results_comparison.csv`. The notebook shows the
measured run, criterion tables, QUBO matrix, and a timing breakdown plot.

## References

- [Paper matching the supplied scan](https://arxiv.org/html/2411.19609v1), equations 7 and 11.
- [D-Wave solver identity and selection](https://docs.dwavequantum.com/en/latest/quantum_research/quantum_solvers_intro.html).
- [D-Wave timing definitions](https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html).
- [Pima mirror and original dataset notes](https://github.com/jbrownlee/Datasets/blob/master/pima-indians-diabetes.names).
- [Random Forest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
