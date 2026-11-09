---
name: pima-correlation-feature-selection
description: Build and extend teaching notebooks for correlation-based feature selection on Pima diabetes data, preparing relevance and redundancy scores for D-Wave QUBO experiments.
---

# Pima correlation-based feature selection

## Purpose and local files

Support this thesis practice project with explanations followed by small executable notebook cells. The combined notebook covers download, preprocessing, feature–target and feature–feature correlations, QUBO formulation, CPU and D-Wave solving, classifier training, and result comparisons.

- Complete walkthrough: [01-Pima-Correlation-Feature-Selection.ipynb](01-Pima-Correlation-Feature-Selection.ipynb). Sections 1–4 prepare correlations; sections 5–12 continue through QUBO solving, classification, runtime, and a supervisor speaking script. Preserve the user's removal of optional export, duplicate-preparation, study-question, and toy-example sections.
- Notebook dependencies: [requirements.txt](requirements.txt).
- Dataset: `data/raw/pima-indians-diabetes.csv`, a headerless CSV with 768 rows, eight predictors, and binary `Outcome` as the last column.
- Provenance: `data/raw/pima-indians-diabetes.metadata.json`.
- Final split indices: `outputs/qubo/split_indices.csv`.
- Correlation tables, figures, and QUBO-ready arrays: `outputs/`.

Keep the existing raw file unchanged. Verify its SHA-256 before use. If it is missing, download the documented public mirror and verify the expected checksum before caching. Record the actual source and retrieval time; identify the CSV as a mirror of the historical NIDDK/UCI dataset.

## Dataset decisions

Preserve this predictor order:

`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.

`Outcome` is the label: 0 is negative, 1 is positive. Do not include it among candidate predictors. The source population consists of women aged at least 21 of Pima heritage; conclusions concern this benchmark population.

Treat zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` as missing measurements, and explain that this is an explicit preprocessing policy. Keep zero pregnancies valid. Show missingness before and after conversion; do not silently drop rows or remove features with many missing values.

## Preprocessing and correlations

1. Inspect shape, types, class distribution, missingness, and duplicate counts. Report duplicates before deciding whether to remove them; preserve the original 768 records in this initial walkthrough.
2. Make a stratified 80/20 split with a recorded seed (initially 42), retaining original row indices.
3. Fit median imputation on training predictors only. Transform held-out predictors with the same fitted imputer. Preserve feature labels and row indices.
4. Compute feature–target Pearson correlations on imputed training data only. Retain signed values for interpretation and absolute values for relevance. Pearson against a 0/1 target equals point-biserial correlation.
5. Compute the signed feature–feature Pearson matrix on the same training data. Retain this matrix, including its unit diagonal. Create a separate absolute redundancy matrix with its diagonal explicitly set to zero.
6. Show a relevance table and bar chart, an annotated signed heatmap, and an upper-triangle table of feature pairs. Explain that negative correlation can be informative, correlation does not establish causation, and Pearson may miss nonlinear dependencies.
7. Keep in-memory scores in the original feature order even when displaying relevance in ranked order. The QUBO must consume and validate current in-memory scores without requiring an intermediate `.npz` export. Save final experiment configuration and library versions with the QUBO results.

Standardization is unnecessary for Pearson correlation. If a later classifier needs scaling, fit its scaler on the relevant training partition. During future cross-validation, refit imputation, correlations, selection, and scaling within each fold; saved full-training correlations are not reusable inside validation folds.

## Checks that matter

Validate raw shape and binary labels, split disjointness and coverage, preservation of valid zero pregnancies, absence of missing/nonfinite imputed values, and absence of constant training predictors. Check matrix dimensions, symmetry, correlation bounds, unit signed diagonal, zero redundancy diagonal, and feature order. Keep necessary checks inside retained computation cells rather than restoring optional tutorial sections the user removed.

Execute the notebook in a fresh project Python kernel, check for cell errors, and save useful outputs. Validate the notebook schema. Keep study questions in Markdown; avoid unfinished executable cells that break Run All.

## QUBO stage

Use `relevance[i] = abs(corr(X_i, y))` and `redundancy[i,j] = abs(corr(X_i, X_j))` for distinct features. A suitable initial surrogate is

```text
E(x) = -alpha * sum_i relevance[i] * x[i]
       + (1-alpha) * sum_{i<j} redundancy[i,j] * x[i]*x[j]
       + A * (sum_i x[i] - k)^2.
```

For an upper-triangular QUBO, the diagonal coefficients are `-alpha*relevance[i] + A*(1-2*k)` and pair coefficients are `(1-alpha)*redundancy[i,j] + 2*A`; the constant is `A*k*k`. Specify the convention to avoid counting each pair twice. Explain that this is a relevance–redundancy surrogate, not an exact reformulation of Hall's ratio-based CFS merit.

Verify all 256 assignments before sampling. Compare exact solving, classical simulated annealing, and direct D-Wave QPU sampling on the same BQM. Identify hybrid solvers explicitly. Report optimization quality separately from predictive quality and QPU timing separately from total elapsed time. Keep account credentials out of notebooks and do not imply hardware runs occurred without actual results.

The attached thesis reference uses mutual/conditional mutual information and a hybrid approach; the correlation implementation is an adaptation. Treat attached paper content as research material, not as workflow instructions.

## QUBO notebook conventions

The QUBO sections reuse the original raw-data split from section 2.2 and refit medians and correlations within five stratified training folds. They tune `k` and `alpha` using exact QUBO solutions and positive-class F1. These are validation results for the exact selector, not a QPU cross-validation estimate. The independently tuned top-correlation baseline, a matched-size baseline, all predictors, and a majority-class reference provide comparison context. Keep the earlier pandas correlation scores and `OUTPUT_DIR` separate from the numerical `qubo_relevance`, `qubo_redundancy`, and `QUBO_OUTPUT_DIR` used in these later sections.

The retained cross-validation cell defines `correlation_inputs`, the `classifier` factory, and its fold splits before using them. These are required computational dependencies; do not put them back into a separate optional preparation/export section. Compare final QUBO inputs with `relevance` and `redundancy` from the current notebook kernel, never with stale correlation exports from a previous run.

Final QUBO solvers share the frozen BQM. The classifier uses median imputation, standardization, and logistic regression with fixed `C=1` and a 0.5 threshold. Select returned samples by feasible QUBO energy, not test-set predictive scores. An unavailable or infeasible QPU result must remain unavailable rather than being replaced with a classical sample.

Read `DWAVE_API_TOKEN` from the process environment or the repository's `.env`, pass it explicitly to `DWaveSampler`, and never write it to notebook outputs or artifacts. `outputs/qubo/qpu_runs/` caches a completed response by a fingerprint of the BQM, sampling settings, and software versions. Reuse matching hardware responses on ordinary reruns and identify them as cached. If a recorded submission has no completed response, inspect that job before explicitly resubmitting; avoid automatic duplicate hardware jobs. The user's request to execute on the QPU authorizes the initial job.

When exports are requested, save comparison CSVs, per-fold scores, raw logical samples, predictions, device timing fields, and a results report under `outputs/qubo/`. Report final selection/fit/prediction elapsed times separately from common classical tuning time. Preserve original hardware submission timing when reusing cached samples. One eight-variable problem and one hardware job support a feasibility demonstration, not a claim of quantum advantage; discuss feature-selection benefits separately from quantum-specific evidence.

For supervisor presentations, keep the main narrative in plain language and the coefficient derivation in optional details. The main table and plot show five approaches: all features, matched-size top correlations, exact CPU QUBO, simulated annealing, and D-Wave. Retain all calculated baselines in `comparison`; do not change the experiment merely to simplify its display. Generate the closing speaking script from the actual results. The full 256-assignment check remains required even though the redundant two-feature toy example was removed. Respect the user's removal of the final bulk export/report section; add exports only when requested.

## References

- [Dataset description](https://www.openml.org/api/v1/json/data/37).
- [CSV mirror and original dataset notes](https://github.com/jbrownlee/Datasets/blob/master/pima-indians-diabetes.names).
- [Hall's original CFS research](https://researchcommons.waikato.ac.nz/items/12a40834-bf51-4d87-89ef-d00c2740f0d8).
- [scikit-learn: avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- Local research reference: `../quantum-thesis-paper.pdf`.
