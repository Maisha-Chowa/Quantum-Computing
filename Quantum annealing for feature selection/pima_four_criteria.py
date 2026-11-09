"""Four-criterion correlation QUBO, fixed k=4, direct Advantage2 QPU, Random Forest.

Run with the repository's .venv Python. See 03-Pima-Four-Criteria-Random-Forest.ipynb.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter

import dimod
from dotenv import dotenv_values
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
DATA_SHA256 = "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af"
K = 4
SEED = 42
NUM_READS = 1000
ANNEALING_TIME_US = 20.0
# Illustrative acquisition-effort units, NOT measured healthcare prices.
FEATURE_COSTS = dict(zip(FEATURES, [1, 4, 1, 2, 5, 1, 2, 1]))
WEIGHTS = {"relevance": 1.0, "redundancy": 0.5, "interaction": 0.5, "cost": 0.25}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_json(path, value):
    """Atomic replacement protects a completed response from partial writes."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=json_default) + "\n")
    temporary.replace(path)


def load_data():
    path = HERE / "data/raw/pima-indians-diabetes.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset checksum differs from the documented Pima mirror.")
    raw = pd.read_csv(path, header=None, names=FEATURES + ["Outcome"])
    if raw.shape != (768, 9) or raw.Outcome.value_counts().to_dict() != {0: 500, 1: 268}:
        raise ValueError("Unexpected Pima shape or labels.")
    raw.index.name = "record_id"
    X = raw[FEATURES].copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    split = train_test_split(X, raw.Outcome.astype(int), test_size=0.2,
                             stratify=raw.Outcome, random_state=SEED)
    train_X, test_X, train_y, test_y = split
    assert set(train_X.index).isdisjoint(test_X.index)
    assert set(train_X.index) | set(test_X.index) == set(raw.index)
    assert (X.Pregnancies == 0).sum() == (raw.Pregnancies == 0).sum()
    return raw, train_X, test_X, train_y, test_y


def abs_corr(a, b):
    """Pearson magnitude; define the score of a constant product as zero."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(abs(a @ b) / denominator, 0, 1)) if denominator > 1e-14 else 0.0


def compute_scores(train_X, train_y, costs=None):
    if list(train_X.columns) != FEATURES or not train_X.index.equals(train_y.index):
        raise ValueError("Training feature order or target alignment is incorrect.")
    if train_X.isna().all().any() or train_y.nunique() != 2:
        raise ValueError("Require observed training values and both target classes.")
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(train_X)
    if not np.isfinite(values).all() or np.any(values.std(axis=0) == 0):
        raise ValueError("Require finite, varying training predictors.")
    z = (values - values.mean(axis=0)) / values.std(axis=0)
    relevance = np.array([abs_corr(values[:, i], train_y) for i in range(8)])
    redundancy = np.abs(np.corrcoef(values, rowvar=False))
    np.fill_diagonal(redundancy, 0)
    interaction = np.zeros((8, 8))
    for i in range(8):
        for j in range(i + 1, 8):
            interaction[i, j] = interaction[j, i] = abs_corr(z[:, i] * z[:, j], train_y)
    costs = FEATURE_COSTS if costs is None else costs
    if set(costs) != set(FEATURES):
        raise ValueError("Provide one cost for every predictor.")
    raw_cost = np.array([costs[name] for name in FEATURES], dtype=float)
    if not np.isfinite(raw_cost).all() or np.any(raw_cost < 0):
        raise ValueError("Feature costs must be finite and nonnegative.")
    normalized_cost = raw_cost / raw_cost.max() if raw_cost.max() > 0 else raw_cost.copy()
    return {"relevance": relevance, "redundancy": redundancy, "interaction": interaction,
            "cost": normalized_cost, "raw_cost": raw_cost, "medians": imputer.statistics_}


def build_qubo(scores, weights=None):
    weights = WEIGHTS.copy() if weights is None else dict(weights)
    if set(weights) != set(WEIGHTS) or any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Provide four finite nonnegative criterion weights.")
    # Divide feature sums by k and pair sums by C(k,2) to balance their scales.
    pairs = K * (K - 1) / 2
    linear = (-weights["relevance"] * scores["relevance"]
              + weights["cost"] * scores["cost"]) / K
    pairwise = (weights["redundancy"] * scores["redundancy"]
                - weights["interaction"] * scores["interaction"]) / pairs
    # Any bit flip changes the base objective by at most this incident bound.
    # A flip toward k reduces A*(sum(x)-k)^2 by at least A, so all global
    # minima have k ones. This guarantee concerns the ideal QUBO, not QPU noise.
    penalty = float(np.max(np.abs(linear) + np.abs(pairwise).sum(axis=1)) + 0.1)
    qubo = {(i, i): float(linear[i] + penalty * (1 - 2 * K)) for i in range(8)}
    qubo.update({(i, j): float(pairwise[i, j] + 2 * penalty)
                 for i in range(8) for j in range(i + 1, 8)})
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=penalty * K**2)
    return bqm, qubo, penalty


def energy_terms(bits, scores, penalty, weights=None):
    weights = WEIGHTS if weights is None else weights
    bits = np.asarray(bits, dtype=float)
    pair_mask = np.triu(np.outer(bits, bits), k=1)
    pairs = K * (K - 1) / 2
    return {
        "relevance": float(-weights["relevance"] * (scores["relevance"] @ bits) / K),
        "redundancy": float(weights["redundancy"] * np.sum(scores["redundancy"] * pair_mask) / pairs),
        "interaction": float(-weights["interaction"] * np.sum(scores["interaction"] * pair_mask) / pairs),
        "cost": float(weights["cost"] * (scores["cost"] @ bits) / K),
        "cardinality_penalty": float(penalty * (bits.sum() - K)**2),
    }


def verify_qubo(bqm, scores, penalty):
    """Check the independent formula and every ground state's cardinality."""
    bits = np.array(list(product([0, 1], repeat=8)))
    direct = np.array([sum(energy_terms(x, scores, penalty).values()) for x in bits])
    np.testing.assert_allclose(bqm.energies((bits, list(range(8)))), direct, atol=1e-10)
    ground = np.isclose(direct, direct.min(), atol=1e-10, rtol=0)
    assert np.all(bits[ground].sum(axis=1) == K)
    return float(direct.min())


def best_feasible(sampleset):
    for row in sampleset.data(fields=["sample", "energy"], sorted_by="energy"):
        if sum(row.sample.values()) == K:
            return np.array([row.sample[i] for i in range(8)], dtype=int)
    return None


def solve_qpu(bqm, output_dir, num_reads=NUM_READS, fresh=False):
    """Direct latest-generation production QPU; reuse completed identical jobs."""
    from dwave.system import DWaveSampler, EmbeddingComposite

    output_dir = Path(output_dir)
    cache_dir = output_dir / "qpu_runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "qubo": [[int(i), int(j), float(v)] for (i, j), v in sorted(bqm.to_qubo()[0].items())],
        "offset": bqm.offset, "features": FEATURES, "k": K, "seed": SEED,
        "num_reads": num_reads, "annealing_time_us": ANNEALING_TIME_US,
        "solver_filter": {"name__regex": "Advantage2_system.*", "topology__type": "zephyr"},
        "versions": {p: version(p) for p in ["dimod", "dwave-system", "dwave-cloud-client"]},
    }
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    completed = cache_dir / f"{fingerprint}.json"
    pending = cache_dir / f"{fingerprint}.submitted.json"
    if pending.exists():
        raise RuntimeError(f"An unfinished submission is recorded at {pending}. Inspect that job before rerunning.")
    if completed.exists() and not fresh:
        saved = json.loads(completed.read_text())
        return dimod.SampleSet.from_serializable(saved["sampleset"]), {**saved["metadata"], "cached": True}
    token = os.getenv("DWAVE_API_TOKEN") or dotenv_values(HERE.parent / ".env").get("DWAVE_API_TOKEN")
    if not token:
        raise RuntimeError("Set DWAVE_API_TOKEN in the environment or repository .env.")

    start = perf_counter()
    # Advantage2 is the current production generation. System numbers and graph
    # IDs are deployment identities, NOT an ordering of hardware generations.
    # Among accessible online production Advantage2 machines choose the least loaded.
    with DWaveSampler(token=token, solver={**spec["solver_filter"], "online": True,
                                         "order_by": "avg_load"},
                      request_timeout=30, polling_timeout=180) as qpu:
        sampler = EmbeddingComposite(qpu, embedding_parameters={"random_seed": SEED})
        setup_s = perf_counter() - start
        metadata = {"status": "submitted", "fingerprint": fingerprint, "specification": spec,
                    "solver_identity": qpu.solver.identity.dict(),
                    "topology": qpu.properties["topology"],
                    "active_qubits": len(qpu.nodelist),
                    "submitted_at_utc": datetime.now(timezone.utc).isoformat()}
        # Record before submission to prevent an accidental duplicate after interruption.
        save_json(pending, metadata)
        job_start = perf_counter()
        try:
            samples = sampler.sample(bqm, num_reads=num_reads, annealing_time=ANNEALING_TIME_US,
                                     auto_scale=True, return_embedding=True,
                                     label="Pima four criteria k=4 Random Forest practice")
            metadata["problem_id"] = samples.wait_id(timeout=60)
            save_json(pending, metadata)
            samples.resolve()  # Required: sample() can return before the remote job finishes.
            call_wall_s = perf_counter() - job_start
            total_wall_s = perf_counter() - start
        except Exception as error:
            metadata["status"] = "interrupted_or_failed"
            # Avoid persisting exception text that might contain connection secrets.
            metadata["error_type"] = type(error).__name__
            save_json(pending, metadata)
            raise RuntimeError(f"QPU job did not complete ({type(error).__name__}); inspect {pending}.") from None
        timing = samples.info.get("timing", {})
        access_us = timing.get("qpu_access_time")
        access_s = access_us / 1e6 if access_us is not None else None
        embedding = samples.info.get("embedding_context", {}).get("embedding", {})
        metadata.update({"status": "completed", "cached": False,
                         "problem_id": samples.info.get("problem_id", metadata["problem_id"]),
                         "timing_us": timing,
                         "setup_wall_s": setup_s, "sample_call_wall_s": call_wall_s,
                         "total_solver_wall_s": total_wall_s, "qpu_access_s": access_s,
                         "sample_call_overhead_s": call_wall_s - access_s if access_s is not None else None,
                         "total_overhead_s": total_wall_s - access_s if access_s is not None else None,
                         "physical_qubits_used": sum(map(len, embedding.values())),
                         "max_chain_length": max(map(len, embedding.values()), default=0)})
        save_json(completed, {"metadata": metadata, "sampleset": samples.to_serializable()})
        pending.unlink()
    return samples, metadata


def evaluate_subset(bits, train_X, test_X, train_y, test_y):
    chosen = [FEATURES[i] for i in np.flatnonzero(bits)]
    model = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                 random_state=SEED, n_jobs=1))
    start = perf_counter()
    model.fit(train_X[chosen], train_y)
    fit_s = perf_counter() - start
    start = perf_counter()
    probabilities = model.predict_proba(test_X[chosen])[:, 1]
    prediction = (probabilities >= 0.5).astype(int)
    predict_s = perf_counter() - start
    return {"num_features": len(chosen), "selected_features": ", ".join(chosen),
            "example_cost_units": sum(FEATURE_COSTS[name] for name in chosen),
            "accuracy": accuracy_score(test_y, prediction),
            "balanced_accuracy": balanced_accuracy_score(test_y, prediction),
            "precision": precision_score(test_y, prediction, zero_division=0),
            "recall": recall_score(test_y, prediction, zero_division=0),
            "f1": f1_score(test_y, prediction, zero_division=0),
            "roc_auc": roc_auc_score(test_y, probabilities),
            "confusion_matrix": confusion_matrix(test_y, prediction).tolist(),
            "fit_s": fit_s, "predict_s": predict_s}


def run_experiment(use_qpu=True, fresh=False, output_dir=None):
    output_dir = Path(output_dir or HERE / "outputs/four_criteria")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, train_X, test_X, train_y, test_y = load_data()
    start = perf_counter()
    scores = compute_scores(train_X, train_y)
    scoring_s = perf_counter() - start
    start = perf_counter()
    bqm, qubo, penalty = build_qubo(scores)
    qubo_build_s = perf_counter() - start
    exact_minimum = verify_qubo(bqm, scores, penalty)
    start = perf_counter()
    exact = dimod.ExactSolver().sample(bqm)
    exact_s = perf_counter() - start
    exact_bits = best_feasible(exact)
    top_bits = np.zeros(8, dtype=int)
    top_bits[np.argsort(-scores["relevance"], kind="stable")[:K]] = 1
    subsets = {"All features": np.ones(8, dtype=int), "Top-4 relevance": top_bits,
               "Exact QUBO (CPU)": exact_bits}
    metadata = {"status": "not_requested", "qpu_access_s": None, "total_overhead_s": None}
    qpu_bits = None
    samples = None
    if use_qpu:
        print("Loading a matching saved QPU job or submitting to an online Advantage2 QPU...", flush=True)
        samples, metadata = solve_qpu(bqm, output_dir, fresh=fresh)
        qpu_bits = best_feasible(samples)
        if qpu_bits is not None:
            subsets["D-Wave QPU"] = qpu_bits
        else:
            metadata["selection_status"] = "no_returned_sample_with_exactly_four_features"
    comparison = pd.DataFrame([{"method": name, **evaluate_subset(bits, train_X, test_X, train_y, test_y)}
                               for name, bits in subsets.items()])
    if samples is not None:
        reads = int(samples.record.num_occurrences.sum())
        counts = np.array([sum(row.sample.values()) for row in samples.data()])
        occurrences = np.array([row.num_occurrences for row in samples.data()])
        energies = np.array([row.energy for row in samples.data()])
        metadata.update({"num_reads_returned": reads,
                         "feasible_read_fraction": float(occurrences[counts == K].sum() / reads),
                         "optimal_read_fraction": float(occurrences[np.isclose(energies, exact_minimum,
                                                                               atol=1e-8, rtol=0)].sum() / reads),
                         "energy_gap": float(bqm.energy(dict(enumerate(qpu_bits))) - exact_minimum)
                         if qpu_bits is not None else None})
        if "chain_break_fraction" in samples.record.dtype.names:
            metadata["weighted_chain_break_fraction"] = float(np.average(samples.record.chain_break_fraction,
                                                                          weights=samples.record.num_occurrences))
    result = {"k": K, "seed": SEED, "weights": WEIGHTS, "costs": FEATURE_COSTS,
              "cost_note": "Illustrative effort units; Pima contains no measured feature costs.",
              "dataset_sha256": DATA_SHA256, "train_rows": len(train_X), "test_rows": len(test_X),
              "split_indices": {"train": train_X.index.tolist(), "test": test_X.index.tolist()},
              "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "dwave-ocean-sdk"]},
              "scores": scores, "qubo_upper_triangle": [[i, j, q] for (i, j), q in qubo.items()],
              "penalty_A": penalty, "offset": bqm.offset, "exact_minimum": exact_minimum,
              "qpu_selected_features": [FEATURES[i] for i in np.flatnonzero(qpu_bits)]
              if qpu_bits is not None else None,
              "selected_energy_terms": energy_terms(qpu_bits if qpu_bits is not None else exact_bits,
                                                    scores, penalty),
              "selected_energy_terms_source": "D-Wave QPU" if qpu_bits is not None else "Exact QUBO (CPU)",
              "timing_cpu_s": {"score_preparation": scoring_s, "qubo_build": qubo_build_s,
                               "exact_solve": exact_s},
              "qpu": metadata, "comparison": comparison.to_dict(orient="records")}
    # Keep hardware/classical results separate so a CPU-only run cannot replace the hardware report.
    stem = "results" if use_qpu else "results_classical"
    save_json(output_dir / f"{stem}.json", result)
    comparison.to_csv(output_dir / f"{stem}_comparison.csv", index=False)
    print(comparison[["method", "num_features", "accuracy", "f1", "roc_auc", "selected_features"]].to_string(index=False))
    if use_qpu:
        print(json.dumps({key: metadata.get(key) for key in ["solver_identity", "problem_id", "cached",
                         "qpu_access_s", "total_solver_wall_s", "total_overhead_s", "energy_gap"]}, indent=2))
    return result, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical-only", action="store_true", help="Verify and run CPU baselines without hardware.")
    parser.add_argument("--fresh", action="store_true", help="Submit a new QPU job even when a completed match is saved.")
    args = parser.parse_args()
    run_experiment(use_qpu=not args.classical_only, fresh=args.fresh)
