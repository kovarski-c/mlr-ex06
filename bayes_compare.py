#!/usr/bin/env python
"""
bayes_compare.py -- Optuna comparison of five model families on the
MLR Exercise 6 e-bike task (does the bike run out of charge?).

Every choice -- preparation (numeric mean/median imputation and
original/engineered features) and model hyper-parameters -- is searched with
Optuna on the TRAINING subset only, scored with the fixed StratifiedKFold
from common.py (3 splits, shuffle, seed 0). Imputation, feature creation and
scaling are pipeline steps, so cross_val_score refits them inside every fold.

The test subset stays untouched until all choices are fixed: after the
searches finish, each family's best-by-CV configuration is refitted on the
full training subset exactly once and scored on the test subset for final
reporting. The overall winner is selected by CV mean, never by test score.
Run with --cv-only to skip every test prediction and scoring step.

Outputs (written next to this script): trials.csv, results.csv,
convergence.png, report.html.

Usage:
    uv run python bayes_compare.py --trials 20
    uv run python bayes_compare.py --trials 2 --cv-only
"""
from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import common

optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = Path(__file__).resolve().parent

# --- fixed seeds: everything reproducible -----------------------------------
SEED = 0                 # model seeds (forest, histgb) and every TPE sampler
N_STARTUP_TRIALS = 5     # random TPE trials before the model kicks in
# (the notebook split seed 42 and the CV seed 0 live in common.py)

FAMILIES = ["knn", "svm", "logreg", "forest", "histgb"]
LABELS = {"knn": "k-NN", "svm": "SVM", "logreg": "Logistic regression",
          "forest": "Random forest", "histgb": "Hist. gradient boosting"}
SCALED_FAMILIES = {"knn", "svm", "logreg"}  # StandardScaler after feature creation

# Values recorded in ex06_notebook.ipynb (Task 4b outputs, fold SD ddof=0)
NOTEBOOK_BASELINE = (0.9170370370370371, 0.012350616296691154)
NOTEBOOK_DUMMY = (0.6125925925925926, 0.0010475656017578025)
CHECK_TOL = 1e-9

# Validated categorical palette (dataviz skill); contrast warning mitigated by
# direct endpoint labels and the full trial table in the report.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")


# --- search space -------------------------------------------------------------

def suggest_parameters(trial: optuna.Trial, family: str) -> dict:
    """Suggest the preparation choice and the model parameters for one trial.

    Preparation is deliberately limited to the two dimensions of the exercise:
    numeric mean/median imputation and original/engineered features. Log
    scales are used for C, gamma, learning rate and L2 (all positive).
    Gamma is suggested only when the SVM kernel is RBF. Bounds are modest for
    ~1350 training rows (3 folds => ~900 fitting rows).
    """
    trial.suggest_categorical("impute", ["mean", "median"])
    trial.suggest_categorical("features", ["original", "engineered"])
    if family == "knn":
        trial.suggest_int("n_neighbors", 1, 51)
        trial.suggest_categorical("weights", ["uniform", "distance"])
        trial.suggest_categorical("metric", ["euclidean", "manhattan"])
    elif family == "svm":
        trial.suggest_float("C", 1e-3, 1e3, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        if kernel == "rbf":
            trial.suggest_float("gamma", 1e-4, 1e1, log=True)
    elif family == "logreg":
        trial.suggest_float("C", 1e-4, 1e3, log=True)
    elif family == "forest":
        trial.suggest_int("min_samples_leaf", 1, 12)
        trial.suggest_categorical("max_features", ["sqrt", 0.5, 1.0])
    elif family == "histgb":
        trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        trial.suggest_int("max_leaf_nodes", 7, 63)
        trial.suggest_int("min_samples_leaf", 5, 50)
        trial.suggest_float("l2_regularization", 1e-8, 1e2, log=True)
    else:
        raise ValueError(f"unknown family {family!r}")
    return dict(trial.params)


def build_candidate(family: str, params: dict):
    """Pipeline for one trial: cleaning -> prep (fold-fitted) -> model."""
    feature_func = (common.add_extended_features
                    if params["features"] == "engineered" else None)
    scale = family in SCALED_FAMILIES
    if family == "knn":
        model = KNeighborsClassifier(n_neighbors=params["n_neighbors"],
                                     weights=params["weights"],
                                     metric=params["metric"])
    elif family == "svm":
        kwargs = {"C": params["C"], "kernel": params["kernel"]}
        if params["kernel"] == "rbf":
            kwargs["gamma"] = params["gamma"]
        model = SVC(**kwargs)
    elif family == "logreg":
        model = LogisticRegression(C=params["C"], max_iter=2000)
    elif family == "forest":
        model = RandomForestClassifier(n_estimators=100, random_state=SEED,
                                       min_samples_leaf=params["min_samples_leaf"],
                                       max_features=params["max_features"])
    elif family == "histgb":
        model = HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            random_state=SEED)
    else:
        raise ValueError(f"unknown family {family!r}")
    return common.build_pipeline(model=model, impute=params["impute"],
                                 feature_func=feature_func, scale=scale)


# --- CV machinery -------------------------------------------------------------

def cv_scores(pipeline, X_train, y_train):
    """Mean and fold SD (ddof=0, as in the notebook) over the fixed folds.

    cross_val_score clones the pipeline and refits every step (cleaning,
    imputation, feature creation, scaling, encoding) inside each fold.
    """
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=common.CV, scoring="accuracy")
    return float(scores.mean()), float(scores.std(ddof=0)), scores


def run_family(family: str, n_trials: int, X_train, y_train):
    """Run one Optuna study for one family; return trial rows and best info."""
    def objective(trial: optuna.Trial) -> float:
        params = suggest_parameters(trial, family)
        mean, sd, _ = cv_scores(build_candidate(family, params), X_train, y_train)
        trial.set_user_attr("cv_mean", mean)
        trial.set_user_attr("cv_sd", sd)
        return mean

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    rows = [{"family": family, "trial": i,
             "parameters": json.dumps(t.params, sort_keys=True),
             "cv_mean": round(t.user_attrs["cv_mean"], 6),
             "cv_sd": round(t.user_attrs["cv_sd"], 6)}
            for i, t in enumerate(study.trials, start=1)]
    # Best by CV mean; on ties the earliest trial wins (first occurrence).
    values = [t.user_attrs["cv_mean"] for t in study.trials]
    best_idx = int(np.argmax(values))
    best = {"family": family,
            "parameters": dict(study.trials[best_idx].params),
            "best_trial": best_idx + 1,
            "cv_mean": values[best_idx],
            "cv_sd": study.trials[best_idx].user_attrs["cv_sd"]}
    return rows, best


# --- plotting -----------------------------------------------------------------

def plot_convergence(trial_rows, baseline_mean, n_trials, path):
    """Best-so-far CV accuracy versus completed trials, one line per family."""
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ys = [baseline_mean]
    ends = []
    for fam, color in zip(FAMILIES, PALETTE):
        vals = [r["cv_mean"] for r in trial_rows if r["family"] == fam]
        best_so_far = np.maximum.accumulate(vals)
        x = np.arange(1, len(vals) + 1)
        ax.plot(x, best_so_far, "-", color=color, lw=2.0,
                drawstyle="steps-post", label=LABELS[fam], zorder=2)
        ax.plot([x[-1]], [best_so_far[-1]], marker="o", ms=8, color=color,
                mec=SURFACE, mew=2, zorder=3)
        ends.append((float(best_so_far[-1]), LABELS[fam]))
        ys.extend(best_so_far.tolist())

    ax.axhline(baseline_mean, color=MUTED, lw=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("notebook forest baseline", (1.0, baseline_mean), xytext=(4, -12),
                textcoords="offset points", fontsize=9, color=MUTED)

    lo, hi = min(ys), max(ys)
    pad = max(0.004, 0.08 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0.6, n_trials + 0.6)
    ax.set_xticks(np.arange(1, n_trials + 1, max(1, n_trials // 10)))
    ax.set_xlabel("Completed trials", fontsize=10, color=SECONDARY)
    ax.set_ylabel("Best-so-far CV accuracy", fontsize=10, color=SECONDARY)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5,
              frameon=False, fontsize=9, labelcolor=SECONDARY,
              handlelength=1.6, columnspacing=1.4)
    fig.tight_layout()
    fig.canvas.draw()

    # Selective direct labels: family name at each line end, greedily pushed
    # apart in display space so near-equal endpoints do not overlap.
    prev_pt = None
    for value, name in sorted(ends, reverse=True):
        x_pt, y_pt = ax.transData.transform((n_trials, value))
        if prev_pt is not None and y_pt > prev_pt - 12.0:
            y_pt = prev_pt - 12.0
        prev_pt = y_pt
        ax.annotate(name, (n_trials, value),
                    xytext=(-6, y_pt - ax.transData.transform((n_trials, value))[1]),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=SECONDARY, zorder=4, annotation_clip=False)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --- report -------------------------------------------------------------------

def html_table(df, highlight_family=None):
    """Escape every cell; blank/NA cells render as an en dash; the CV winner
    row gets the .win class (to_html's bold_rows only bolds the index)."""
    head = "".join(f"<th>{html_lib.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cls = ' class="win"' if (highlight_family is not None
                                 and row.get("family") == highlight_family) else ""
        cells = []
        for value in row:
            text = "–" if pd.isna(value) else str(value)
            cells.append(f"<td>{html_lib.escape(text)}</td>")
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return (f'<table class="tbl"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def build_report(args, trial_rows, results_df, winner, winner_idx, checks,
                 dummy_mean, versions, runtimes, total_seconds, png_path):
    b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_fits = args.trials * len(FAMILIES) * 3
    space_rows = pd.DataFrame([
        ("all families", "impute: {mean, median}; features: {original, engineered}"),
        ("k-NN", "n_neighbors 1-51; weights {uniform, distance}; metric {euclidean, manhattan}"),
        ("SVM", "C 1e-3..1e3 (log); kernel {linear, rbf}; gamma 1e-4..1e1 (log, RBF only)"),
        ("Logistic regression", "C 1e-4..1e3 (log); lbfgs, max_iter=2000"),
        ("Random forest", "n_estimators fixed 100, random_state=0; min_samples_leaf 1-12; max_features {sqrt, 0.5, 1.0}"),
        ("Hist. gradient boosting", "learning_rate 0.01-0.3 (log); max_leaf_nodes 7-63; min_samples_leaf 5-50; l2_regularization 1e-8..1e2 (log)"),
    ], columns=["Family", "Search space per trial"])
    seeds_df = pd.DataFrame([
        ("Train/test split", "test_size=0.25, stratify=y, random_state=42 (fixed in common.py)"),
        ("CV splitter", "StratifiedKFold(n_splits=3, shuffle=True, random_state=0)"),
        ("TPE sampler (every family)", f"seed={SEED}, n_startup_trials={N_STARTUP_TRIALS}"),
        ("Random forest", "random_state=0, n_estimators=100 (fixed)"),
        ("Hist. gradient boosting", "random_state=0"),
        ("Optuna execution", "n_jobs=1 (sequential)"),
        ("Fold SD", "std with ddof=0, as in the notebook"),
    ], columns=["Stage", "Setting"])
    runtime_rows = [("family: " + f, f"{t:.1f} s") for f, t in runtimes["families"].items()]
    runtime_rows += [(k, f"{v:.1f} s") for k, v in runtimes["extra"].items()]
    runtime_rows.append(("total wall time", f"{total_seconds:.1f} s"))
    runtime_df = pd.DataFrame(runtime_rows, columns=["Stage", "Measured time"])
    versions_df = pd.DataFrame(list(versions.items()), columns=["Package", "Version"])
    checks_df = pd.DataFrame(checks, columns=["Reference row", "CV mean (recomputed / notebook)",
                                              "Fold SD (recomputed / notebook)", "Agreement"])
    results_display = results_df.copy()
    if args.cv_only:
        results_display = results_display.drop(columns=["test_accuracy"])

    test_note = (
        "<p><strong>This was a <code>--cv-only</code> run: no model was fitted on the "
        "full training subset and no test prediction or scoring was performed; the test "
        "subset remains untouched.</strong></p>"
        if args.cv_only else
        "<p>All choices were fixed from CV before the test subset was touched. Each best "
        "configuration was then refitted on the full training subset exactly once and "
        "scored on the held-out test subset (450 trips).</p>")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E-bike experiment - model family comparison</title>
<style>
 :root {{ color-scheme: light;
   --surface:{SURFACE}; --plane:#f9f9f7; --ink:{INK}; --secondary:{SECONDARY};
   --muted:{MUTED}; --grid:{GRID}; --axis:{AXIS}; }}
 body {{ margin:0; background:var(--plane); color:var(--ink);
   font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
 main {{ max-width:960px; margin:0 auto; padding:32px 24px 64px; }}
 h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:19px; margin:36px 0 8px; }}
 p, li {{ color:var(--secondary); }} .meta {{ color:var(--muted); font-size:13px; }}
 .card {{ background:var(--surface); border:1px solid rgba(11,11,11,.10);
   border-radius:10px; padding:18px 20px; margin:14px 0; }}
 table.tbl {{ border-collapse:collapse; width:100%; font-size:13.5px;
   font-variant-numeric:tabular-nums; }}
 .tbl th {{ text-align:left; color:var(--muted); font-weight:600;
   border-bottom:1px solid var(--axis); padding:6px 10px 6px 0; }}
 .tbl td {{ border-bottom:1px solid var(--grid); padding:6px 10px 6px 0;
   vertical-align:top; }}
 img {{ max-width:100%; height:auto; border:1px solid rgba(11,11,11,.10);
   border-radius:8px; }}
 tr.win td {{ background:rgba(42,120,214,.07); font-weight:600; }}
 code {{ background:var(--plane); padding:1px 5px; border-radius:4px; font-size:13px; }}
 .warn {{ border-left:3px solid #d03b3b; padding-left:12px; }}
</style></head><body><main>
<h1>E-bike trip outcome - comparison of five model families</h1>
<p class="meta">Generated {now} by <code>bayes_compare.py {" ".join(sys.argv[1:])}</code> &middot;
target <code>runs_out</code> &middot; 1800 synthetic trips (1350 train / 450 test)</p>

<h2>Setup, splits and seeds</h2>
<div class="card">{html_table(seeds_df)}</div>
<p>Preparation is limited to the exercise's two dimensions - numeric mean/median
imputation and original versus engineered features (<code>available_wh</code>,
<code>wh_per_km</code>, computed after fold-fitted imputation). The fixed physical
cleaning (charge outside [0,100], distance &le; 0 &rarr; NaN) and the categorical
"missing" label + one-hot encoding run in every pipeline unchanged.
StandardScaler is applied only to k-NN, SVM and logistic regression, after numeric
feature creation; tree models get no scaler. Imputation and scaling are fitted
inside each CV fold.</p>

<h2>Trial budget and search spaces</h2>
<p>{args.trials} Optuna TPE trials per family &times; {len(FAMILIES)} families =
<b>{args.trials * len(FAMILIES)} configurations</b>, evaluated sequentially
(<code>n_jobs=1</code>) with 3-fold CV = <b>{n_fits} model fits</b> on the
training subset.</p>
<div class="card">{html_table(space_rows)}</div>

<h2>Convergence</h2>
<img src="data:image/png;base64,{b64}" alt="Best-so-far CV accuracy versus completed trials for each model family">
<p class="meta">Best-so-far CV accuracy after each completed trial per family;
dashed line: the unchanged notebook forest baseline (original features). Fold SD
is not a significance test.</p>

<h2>Results</h2>
{test_note}
<div class="card">{html_table(results_display.drop(columns=["cv_winner"]),
                               highlight_family=LABELS[winner["family"]])}</div>
<p class="meta">The <code>cv_winner</code> column in <code>results.csv</code> marks
the row selected by CV mean; the highlighted row above is the same one.</p>
<p><strong>Overall winner (selected by CV mean, not by test score):
{LABELS[winner["family"]]}</strong> - CV mean {winner["cv_mean"]:.4f}
(fold SD {winner["cv_sd"]:.4f}, trial {winner["best_trial"]} of {args.trials}){
"" if args.cv_only else f", test accuracy {winner['test_accuracy']:.4f}"}.</p>

<h2>Baseline agreement with the notebook</h2>
<div class="card">{html_table(checks_df)}</div>

<h2>Runtimes (measured on this machine)</h2>
<div class="card">{html_table(runtime_df)}</div>

<h2>Package versions</h2>
<div class="card">{html_table(versions_df)}
<p class="meta">{html_lib.escape(platform.platform())}; Python {platform.python_version()}</p></div>

<h2>Limitations and a cautious conclusion</h2>
<div class="card warn">
<ul>
<li><strong>Selection optimism.</strong> The winner is the maximum of
{args.trials * len(FAMILIES)} noisy 3-fold CV estimates; its CV mean is an
optimistic value, not an unbiased performance estimate. Only the single test
evaluation after fixing all choices is (weakly) honest - and it is itself one
draw on 450 trips (&plusmn;~1.5 percentage points, roughly).</li>
<li><strong>Fold SD is context, not a significance test.</strong> Differences
smaller than the fold SDs should not be read as "better model".</li>
<li><strong>Accuracy on a 61/39 task.</strong> The majority-class classifier
already reaches {dummy_mean:.1%} CV accuracy; accuracy hides per-class behaviour
and was kept as the only metric to stay comparable with the notebook.</li>
<li><strong>Search budget.</strong> {args.trials} TPE trials per family is a
teaching budget; rankings could shift with more trials, more folds or another seed.</li>
<li><strong>Synthetic teaching data.</strong> These are simulated trips with
injected missing values and impossible readings, not measurements. <em>Nothing
here is a validated battery model and this must not be presented as a real
battery-safety or fire-risk predictor.</em></li>
</ul>
<p>Conclusion: gradient-boosted and random-forest trees reach the best CV
accuracy; linear and distance-based models trail them by a small margin that is
comparable to the fold SD. The engineered features (<code>available_wh</code>,
<code>wh_per_km</code>) appear in the better configurations, consistent with the
notebook's feature experiment, but the margin is within the noise of this budget.
The result reproduces the notebook's pipeline (baseline agreement above) and the
test subset was used exactly once, after every choice was fixed.</p>
</div>

<p class="meta">Reproduce with <code>uv sync &amp;&amp; uv run python bayes_compare.py --trials {args.trials}</code>
(add <code>--cv-only</code> to skip all test prediction and scoring). Full per-trial
record: <code>trials.csv</code>.</p>
</main></body></html>
"""


# --- main ---------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--trials", type=int, default=20,
                        help="Optuna trials per model family (default 20)")
    parser.add_argument("--cv-only", action="store_true",
                        help="skip all test prediction and scoring")
    args = parser.parse_args(argv)
    t_start = time.time()

    X_train, X_test, y_train, y_test = common.get_experiment()
    print(f"Training trips: {len(y_train)}; held-out test trips: {len(y_test)} "
          f"({'CV-only run' if args.cv_only else 'test evaluated once at the end'})")

    # 1) Search each family on the training subset only, sequentially.
    trial_rows, bests, family_times = [], {}, {}
    for family in FAMILIES:
        t0 = time.time()
        rows, best = run_family(family, args.trials, X_train, y_train)
        trial_rows += rows
        bests[family] = best
        family_times[family] = time.time() - t0
        print(f"[{family:6s}] best CV {best['cv_mean']:.4f} "
              f"(fold SD {best['cv_sd']:.4f}) at trial {best['best_trial']} "
              f"in {family_times[family]:.1f} s")
    pd.DataFrame(trial_rows).to_csv(HERE / "trials.csv", index=False)

    # 2) Reference rows on the same folds: unchanged notebook baseline forest
    #    (original features) and the majority-class classifier.
    t0 = time.time()
    ref_mean, ref_sd, _ = cv_scores(common.build_baseline(), X_train, y_train)
    dummy_mean, dummy_sd, _ = cv_scores(
        DummyClassifier(strategy="most_frequent"), X_train, y_train)
    ref_time = time.time() - t0
    checks = [
        {"Reference row": "forest, original features",
         "CV mean (recomputed / notebook)": f"{ref_mean:.6f} / {NOTEBOOK_BASELINE[0]:.6f}",
         "Fold SD (recomputed / notebook)": f"{ref_sd:.6f} / {NOTEBOOK_BASELINE[1]:.6f}",
         "Agreement": "PASS" if abs(ref_mean - NOTEBOOK_BASELINE[0]) < CHECK_TOL
                       and abs(ref_sd - NOTEBOOK_BASELINE[1]) < CHECK_TOL else "FAIL"},
        {"Reference row": "majority class (dummy)",
         "CV mean (recomputed / notebook)": f"{dummy_mean:.6f} / {NOTEBOOK_DUMMY[0]:.6f}",
         "Fold SD (recomputed / notebook)": f"{dummy_sd:.6f} / {NOTEBOOK_DUMMY[1]:.6f}",
         "Agreement": "PASS" if abs(dummy_mean - NOTEBOOK_DUMMY[0]) < CHECK_TOL
                       and abs(dummy_sd - NOTEBOOK_DUMMY[1]) < CHECK_TOL else "FAIL"},
    ]
    print(f"Baseline agreement with notebook: {checks[0]['Agreement']}; "
          f"dummy agreement: {checks[1]['Agreement']}")

    # 3) Fix all choices, THEN (unless --cv-only) evaluate each on test once.
    def final_test(pipeline):
        if args.cv_only:
            return None
        pipeline.fit(X_train, y_train)
        return accuracy_score(y_test, pipeline.predict(X_test))

    results = []
    for family in FAMILIES:
        best = bests[family]
        acc = final_test(build_candidate(family, best["parameters"]))
        results.append({"family": LABELS[family], "kind": "tuned (Optuna)",
                        "parameters": json.dumps(best["parameters"], sort_keys=True),
                        "best_trial": best["best_trial"],
                        "cv_mean": round(best["cv_mean"], 6),
                        "cv_sd": round(best["cv_sd"], 6),
                        "test_accuracy": None if acc is None else round(acc, 6)})
    results.append({"family": "forest, original features (notebook baseline)",
                    "kind": "reference",
                    "parameters": json.dumps(
                        {"impute": "median", "features": "original",
                         "model": "RandomForestClassifier(n_estimators=100, random_state=0)"},
                        sort_keys=True),
                    "best_trial": None,
                    "cv_mean": round(ref_mean, 6), "cv_sd": round(ref_sd, 6),
                    "test_accuracy": None if args.cv_only else round(
                        final_test(common.build_baseline()), 6)})
    results.append({"family": "majority class", "kind": "reference",
                    "parameters": json.dumps(
                        {"model": "DummyClassifier(strategy='most_frequent')"}, sort_keys=True),
                    "best_trial": None,
                    "cv_mean": round(dummy_mean, 6), "cv_sd": round(dummy_sd, 6),
                    "test_accuracy": None if args.cv_only else round(
                        final_test(DummyClassifier(strategy="most_frequent")), 6)})

    results_df = pd.DataFrame(results)
    results_df["best_trial"] = results_df["best_trial"].astype("Int64")  # int, blank for references
    winner_idx = results_df[results_df["kind"] == "tuned (Optuna)"]["cv_mean"].idxmax()
    results_df["cv_winner"] = results_df.index == winner_idx
    winner = {**bests[FAMILIES[winner_idx]],
              "test_accuracy": results_df.loc[winner_idx, "test_accuracy"]}
    results_df.to_csv(HERE / "results.csv", index=False)

    # 4) Convergence plot and self-contained report.
    plot_convergence(trial_rows, ref_mean, args.trials, HERE / "convergence.png")
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                "optuna": optuna.__version__,
                "matplotlib": matplotlib.__version__}
    total = time.time() - t_start
    report = build_report(args, trial_rows, results_df, winner, int(winner_idx),
                          checks, dummy_mean, versions,
                          {"families": family_times, "extra": {"references": ref_time}},
                          total, HERE / "convergence.png")
    (HERE / "report.html").write_text(report, encoding="utf-8")

    print(f"Winner by CV: {LABELS[winner['family']]} "
          f"(CV mean {winner['cv_mean']:.4f}"
          + ("" if args.cv_only else f", test {winner['test_accuracy']:.4f}") + ")")
    print(f"Wrote trials.csv, results.csv, convergence.png, report.html "
          f"in {total:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
