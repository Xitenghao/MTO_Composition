from __future__ import annotations
from common import alfa, alfainv, safe_alfainv, split_frames, RESULT_DIR, FEATURE_COLUMNS
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.dont_write_bytecode = True
SEED = 12345
PARTS = ["Sethylene", "Spropylene", "other"]
RF_BOUNDS = ((0.35, 40, 4, 2, 1), (0.75, 180, 20, 6, 4))
XGB_BOUNDS = (
    (0.35, 60, 3, 1, 0.03, 0.0, 0.6, 0.6, 0.0, 0.0),
    (0.75, 200, 12, 4, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0),
)
RF_DEFAULT = dict(
    n_estimators=100, max_depth=8, min_samples_split=2, min_samples_leaf=1
)
XGB_DEFAULT = dict(n_estimators=100, max_depth=8)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def closure(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected a finite nonnegative composition matrix")
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Composition has zero total")
    return values / totals


def ilr_targets(frame):
    values = frame[PARTS].to_numpy(dtype=float).copy()
    values[values <= 0] = 1e-06
    return alfa(closure(values), a=0, h=True)["aff"]


def inverse_or_error(values, alpha, context):
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        result = alfainv(values, a=alpha, h=True)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(
            f"Invalid inverse alpha prediction: {context}; no clipping applied"
        )
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-08):
        raise ValueError(f"Prediction fails closure: {context}")
    return result


def make_splits():
    data, frames = split_frames()
    return (
        data,
        frames,
        {
            split: {
                "x": frame.loc[:, FEATURE_COLUMNS],
                "y": closure(frame[PARTS].to_numpy(dtype=float)),
                "model_row_indices": frame.index.to_numpy(dtype=int),
            }
            for split, frame in frames.items()
        },
    )


def config_from_position(kind, position):
    p = list(position)
    if kind == "rf":
        names = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
        casts = [int] * 4
    else:
        names = [
            "n_estimators",
            "max_depth",
            "min_child_weight",
            "learning_rate",
            "gamma",
            "colsample_bytree",
            "subsample",
            "reg_alpha",
            "reg_lambda",
        ]
        casts = [int, int, int, float, float, float, float, float, float]
    values = {
        name: int(np.round(value)) if cast is int else float(value)
        for name, value, cast in zip(names, p[1:], casts)
    }
    return {"alpha": float(p[0]), **values}


def learner(kind, params, n_jobs):
    params = params.copy()
    params.pop("alpha", None)
    if kind == "rf":
        return RandomForestRegressor(**params, random_state=SEED, n_jobs=n_jobs)
    return XGBRegressor(
        **params,
        random_state=SEED,
        n_jobs=n_jobs,
        tree_method="hist",
        enable_categorical=True,
        multi_strategy="multi_output_tree",
    )


def objective_factory(kind, frames, out, n_jobs):
    joint = pd.concat([frames["Train"], frames["Valid"]], axis=0)
    x = joint.loc[:, FEATURE_COLUMNS].reset_index(drop=True)
    y = closure(joint[PARTS].to_numpy(dtype=float))
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(x))
    state = {"eval": 0, "best_mse": float("inf")}
    history = out / f"{kind}_objective_history.jsonl"

    def objective(position):
        params = config_from_position(kind, position)
        z = alfa(y, a=params["alpha"], h=True)["aff"]
        scores = []
        started = time.perf_counter()
        reason = None
        for fold_no, (train_idx, valid_idx) in enumerate(folds):
            model = learner(kind, params, n_jobs)
            model.fit(x.iloc[train_idx], z[train_idx])
            prediction, clipped = safe_alfainv(
                model.predict(x.iloc[valid_idx]), a=params["alpha"]
            )
            if clipped or not np.isfinite(prediction).all():
                reason = "invalid inverse transform; candidate rejected"
                scores = []
                break
            scores.append(float(mean_squared_error(y[valid_idx], prediction)))
        mse = float(np.mean(scores)) if scores else 1000000000000.0
        state["eval"] += 1
        record = {
            "eval": state["eval"],
            "mse": mse,
            "fold_mse": scores,
            "params": params,
            "elapsed_seconds": time.perf_counter() - started,
            "reason": reason,
        }
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if mse < state["best_mse"]:
            state["best_mse"] = mse
            write_json(
                out / f"{kind}_params.json",
                {
                    "eval": state["eval"],
                    "best_mse": mse,
                    "best_params": params,
                    "cv_folds": 5,
                    "optimizer_seed": SEED,
                },
            )
            print(f"{kind} eval={state['eval']} best_mse={mse:.10g}", flush=True)
        return mse

    return objective


def tune(kind, frames, out, args):
    import pymetaheuristic

    if pymetaheuristic.__version__ != "7.4.4":
        raise RuntimeError("This runner is pinned to pymetaheuristic 7.4.4")
    output = out / f"{kind}_gmo.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("completed") and saved.get("cv_folds") == 5:
            print(f"Reusing completed five-fold run {output}", flush=True)
            return
        raise FileExistsError(
            f"Unfinished run needs a fresh output directory: {output}"
        )
    objective = objective_factory(kind, frames, out, args.n_jobs)
    low, high = RF_BOUNDS if kind == "rf" else XGB_BOUNDS
    population, steps = (15, 50) if kind == "rf" else (10, 40)
    write_json(output, {"completed": False, "cv_folds": 5})
    started = time.perf_counter()
    result = pymetaheuristic.optimize(
        algorithm="gmo",
        target_function=objective,
        size=population,
        min_values=low,
        max_values=high,
        max_steps=steps,
        epsilon=0.01,
        verbose=True,
        start_init=None,
        target_value=None,
        seed=SEED,
    )
    write_json(
        output,
        {
            "completed": True,
            "cv_folds": 5,
            "best_position": list(result.best_position),
            "best_fitness": float(result.best_fitness),
            "evaluations": int(result.evaluations),
            "steps": int(result.steps),
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_version": pymetaheuristic.__version__,
        },
    )


def geometry_metrics(y, pred):
    if (y <= 0).any() or (pred <= 0).any():
        return {
            "MAE_A": None,
            "RMSE_A": None,
            "R2_A": None,
            "geometry_note": "undefined with nonpositive components; no score-only replacement applied",
        }
    logy, logpred = (np.log(y), np.log(pred))
    clr_y = logy - logy.mean(axis=1, keepdims=True)
    clr_pred = logpred - logpred.mean(axis=1, keepdims=True)
    distances = np.linalg.norm(clr_y - clr_pred, axis=1)
    center_clr = clr_y.mean(axis=0)
    ss_total = float(((clr_y - center_clr) ** 2).sum())
    ss_error = float((distances**2).sum())
    mae, rmse = (float(distances.mean()), float(np.sqrt((distances**2).mean())))
    assert mae <= rmse + 1e-12
    r2 = 1.0 - ss_error / ss_total if ss_total > 0 else None
    if r2 is not None:
        assert np.isclose(r2, 1.0 - len(y) * rmse**2 / ss_total)
    center = np.exp(center_clr)
    center /= center.sum()
    return {
        "MAE_A": mae,
        "RMSE_A": rmse,
        "R2_A": r2,
        "SST_A": ss_total,
        "Aitchison_center": center.tolist(),
        "geometry_note": "strictly positive; no replacement",
    }


def metric_row(name, split, y, pred):
    if not np.isfinite(pred).all():
        raise ValueError(f"Nonfinite prediction: {name} {split}")
    return {
        "Model": name,
        "Split": split,
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred, multioutput="uniform_average")),
        **geometry_metrics(y, pred),
    }


def save_prediction(name, predictions, splits, out, rows):
    arrays = {}
    for split, pred in predictions.items():
        pred = np.asarray(pred, dtype=float)
        y = splits[split]["y"]
        if pred.shape != y.shape:
            raise ValueError(f"Prediction shape mismatch for {name} {split}")
        arrays[f"{split}_truth"] = y
        arrays[f"{split}_prediction"] = pred
        arrays[f"{split}_model_rows"] = splits[split]["model_row_indices"]
        rows.append(metric_row(name, split, y, pred))
    np.savez_compressed(out / "predictions" / f"{name}.npz", **arrays)


def fit_predictions(model, targets, splits, alpha):
    model.fit(splits["Train"]["x"], targets)
    predictions = {}
    for split, values in splits.items():
        pred = model.predict(values["x"])
        predictions[split] = (
            inverse_or_error(pred, alpha, split) if alpha is not None else pred
        )
    return predictions


def load_config(kind, out):
    source = out if (out / f"{kind}_params.json").is_file() else RESULT_DIR
    payload = json.loads((source / f"{kind}_params.json").read_text(encoding="utf-8"))
    if payload.get("cv_folds") != 5:
        raise RuntimeError("Refusing an old or unspecified-fold parameter file")
    complete = json.loads((source / f"{kind}_gmo.json").read_text(encoding="utf-8"))
    if not complete.get("completed") or complete.get("cv_folds") != 5:
        raise RuntimeError("Optimization did not complete")
    return payload["best_params"]


def score(frames, splits, out, args):
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    rows, ablations = ([], [])
    configs = {kind: load_config(kind, out) for kind in ["rf", "xgb"]}
    y_train = splits["Train"]["y"]
    ilr_train = ilr_targets(frames["Train"])
    for kind, short in [("rf", "RF"), ("xgb", "XGBoost")]:
        config = configs[kind]
        alpha = config["alpha"]
        preds = fit_predictions(
            learner(kind, config, args.n_jobs),
            alfa(y_train, a=alpha, h=True)["aff"],
            splits,
            alpha,
        )
        save_prediction(
            "alpha-RFCR" if kind == "rf" else "alpha-XGBCR", preds, splits, out, rows
        )
        preds = fit_predictions(
            learner(kind, config, args.n_jobs), ilr_train, splits, 0
        )
        save_prediction("ILR-" + short, preds, splits, out, rows)
        default = RF_DEFAULT if kind == "rf" else XGB_DEFAULT
        default_alpha = 0.3
        for transform, hyperopt, a, params in [
            (False, False, None, default),
            (False, True, None, config),
            (True, False, default_alpha, default),
            (True, True, alpha, config),
        ]:
            targets = y_train if a is None else alfa(y_train, a=a, h=True)["aff"]
            preds = fit_predictions(
                learner(kind, params, args.n_jobs), targets, splits, a
            )
            name = f"ablation-{short}-alpha{int(transform)}-params{int(hyperopt)}"
            case_rows = []
            save_prediction(name, preds, splits, out, case_rows)
            for row in case_rows:
                row.update(
                    {
                        "Alpha_Transform": transform,
                        "Hyperparameter_Optimization": hyperopt,
                        "parameter_source": "fivefold alpha-model search parameters reused"
                        if hyperopt
                        else "original default",
                        "Base_Model": short,
                    }
                )
            ablations.extend(case_rows)
    candidates = []
    for alpha in np.arange(20, 81, dtype=float) / 100:
        model = LinearRegression().fit(
            splits["Train"]["x"], alfa(y_train, a=alpha, h=True)["aff"]
        )
        training_inverse_valid = True
        try:
            inverse_or_error(
                model.predict(splits["Train"]["x"]), alpha, "alpha-MLR training"
            )
        except ValueError:
            training_inverse_valid = False
        try:
            pred = inverse_or_error(
                model.predict(splits["Valid"]["x"]), alpha, "alpha-MLR validation"
            )
            mse = float(mean_squared_error(splits["Valid"]["y"], pred))
        except ValueError:
            mse = None
        candidates.append(
            {
                "alpha": float(alpha),
                "training_inverse_valid": training_inverse_valid,
                "validation_inverse_valid": mse is not None,
                "validation_mse": mse,
            }
        )
    eligible = [
        r
        for r in candidates
        if r["training_inverse_valid"] and r["validation_mse"] is not None
    ]
    if not eligible:
        raise ValueError(
            "No alpha-MLR candidate has valid training and validation inverses"
        )
    alpha = min(eligible, key=lambda r: (r["validation_mse"], r["alpha"]))["alpha"]
    write_json(
        out / "alpha_mlr_search.json",
        {
            "candidates": candidates,
            "selected_alpha": alpha,
            "selection_rule": "valid training and validation inverses; minimum validation MSE; test excluded",
        },
    )
    preds = fit_predictions(
        LinearRegression(), alfa(y_train, a=alpha, h=True)["aff"], splits, alpha
    )
    save_prediction("alpha-MLR", preds, splits, out, rows)
    preds = fit_predictions(LinearRegression(), ilr_train, splits, 0)
    save_prediction("ILR-MLR", preds, splits, out, rows)
    ridge_candidates = []
    for alpha in [1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
            splits["Train"]["x"], ilr_train
        )
        pred = inverse_or_error(
            model.predict(splits["Valid"]["x"]), 0, "ILR-Ridge validation"
        )
        ridge_candidates.append(
            {
                "alpha": alpha,
                "validation_mse": float(mean_squared_error(splits["Valid"]["y"], pred)),
            }
        )
    ridge_alpha = min(
        ridge_candidates, key=lambda r: (r["validation_mse"], r["alpha"])
    )["alpha"]
    write_json(
        out / "ilr_ridge_search.json",
        {"candidates": ridge_candidates, "selected_alpha": ridge_alpha},
    )
    preds = fit_predictions(
        make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha)), ilr_train, splits, 0
    )
    save_prediction("ILR-Ridge", preds, splits, out, rows)
    if args.rscript is not None:
        preds = run_dirichlet(frames, out, args.rscript)
        save_prediction("DirichletReg", preds, splits, out, rows)
    else:
        saved = RESULT_DIR / "predictions/DirichletReg.npz"
        if saved.is_file():
            with np.load(saved, allow_pickle=False) as archive:
                for split, values in splits.items():
                    if not (
                        np.array_equal(
                            archive[f"{split}_model_rows"], values["model_row_indices"]
                        )
                        and np.array_equal(archive[f"{split}_truth"], values["y"])
                    ):
                        raise ValueError(
                            "Saved Dirichlet predictions use a different data split"
                        )
                preds = {split: archive[f"{split}_prediction"] for split in splits}
            save_prediction("DirichletReg", preds, splits, out, rows)
        else:
            print(
                "DirichletReg omitted: use --rscript to fit the R baseline.", flush=True
            )
    write_json(out / "metrics.json", rows)
    write_json(out / "ablation_metrics.json", ablations)
    cols = [
        "Model",
        "Split",
        "MAE",
        "RMSE",
        "R2",
        "MAE_A",
        "RMSE_A",
        "R2_A",
        "geometry_note",
    ]
    pd.DataFrame(rows)[cols].to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(ablations).drop(columns=["Aitchison_center"], errors="ignore").to_csv(
        out / "ablation_metrics.csv", index=False
    )
    print(
        pd.DataFrame(rows).query("Split == 'Test'")[cols].to_string(index=False),
        flush=True,
    )


DIRICHLET_R = '\nargs <- commandArgs(trailingOnly=TRUE)\nsuppressPackageStartupMessages(library(DirichletReg))\nfolder <- args[1]\ntrain <- read.csv(file.path(folder,"Train.csv"),check.names=FALSE)\nvalid <- read.csv(file.path(folder,"Valid.csv"),check.names=FALSE)\ntest <- read.csv(file.path(folder,"Test.csv"),check.names=FALSE)\ncomponent_names <- c("Sethylene","Spropylene","other")\ntrain$comp <- DR_data(train[,component_names])\nvalid$comp <- DR_data(valid[,component_names])\ntest$comp <- DR_data(test[,component_names])\npredictors <- setdiff(colnames(train),c(component_names,"comp"))\nfit <- DirichReg(as.formula(paste("comp ~",paste(predictors,collapse=" + "))),data=train)\nfor (split in c("Train","Valid","Test")) {\n d <- get(tolower(split))\n pred <- as.data.frame(predict(fit,newdata=d))\n if (ncol(pred) != 3) stop("Unexpected DirichletReg prediction dimension")\n colnames(pred) <- component_names\n write.csv(pred,file.path(folder,paste0(split,"_prediction.csv")),row.names=FALSE)\n}\ncapture.output(sessionInfo(),file=file.path(folder,"R_sessionInfo.txt"))\nsaveRDS(fit,file.path(folder,"fit.rds"))\n'


def run_dirichlet(frames, out, rscript):
    directory = out / "dirichlet"
    directory.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        values = frame[PARTS].to_numpy(dtype=float).copy()
        values[values <= 0] = 1e-06
        response = pd.DataFrame(closure(values), columns=PARTS)
        x = frame.loc[:, FEATURE_COLUMNS].reset_index(drop=True).copy()
        x.columns = [f"x{i}" for i in range(x.shape[1])]
        pd.concat([x, response], axis=1).to_csv(directory / f"{split}.csv", index=False)
    script = directory / "fit_dirichlet.R"
    script.write_text(DIRICHLET_R, encoding="utf-8")
    subprocess.run([str(rscript), str(script), str(directory)], check=True)
    return {
        split: pd.read_csv(directory / f"{split}_prediction.csv")[PARTS].to_numpy(
            dtype=float
        )
        for split in frames
    }


def main():
    parser = argparse.ArgumentParser(
        description="Five-fold tuning, model comparison, and ablation."
    )
    parser.add_argument("--output", type=Path, default=RESULT_DIR)
    parser.add_argument(
        "--stage", choices=["check", "tune", "score", "all"], default="check"
    )
    parser.add_argument("--model", choices=["rf", "xgb", "both"], default="both")
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--rscript", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _, frames, splits = make_splits()
    print({name: len(frame) for name, frame in frames.items()}, flush=True)
    kinds = ["rf", "xgb"] if args.model == "both" else [args.model]
    if args.stage in ["tune", "all"]:
        for kind in kinds:
            tune(kind, frames, out, args)
    if args.stage in ["score", "all"]:
        score(frames, splits, out, args)


if __name__ == "__main__":
    main()
