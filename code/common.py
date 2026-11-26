import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FEATURE_COLUMNS = [
    "A/T", "FDSi", "MDa", "MDb", "MDc", "Mdi", "Crystal_size",
    "Reaction_temp", "WHSV", "TOS", "CD1", "CD2", "AS0",
    "Modification0", "LRS8", "LRS10",
]
COMPOSITION_COLUMNS = ["Sethylene", "Spropylene", "other"]
INTEGER_COLUMNS = {"CD1", "CD2", "AS0", "Modification0", "LRS8", "LRS10"}
COLUMN_DTYPES = {
    column: "int64" if column in INTEGER_COLUMNS else "float64"
    for column in FEATURE_COLUMNS + COMPOSITION_COLUMNS
}

def helm(n):
    mat = np.zeros((n - 1, n))
    i = np.arange(2, n + 1)
    r = 1 / np.sqrt(i * (i - 1))
    for j in range(1, n):
        mat[j - 1, 0:j] = np.repeat(r[j - 1], j)
        mat[j - 1, j] = -j * r[j - 1]
    return mat

def alfa(x, a, h=True):
    d = x.shape[1]
    if abs(a) > 1e-09:
        z = x**a
        ta = np.sum(z, axis=1)
        z = d / a * z / ta[:, None] - 1 / a
        sa = np.sum(np.log(ta))
    else:
        xa = np.log(x)
        z = xa - np.mean(xa, axis=1)[:, None]
        sa = x.shape[0] * np.log(d)
    if h:
        aff = np.dot(z, helm(d).T)
        return {"sa": sa, "aff": aff}
    return {"sa": sa, "aff": z}

def alfainv(x, a, h=True):
    x = np.array(x)
    d = x.shape[1]
    if d == 1:
        x = x.T
    if h:
        y = np.dot(x, helm(d + 1))
    else:
        y = x
    if a != 0:
        z = (a * y + 1) ** (1 / a)
        z = z / np.sum(z, axis=1)[:, None]
    else:
        ey = np.exp(y)
        z = ey / np.sum(ey, axis=1)[:, None]
    return z

def safe_alfainv(x, a, eps=1e-08):
    x = np.asarray(x, dtype=float)
    d = x.shape[1]
    y = np.dot(x, helm(d + 1))
    lower_bound = -1 / a + eps
    clipped = np.maximum(y, lower_bound)
    z = (a * clipped + 1) ** (1 / a)
    z = z / np.sum(z, axis=1, keepdims=True)
    return (z, not np.allclose(clipped, y))

def closure(values):
    values = np.asarray(values, dtype=float)
    return values / values.sum(axis=1, keepdims=True)

def load_best_params(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("cv_folds") != 5:
        raise ValueError(f"Refusing legacy/unspecified-fold plot parameters: {path}")
    params = payload["best_params"].copy()
    alpha = float(params.pop("alpha"))
    return (alpha, params)

def split_frames():
    files = {"Train": "train.csv", "Valid": "validation.csv", "Test": "test.csv"}
    missing = [file for file in files.values() if not (DATA_DIR / file).is_file()]
    if missing:
        raise FileNotFoundError(
            "Datasets are not publicly distributed. Request access from the "
            "corresponding author and place approved CSV files in data/: "
            + ", ".join(missing)
        )
    frames = {name: pd.read_csv(DATA_DIR / file, dtype=COLUMN_DTYPES,
                               float_precision="round_trip") for name, file in files.items()}
    for name, frame in frames.items():
        if list(frame.columns) != FEATURE_COLUMNS + COMPOSITION_COLUMNS:
            raise ValueError(f"Unexpected columns in {name}")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite model inputs in {name}")
    return pd.concat(frames.values(), ignore_index=True), frames


def prepare_splits():
    _, frames = split_frames()
    target = frames["Train"][COMPOSITION_COLUMNS].to_numpy(dtype=float).copy()
    target[target <= 0] = 1e-6
    output = {"y_train_ilr": alfa(closure(target), a=0, h=True)["aff"]}
    for split, suffix in [("Train", "train"), ("Valid", "valid"), ("Test", "test")]:
        output[f"x_{suffix}"] = frames[split][FEATURE_COLUMNS]
        output[f"y_{suffix}_comp"] = closure(frames[split][COMPOSITION_COLUMNS].to_numpy(dtype=float))
    return output
