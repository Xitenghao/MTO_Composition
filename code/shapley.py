from __future__ import annotations
from common import (
    alfa,
    alfainv,
    helm,
    load_best_params,
    prepare_splits,
    FIG_DIR,
    RESULT_DIR,
)
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

UPSTREAM_REPOSITORY = "https://github.com/shapley-composition/ecai-2024-code-and-exp"
PART_NAMES = ("Ethylene", "Propylene", "Other products")
PART_COLORS = ("#0072B2", "#D55E00", "#009E73")
FEATURE_VALUE_CMAP = LinearSegmentedColormap.from_list(
    "feature_value", ("#2C7BB6", "#7B66C2", "#D7191C")
)


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    columns: tuple[str, ...]


FEATURE_GROUPS = (
    FeatureGroup("A/T", ("A/T",)),
    FeatureGroup("$\\mathrm{FD}_{\\mathrm{Si}}$", ("FDSi",)),
    FeatureGroup("MDa", ("MDa",)),
    FeatureGroup("MDb", ("MDb",)),
    FeatureGroup("MDc", ("MDc",)),
    FeatureGroup("MDi", ("Mdi",)),
    FeatureGroup("CS", ("Crystal_size",)),
    FeatureGroup("RT", ("Reaction_temp",)),
    FeatureGroup("WHSV", ("WHSV",)),
    FeatureGroup("TOS", ("TOS",)),
    FeatureGroup("CD", ("CD1", "CD2")),
    FeatureGroup("AS", ("AS0",)),
    FeatureGroup("Mod", ("Modification0",)),
    FeatureGroup("LRS", ("LRS8", "LRS10")),
)


def closure(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 1e-12, None)
    return values / values.sum(axis=1, keepdims=True)


def ilr(values: np.ndarray) -> np.ndarray:
    values = closure(values)
    log_values = np.log(values)
    clr_values = log_values - log_values.mean(axis=1, keepdims=True)
    return clr_values @ helm(values.shape[1]).T


def ilr_inverse(values: np.ndarray, n_parts: int = 3) -> np.ndarray:
    values = np.atleast_2d(np.asarray(values, dtype=float))
    clr_values = values @ helm(n_parts)
    return closure(np.exp(clr_values))


def class_direction_ilr(n_parts: int = 3) -> np.ndarray:
    """Return unit Aitchison directions toward each composition part."""
    directions_clr = np.eye(n_parts) - np.full((n_parts, n_parts), 1.0 / n_parts)
    directions_clr /= np.linalg.norm(directions_clr, axis=1, keepdims=True)
    return directions_clr @ helm(n_parts).T


class AlphaRFCompositionModel:
    def __init__(self, alpha: float, params: dict[str, float | int]) -> None:
        self.alpha = alpha
        self.model = RandomForestRegressor(**params, random_state=12345, n_jobs=-1)

    def fit(
        self, x: pd.DataFrame, compositions: np.ndarray
    ) -> "AlphaRFCompositionModel":
        values = np.asarray(compositions, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Expected finite nonnegative compositions")
        totals = values.sum(axis=1, keepdims=True)
        if (totals <= 0).any():
            raise ValueError("Composition total must be positive")
        if not np.allclose(totals, 1.0, atol=1e-12):
            raise ValueError("Expected already closed response compositions")
        transformed = alfa(values, a=self.alpha, h=True)["aff"]
        self.model.fit(x.to_numpy(dtype=float), transformed)
        return self

    def predict_composition(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy(dtype=float)
        transformed = self.model.predict(x)
        return closure(alfainv(transformed, a=self.alpha, h=True))


class GroupedShapleyCompositionExplainer:
    def __init__(
        self,
        model: AlphaRFCompositionModel,
        background: pd.DataFrame,
        groups: tuple[FeatureGroup, ...],
        seed: int,
    ) -> None:
        self.model = model
        self.columns = list(background.columns)
        self.background = background.to_numpy(dtype=float)
        self.groups = groups
        self.rng = np.random.default_rng(seed)
        self.group_indices = [
            np.asarray(
                [self.columns.index(column) for column in group.columns], dtype=int
            )
            for group in groups
        ]
        covered = sorted((index for indices in self.group_indices for index in indices))
        if covered != list(range(len(self.columns))):
            raise ValueError(
                "Feature groups must cover every model-matrix column exactly once."
            )
        self.base = ilr(self.model.predict_composition(background)).mean(axis=0)

    def explain(
        self, x: np.ndarray, permutations: int
    ) -> tuple[np.ndarray, dict[str, float]]:
        n_groups = len(self.groups)
        n_features = self.background.shape[1]
        orders = np.empty((permutations, n_groups), dtype=int)
        coalition_batches = np.empty(
            (permutations, n_groups + 1, n_features), dtype=float
        )
        for repetition in range(permutations):
            background_row = self.background[
                self.rng.integers(len(self.background))
            ].copy()
            order = self.rng.permutation(n_groups)
            orders[repetition] = order
            coalition_rows = np.repeat(background_row[None, :], n_groups + 1, axis=0)
            current = background_row.copy()
            for step, group_index in enumerate(order, start=1):
                indices = self.group_indices[group_index]
                current[indices] = x[indices]
                coalition_rows[step] = current
            coalition_batches[repetition] = coalition_rows
        coalition_ilr = ilr(
            self.model.predict_composition(
                coalition_batches.reshape(permutations * (n_groups + 1), n_features)
            )
        ).reshape(permutations, n_groups + 1, 2)
        increments = np.diff(coalition_ilr, axis=1)
        contributions = np.empty((permutations, n_groups, 2), dtype=float)
        row_indices = np.arange(permutations)[:, None]
        contributions[row_indices, orders] = increments
        shapley = contributions.mean(axis=0)
        prediction = ilr(self.model.predict_composition(x.reshape(1, -1)))[0]
        raw_residual = prediction - self.base - shapley.sum(axis=0)
        uncertainty = contributions.std(axis=0, ddof=1).mean(axis=1)
        if np.allclose(uncertainty.sum(), 0.0):
            weights = np.full(n_groups, 1.0 / n_groups)
        else:
            weights = uncertainty / uncertainty.sum()
        shapley += weights[:, None] * raw_residual[None, :]
        adjusted_residual = prediction - self.base - shapley.sum(axis=0)
        diagnostics = {
            "raw_local_accuracy_error": float(np.linalg.norm(raw_residual)),
            "adjusted_local_accuracy_error": float(np.linalg.norm(adjusted_residual)),
        }
        return (shapley, diagnostics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of test samples to explain; defaults to the complete test set.",
    )
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIG_DIR)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from the saved Shapley-composition arrays.",
    )
    return parser.parse_args()


def grouped_feature_values(test_data: pd.DataFrame) -> np.ndarray:
    """Return one display value per original feature group."""
    values = []
    for group in FEATURE_GROUPS:
        if group.name == "AS":
            values.append(1.0 - test_data["AS0"].to_numpy(dtype=float))
        elif group.name == "Mod":
            values.append(1.0 - test_data["Modification0"].to_numpy(dtype=float))
        elif len(group.columns) == 1:
            values.append(test_data[group.columns[0]].to_numpy(dtype=float))
        elif group.name == "CD":
            values.append(
                np.select(
                    [test_data["CD1"].eq(1), test_data["CD2"].eq(1)],
                    [1.0, 2.0],
                    default=3.0,
                )
            )
        elif group.name == "LRS":
            values.append(
                np.select(
                    [test_data["LRS8"].eq(1), test_data["LRS10"].eq(1)],
                    [8.0, 10.0],
                    default=12.0,
                )
            )
        else:
            raise ValueError(f"No display-value rule is defined for {group.name}.")
    return np.column_stack(values)


def normalize_feature_values(values: np.ndarray) -> np.ndarray:
    """Map values to robust within-feature ranks for visualization."""
    normalized = np.full_like(values, np.nan, dtype=float)
    for feature_index in range(values.shape[1]):
        column = values[:, feature_index]
        finite = np.isfinite(column)
        valid = column[finite]
        if not len(valid):
            continue
        unique = np.unique(valid)
        if len(unique) == 1:
            normalized[finite, feature_index] = 0.5
        elif len(unique) <= 3:
            normalized[finite, feature_index] = (valid - unique[0]) / (
                unique[-1] - unique[0]
            )
        else:
            order = np.argsort(valid, kind="mergesort")
            sorted_values = valid[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
            ends = np.r_[starts[1:], len(valid)]
            ranks = np.empty(len(valid), dtype=float)
            for start, end in zip(starts, ends):
                ranks[order[start:end]] = 0.5 * (start + end - 1)
            ranks /= len(valid) - 1
            ranks[valid == unique[0]] = 0.0
            ranks[valid == unique[-1]] = 1.0
            normalized[finite, feature_index] = ranks
    return normalized


def select_quantile_explanations(strengths: np.ndarray) -> tuple[np.ndarray, list[str]]:
    total_strength = strengths.sum(axis=1)
    quantiles = np.quantile(total_strength, [0.25, 0.5, 0.75])
    positions = np.asarray(
        [int(np.argmin(np.abs(total_strength - target))) for target in quantiles],
        dtype=int,
    )
    return (
        positions,
        ["Lower-quartile sample", "Median sample", "Upper-quartile sample"],
    )


def plot_global_direction_map(
    feature_names: list[str],
    shapley_array: np.ndarray,
    feature_values: np.ndarray,
    overall_strength: np.ndarray,
    class_directions: np.ndarray,
    output_path: Path,
    top_features: int = 6,
) -> None:
    order = np.argsort(overall_strength)[::-1][:top_features]
    normalized_values = normalize_feature_values(feature_values)
    selected_vectors = shapley_array[:, order, :]
    limit = max(0.45, np.quantile(np.abs(selected_vectors), 0.995) * 1.15)
    cmap = FEATURE_VALUE_CMAP
    if top_features <= 6:
        nrows, ncols, figsize = (2, 3, (10.8, 7.1))
    else:
        ncols = 3
        nrows = int(np.ceil(len(order) / ncols))
        figsize = (10.8, 13.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    flat_axes = np.atleast_1d(axes).ravel()
    used_axes = flat_axes[: len(order)]
    for ax, feature_index in zip(used_axes, order):
        vectors = shapley_array[:, feature_index, :]
        for direction, part_label, color in zip(
            class_directions, ("E", "P", "O"), PART_COLORS
        ):
            endpoint = direction * limit * 0.78
            ax.plot(
                [0, endpoint[0]],
                [0, endpoint[1]],
                color=color,
                linewidth=1.1,
                alpha=0.75,
                zorder=1,
            )
            ax.scatter(*endpoint, s=13, color=color, zorder=2)
            ax.annotate(
                part_label,
                endpoint,
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=10,
                color=color,
            )
        many_points = len(vectors) > 200
        ax.scatter(
            vectors[:, 0],
            vectors[:, 1],
            c=normalized_values[:, feature_index],
            cmap=cmap,
            norm=Normalize(0, 1),
            s=14 if many_points else 26,
            alpha=0.58 if many_points else 0.88,
            edgecolor="none" if many_points else "white",
            linewidth=0 if many_points else 0.25,
            rasterized=True,
            zorder=3,
        )
        ax.axhline(0, color="#888888", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#888888", linewidth=0.6, zorder=0)
        ax.set_title(feature_names[feature_index], fontsize=12)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=10.5)
        ax.grid(color="#e3e3e3", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for panel_index, ax in enumerate(used_axes):
        if panel_index + ncols >= len(order):
            ax.set_xlabel("ILR1", fontsize=11)
        if panel_index % ncols == 0:
            ax.set_ylabel("ILR2", fontsize=11)
    hidden_axes = flat_axes[len(order) :]
    for ax in hidden_axes:
        ax.set_visible(False)
    fig.subplots_adjust(
        left=0.08, right=0.9, bottom=0.08, top=0.95, wspace=0.16, hspace=0.24
    )
    scalar = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    if len(hidden_axes) and top_features > 6:
        position = hidden_axes[0].get_position()
        colorbar_axis = fig.add_axes(
            [
                position.x0 + 0.42 * position.width,
                position.y0 + 0.12 * position.height,
                0.12 * position.width,
                0.76 * position.height,
            ]
        )
        colorbar = fig.colorbar(scalar, cax=colorbar_axis)
    else:
        colorbar = fig.colorbar(
            scalar, ax=used_axes.tolist(), fraction=0.025, pad=0.025
        )
    colorbar.set_label("Relative feature value", fontsize=11)
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.ax.tick_params(labelsize=10)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_ilr_paths(
    feature_names: list[str],
    local_shapley: np.ndarray,
    base_ilr: np.ndarray,
    class_directions: np.ndarray,
    observed_ilr: np.ndarray,
    sample_labels: list[str],
    output_path: Path,
    top_features: int = 4,
) -> None:
    feature_colors = plt.get_cmap("tab20")(np.linspace(0, 0.9, len(feature_names)))
    all_paths = []
    all_path_features = []
    for sample_shapley in local_shapley:
        order = np.argsort(np.linalg.norm(sample_shapley, axis=1))[::-1]
        shown = order[:top_features]
        vectors = [sample_shapley[index] for index in shown]
        labels = [int(index) for index in shown]
        if len(order) > top_features:
            vectors.append(sample_shapley[order[top_features:]].sum(axis=0))
            labels.append(-1)
        points = [np.asarray(base_ilr, dtype=float).copy()]
        for vector in vectors:
            points.append(points[-1] + vector)
        all_paths.append(np.asarray(points))
        all_path_features.append(labels)
    extent = max(
        1.0,
        max((np.max(np.abs(points)) for points in all_paths)) * 1.2,
        np.max(np.abs(observed_ilr)) * 1.15,
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.8), sharex=True, sharey=True)
    used_features: list[int] = []
    for ax, points, labels, observed, sample_label in zip(
        axes, all_paths, all_path_features, observed_ilr, sample_labels
    ):
        for direction, part_name, color in zip(
            class_directions, PART_NAMES, PART_COLORS
        ):
            boundary_end = -direction * extent
            ax.plot(
                [0, boundary_end[0]],
                [0, boundary_end[1]],
                color="#aaaaaa",
                linestyle=(0, (2, 3)),
                linewidth=0.75,
                zorder=0,
            )
            endpoint = direction * extent * 0.72
            ax.plot(
                [0, endpoint[0]],
                [0, endpoint[1]],
                color=color,
                linewidth=1.25,
                alpha=0.85,
                zorder=1,
            )
            ax.scatter(*endpoint, s=15, color=color, zorder=2)
            ax.annotate(
                part_name,
                endpoint,
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=11,
                color=color,
            )
        for segment_index, feature_index in enumerate(labels):
            color = "#777777" if feature_index == -1 else feature_colors[feature_index]
            start, end = (points[segment_index], points[segment_index + 1])
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=2.8,
                zorder=4,
            )
            if feature_index not in used_features:
                used_features.append(feature_index)
        prediction = points[-1]
        ax.scatter(
            *base_ilr, s=38, color="#555555", edgecolor="white", linewidth=0.6, zorder=6
        )
        ax.scatter(
            *prediction,
            s=48,
            color="#E69F00",
            edgecolor="white",
            linewidth=0.6,
            zorder=6,
        )
        ax.scatter(
            *observed, s=42, marker="x", color="#222222", linewidth=1.3, zorder=6
        )
        ax.set_title(sample_label, fontsize=13)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#e1e1e1", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for ax in axes:
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_xlabel("ILR1", fontsize=12)
        ax.tick_params(labelsize=10.5)
    axes[0].set_ylabel("ILR2", fontsize=12)
    feature_handles = [
        Line2D(
            [0],
            [0],
            color="#777777" if index == -1 else feature_colors[index],
            linewidth=1.8,
            marker="o",
            markersize=3,
            label="Remaining features" if index == -1 else feature_names[index],
        )
        for index in used_features
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#555555",
            label="Base",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#E69F00",
            label="Prediction",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#222222",
            linestyle="none",
            label="Observed",
            markersize=6,
        ),
    ]
    fig.legend(
        handles=feature_handles + marker_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.23, top=0.92, wspace=0.14)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_component_importance(
    feature_names: list[str],
    mean_absolute_projection: np.ndarray,
    standard_error: np.ndarray,
    overall_strength: np.ndarray,
    output_path: Path,
) -> None:
    order = np.argsort(overall_strength)
    y = np.arange(len(feature_names), dtype=float)
    offsets = (-0.29, 0.0, 0.29)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for part_index, (part_name, color, offset) in enumerate(
        zip(PART_NAMES, PART_COLORS, offsets)
    ):
        ax.barh(
            y + offset,
            mean_absolute_projection[order, part_index],
            height=0.26,
            xerr=standard_error[order, part_index],
            label=part_name,
            color=color,
            edgecolor="none",
            error_kw={"elinewidth": 0.75, "capsize": 1.8, "capthick": 0.75},
        )
    ax.set_yticks(y, [feature_names[index] for index in order])
    ax.set_xlabel("Mean absolute projection of Shapley composition")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_explanations(splits, saved, figure_dir):
    shapley_array = saved["shapley_ilr"]
    selected = saved["sample_indices"]
    feature_names = saved["feature_names"].tolist()
    component_projections = saved["component_projections"]
    class_directions = saved["class_direction_ilr"]
    base_ilr = saved["base_ilr"]
    strengths = np.linalg.norm(shapley_array, axis=2)
    selected_test_data = splits["x_test"].reset_index(drop=True).iloc[selected]
    display_values = grouped_feature_values(selected_test_data)
    path_positions, path_labels = select_quantile_explanations(strengths)
    path_test_rows = selected[path_positions]
    path_observed_ilr = ilr(np.asarray(splits["y_test_comp"])[path_test_rows])
    projection_standard_error = np.abs(component_projections).std(
        axis=0, ddof=1
    ) / np.sqrt(len(selected))
    plot_component_importance(
        feature_names,
        np.abs(component_projections).mean(axis=0),
        projection_standard_error,
        strengths.mean(axis=0),
        figure_dir / "shap_components.png",
    )
    plot_global_direction_map(
        feature_names,
        shapley_array,
        display_values,
        strengths.mean(axis=0),
        class_directions,
        figure_dir / "shap_directions.png",
    )
    plot_global_direction_map(
        feature_names,
        shapley_array,
        display_values,
        strengths.mean(axis=0),
        class_directions,
        figure_dir / "shap_all.png",
        top_features=len(feature_names),
    )
    plot_ilr_paths(
        feature_names,
        shapley_array[path_positions],
        base_ilr,
        class_directions,
        path_observed_ilr,
        path_labels,
        figure_dir / "shap_paths.png",
    )
    return path_test_rows


def plot_saved(figure_dir=FIG_DIR, result_dir=RESULT_DIR):
    with np.load(result_dir / "shap_values.npz", allow_pickle=False) as saved:
        plot_explanations(prepare_splits(), saved, figure_dir)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    splits = prepare_splits()
    saved_values_path = args.output_dir / "shap_values.npz"
    if args.plot_only:
        saved = np.load(saved_values_path, allow_pickle=False)
        plot_explanations(splits, saved, args.figure_dir)
        return
    rf_alpha, rf_params = load_best_params(RESULT_DIR / "rf_params.json")
    rf_params = {
        "n_estimators": int(rf_params["n_estimators"]),
        "max_depth": int(rf_params["max_depth"]),
        "min_samples_split": int(rf_params["min_samples_split"]),
        "min_samples_leaf": int(rf_params["min_samples_leaf"]),
    }
    model = AlphaRFCompositionModel(rf_alpha, rf_params).fit(
        splits["x_train"], splits["y_train_comp"]
    )
    explainer = GroupedShapleyCompositionExplainer(
        model, splits["x_train"], FEATURE_GROUPS, seed=args.seed
    )
    test_data = splits["x_test"].reset_index(drop=True)
    if args.samples is not None and args.samples > len(test_data):
        raise ValueError(
            f"Requested {args.samples} samples, but the test set has {len(test_data)}."
        )
    if args.samples is None:
        selected = np.arange(len(test_data))
    else:
        selected = np.sort(
            np.random.default_rng(args.seed).choice(
                len(test_data), size=args.samples, replace=False
            )
        )
    n_explained = len(selected)
    all_shapley = []
    diagnostics = []
    for output_index, sample_index in enumerate(selected, start=1):
        print(
            f"Explaining test sample {output_index}/{len(selected)} (row {sample_index})",
            flush=True,
        )
        shapley, diagnostic = explainer.explain(
            test_data.iloc[sample_index].to_numpy(dtype=float),
            permutations=args.permutations,
        )
        all_shapley.append(shapley)
        diagnostics.append(diagnostic)
    shapley_array = np.asarray(all_shapley)
    strengths = np.linalg.norm(shapley_array, axis=2)
    feature_names = [group.name for group in FEATURE_GROUPS]
    class_directions = class_direction_ilr(len(PART_NAMES))
    component_projections = np.einsum("sfd,kd->sfk", shapley_array, class_directions)
    absolute_projections = np.abs(component_projections)
    summary = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_aitchison_norm": strengths.mean(axis=0),
            "standard_deviation": strengths.std(axis=0, ddof=1),
            "standard_error": strengths.std(axis=0, ddof=1) / np.sqrt(n_explained),
            "median_aitchison_norm": np.median(strengths, axis=0),
        }
    ).sort_values("mean_aitchison_norm", ascending=False)
    component_summary = pd.DataFrame(
        [
            {
                "feature": feature_names[feature_index],
                "component": PART_NAMES[part_index],
                "mean_absolute_projection": absolute_projections[
                    :, feature_index, part_index
                ].mean(),
                "standard_deviation": absolute_projections[
                    :, feature_index, part_index
                ].std(ddof=1),
                "standard_error": absolute_projections[
                    :, feature_index, part_index
                ].std(ddof=1)
                / np.sqrt(n_explained),
                "mean_signed_projection": component_projections[
                    :, feature_index, part_index
                ].mean(),
            }
            for feature_index in range(len(feature_names))
            for part_index in range(len(PART_NAMES))
        ]
    )
    total_local_strength = strengths.sum(axis=1)
    representative_position = int(
        np.argmin(np.abs(total_local_strength - np.median(total_local_strength)))
    )
    representative_test_row = int(selected[representative_position])
    representative_x = test_data.iloc[representative_test_row].to_numpy(dtype=float)
    representative_prediction = model.predict_composition(
        representative_x.reshape(1, -1)
    )[0]
    base_composition = ilr_inverse(explainer.base)[0]
    local_summary = pd.DataFrame(
        {
            "feature": feature_names,
            "aitchison_norm": strengths[representative_position],
            **{
                f"projection_{part_name.lower().replace(' ', '_')}": component_projections[
                    representative_position, :, part_index
                ]
                for part_index, part_name in enumerate(PART_NAMES)
            },
            **{
                f"shapley_composition_{part_name.lower().replace(' ', '_')}": ilr_inverse(
                    shapley_array[representative_position]
                )[:, part_index]
                for part_index, part_name in enumerate(PART_NAMES)
            },
        }
    ).sort_values("aitchison_norm", ascending=False)
    summary.to_csv(args.output_dir / "shap_importance.csv", index=False)
    component_summary.to_csv(args.output_dir / "shap_components.csv", index=False)
    local_summary.to_csv(args.output_dir / "shap_local.csv", index=False)
    np.savez_compressed(
        args.output_dir / "shap_values.npz",
        shapley_ilr=shapley_array,
        sample_indices=selected,
        feature_names=np.asarray(feature_names),
        base_ilr=explainer.base,
        component_projections=component_projections,
        class_direction_ilr=class_directions,
        representative_test_row=representative_test_row,
    )
    path_test_rows = plot_explanations(
        splits,
        {
            "shapley_ilr": shapley_array,
            "sample_indices": selected,
            "feature_names": np.asarray(feature_names),
            "component_projections": component_projections,
            "class_direction_ilr": class_directions,
            "base_ilr": explainer.base,
        },
        args.figure_dir,
    )
    metadata = {
        "method": "Grouped permutation Shapley compositions in Helmert ILR coordinates",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "model": "alpha-RFCR",
        "alpha": rf_alpha,
        "model_parameters": rf_params,
        "random_seed": args.seed,
        "explained_test_samples": n_explained,
        "permutations_per_sample": args.permutations,
        "component_importance": "Mean absolute projection onto unit Aitchison direction for each product component",
        "component_names": PART_NAMES,
        "representative_test_row": representative_test_row,
        "representative_selection": "Test sample whose summed local Shapley-composition norm is closest to the median among explained samples",
        "representative_base_composition": base_composition.tolist(),
        "representative_prediction": representative_prediction.tolist(),
        "representative_observed_composition": splits["y_test_comp"][
            representative_test_row
        ].tolist(),
        "local_path_selection": "Explained samples closest to the lower quartile, median, and upper quartile of summed local Aitchison norms",
        "local_path_test_rows": path_test_rows.tolist(),
        "mean_raw_local_accuracy_error": float(
            np.mean([item["raw_local_accuracy_error"] for item in diagnostics])
        ),
        "max_adjusted_local_accuracy_error": float(
            np.max([item["adjusted_local_accuracy_error"] for item in diagnostics])
        ),
    }
    (args.output_dir / "shap_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
