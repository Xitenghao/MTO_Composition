import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import plot_tree
from common import COMPOSITION_COLUMNS, FIG_DIR, RESULT_DIR, alfa, load_best_params, prepare_splits
from experiments import learner

def set_plot_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 16
    plt.rcParams["axes.unicode_minus"] = False

def draw_composition_error_plot(
    y_train,
    y_valid,
    y_test,
    y_train_pred,
    y_valid_pred,
    y_test_pred,
    model_name,
    figure_path,
    test_color,
    component_names,
):
    component_label_map = {
        "Sethylene": "$y_{\\mathrm{ethylene}}$",
        "Spropylene": "$y_{\\mathrm{propylene}}$",
        "other": "$y_{\\mathrm{other}}$",
    }
    label_fontsize = 16
    title_fontsize = 18
    tick_fontsize = 13
    legend_fontsize = 10
    metrics_fontsize = 13
    inset_title_fontsize = 13

    def draw_one_target(
        ax,
        show_legend,
        y_train_col,
        y_valid_col,
        y_test_col,
        y_train_pred_col,
        y_valid_pred_col,
        y_test_pred_col,
        title,
    ):
        ax.plot(
            [y_train_col.min(), y_train_col.max()],
            [y_train_col.min(), y_train_col.max()],
            "k--",
            lw=1.2,
        )
        ax.scatter(
            y_train_pred_col, y_train_col, c="black", label="Train", s=15, alpha=0.7
        )
        ax.scatter(
            y_valid_pred_col, y_valid_col, c="green", label="Valid", s=15, alpha=0.7
        )
        ax.scatter(
            y_test_pred_col, y_test_col, c=test_color, label="Test", s=15, alpha=0.7
        )
        ax.set_xlabel(f"{model_name} Predicted {title}", fontsize=label_fontsize)
        ax.set_ylabel(f"Observed {title}", fontsize=label_fontsize)
        ax.set_title(f"Response component {title}", fontsize=title_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        if show_legend:
            ax.legend(fontsize=legend_fontsize, frameon=True)
        inset_metrics = ax.inset_axes([0.027, 0.56, 0.45, 0.3])
        inset_metrics.axis("off")
        metrics_text = f"Train $R^2$: {r2_score(y_train_col, y_train_pred_col):.3f}\nValid $R^2$: {r2_score(y_valid_col, y_valid_pred_col):.3f}\nTest $R^2$: {r2_score(y_test_col, y_test_pred_col):.3f}"
        inset_metrics.text(
            -0.001,
            0.48,
            metrics_text,
            fontsize=metrics_fontsize,
            verticalalignment="center",
            bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="square,pad=0.5"),
        )
        inset_error = ax.inset_axes([0.6, 0.1, 0.35, 0.35])
        inset_error.hist(y_test_col - y_test_pred_col, bins=200, color="gray")
        inset_error.set_title("Error", fontsize=inset_title_fontsize)
        inset_error.set_xlim(-0.5, 0.5)
        inset_error.tick_params(
            axis="both",
            which="both",
            labelsize=11,
            labelleft=False,
            labelbottom=True,
            length=0,
        )

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    fig, axes = plt.subplots(
        1, len(component_names), figsize=(5.5 * len(component_names), 5.3)
    )
    axes = np.atleast_1d(axes)
    for idx, name in enumerate(component_names):
        display_name = component_label_map.get(name, name)
        draw_one_target(
            axes[idx],
            idx == 0,
            y_train[:, idx],
            y_valid[:, idx],
            y_test[:, idx],
            y_train_pred[:, idx],
            y_valid_pred[:, idx],
            y_test_pred[:, idx],
            display_name,
        )
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def composition_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred, multioutput="uniform_average")
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred, multioutput="uniform_average"),
        "r2": r2_score(y_true, y_pred, multioutput="uniform_average"),
    }

def draw_predictions(
    model_name, y_train_comp, y_valid_comp, y_test_comp, preds, output_path, test_color
):
    y_train_pred_comp, y_valid_pred_comp, y_test_pred_comp = preds
    for split, y_true, y_pred in (
        ("Train", y_train_comp, y_train_pred_comp),
        ("Valid", y_valid_comp, y_valid_pred_comp),
        ("Test", y_test_comp, y_test_pred_comp),
    ):
        metrics = composition_metrics(y_true, y_pred)
        print(model_name, split, metrics["mae"], metrics["rmse"], metrics["r2"])
    draw_composition_error_plot(
        y_train_comp,
        y_valid_comp,
        y_test_comp,
        y_train_pred_comp,
        y_valid_pred_comp,
        y_test_pred_comp,
        model_name,
        output_path,
        test_color,
        COMPOSITION_COLUMNS,
    )

def draw_tree(model, feature_names):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(17, 6))
    plot_tree(
        model.estimators_[0],
        max_depth=2,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=9,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rf_tree.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_models():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = {"alpha-RFCR": r"$\alpha$-RFCR", "alpha-XGBCR": r"$\alpha$-XGBCR"}
    for path in sorted((RESULT_DIR / "predictions").glob("*.npz")):
        if path.stem.startswith("ablation-"):
            continue
        with np.load(path, allow_pickle=False) as saved:
            truth = [saved[f"{s}_truth"] for s in ["Train", "Valid", "Test"]]
            prediction = [saved[f"{s}_prediction"] for s in ["Train", "Valid", "Test"]]
        draw_predictions(labels.get(path.stem, path.stem), *truth, prediction,
                         FIG_DIR / f"{path.stem}_fit.png", "red")
    splits = prepare_splits()
    alpha, params = load_best_params(RESULT_DIR / "rf_params.json")
    model = learner("rf", params, 6).fit(splits["x_train"], alfa(splits["y_train_comp"], a=alpha)["aff"])
    draw_tree(model, list(splits["x_train"].columns))


def main():
    parser = argparse.ArgumentParser(description="Plot public model results and Shapley compositions.")
    parser.add_argument("--group", choices=["all", "models", "shapley"], default="all")
    args = parser.parse_args()
    if args.group in ["all", "models"]:
        plot_models()
    if args.group in ["all", "shapley"]:
        from shapley import plot_saved
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        with plt.rc_context():
            plt.rcdefaults()
            plot_saved()


if __name__ == "__main__":
    main()
