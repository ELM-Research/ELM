import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path

from configs.config import get_args
from utils.dir_file_manager import DirFileManager

DIR_FILE_MANAGER = DirFileManager()
PERTURB_ORDER = ["None", "noise", "only_text", "zeros"]
PASTEL_COLORS = ["#A8D8EA", "#F5B7B1", "#ABEBC6", "#D7BDE2"]


def extract_metrics(json_data):
    return {
        "Accuracy": {"mean": json_data["ACC"]["mean"], "std": json_data["ACC"]["std"]},
        "F1": {"mean": json_data["F1"]["mean"], "std": json_data["F1"]["std"]},
    }


def parse_perturbation(filename):
    stem = Path(filename).stem
    for p in ["only_text", "noise", "zeros", "None"]:
        if stem.endswith(f"_{p}"):
            return p
    return None


def load_run_config(json_dir):
    """Load config.yaml from the run directory.
    Expected layout: runs/{llm}_{encoder}/{data}/{run_id}/checkpoints/
    config.yaml lives in {run_id}/ (parent of checkpoints/).
    Returns the config dict or None if not found.
    """
    p = Path(json_dir).resolve()
    run_dir = p.parent if p.name == "checkpoints" else p
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return None


def derive_model_name(json_dir):
    """Derive model label from config.yaml, falling back to directory name."""
    cfg = load_run_config(json_dir)
    if cfg:
        elm = cfg.get("elm", "?")
        llm = cfg.get("llm", "?")
        encoder = cfg.get("encoder", "?")
        return f"ELM: {elm}\nLLM: {llm}\nEncoder: {encoder}"
    p = Path(json_dir).resolve()
    if p.name == "checkpoints":
        return p.parent.parent.parent.name
    return p.name


def derive_dataset_name(json_dir):
    """Derive dataset name from config.yaml, falling back to directory name."""
    cfg = load_run_config(json_dir)
    if cfg:
        data = cfg.get("data", [])
        if isinstance(data, list):
            return "_".join(data) if data else "unknown"
        return str(data)
    p = Path(json_dir).resolve()
    if p.name == "checkpoints":
        return p.parent.parent.name
    return "unknown"


def collect_dir_metrics(json_dir, ckpt_type):
    metrics = {}
    for f in sorted(Path(json_dir).glob("*.json")):
        if ckpt_type in f.name:
            perturb = parse_perturbation(f.name)
            if perturb is not None:
                metrics[perturb] = extract_metrics(DIR_FILE_MANAGER.open_json(f))
    return metrics


def find_aligned_perturbations(all_metrics):
    if not all_metrics:
        return []
    common = set.intersection(*(set(m.keys()) for m in all_metrics.values()))
    return [p for p in PERTURB_ORDER if p in common]


def plot_metric_comparison(all_metrics, model_names, perturbations, metric, dataset, save_path):
    n_models = len(model_names)
    n_perturbs = len(perturbations)
    x = np.arange(n_models)
    width = 0.8 / n_perturbs

    fig, ax = plt.subplots(figsize=(n_models * 5, 6))
    for i, perturb in enumerate(perturbations):
        means = [all_metrics[name][perturb][metric]["mean"] for name in model_names]
        stds = [all_metrics[name][perturb][metric]["std"] for name in model_names]
        offset = (i - (n_perturbs - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=perturb, capsize=4, color=PASTEL_COLORS[i % len(PASTEL_COLORS)],
               edgecolor=PASTEL_COLORS[i % len(PASTEL_COLORS)], linewidth=0.1)

    ax.set_ylabel(metric, fontsize=17)
    ax.set_title(f"The Effect of Perturbations on {dataset}", fontsize=19)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=14, ha="center")
    ax.tick_params(axis="y", labelsize=15)
    ax.legend(title="Perturbation", fontsize=12, title_fontsize=13,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    if metric == "F1":
        top = 100
    elif "ecg-qa" in dataset:
        top = 100
    elif "ecg-instruct-45k" in dataset:
        top = 30
    ax.set_ylim(bottom=0, top = top)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {metric} plot to {save_path}")


def visualize_metric_comparison(json_dirs, output_dir, ckpt_type):
    all_metrics = {}
    model_names = []
    datasets = set()
    for d in json_dirs:
        name = derive_model_name(d)
        metrics = collect_dir_metrics(d, ckpt_type)
        if not metrics:
            print(f"Warning: no result JSONs in {d}, skipping.")
            continue
        all_metrics[name] = metrics
        model_names.append(name)
        datasets.add(derive_dataset_name(d))

    if not all_metrics:
        print("No metrics found in any directory.")
        return

    perturbations = find_aligned_perturbations(all_metrics)
    if not perturbations:
        print("No aligned perturbation types found across all directories.")
        return

    dataset_label = "_".join(sorted(datasets))
    print(f"Models: {model_names}")
    print(f"Dataset: {dataset_label}")
    print(f"Aligned perturbations: {perturbations}")
    os.makedirs(output_dir, exist_ok=True)
    for metric in ["Accuracy", "F1"]:
        save_path = os.path.join(output_dir, f"{metric}_{dataset_label}_comparison.png")
        plot_metric_comparison(all_metrics, model_names, perturbations, metric, dataset_label, save_path)


def main():
    args = get_args("analyze")
    if args.json_dirs:
        visualize_metric_comparison(args.json_dirs, args.output_dir, args.ckpt_type)


if __name__ == "__main__":
    main()