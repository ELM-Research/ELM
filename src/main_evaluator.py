import gc
import json
import os
import torch
from pathlib import Path

from configs.config import get_args
from utils.gpu_manager import GPUSetup
from utils.seed_manager import set_seed
from dataloaders.build_dataloader import BuildDataLoader
from elms.build_elm import BuildELM
from runners.evaluator import evaluate, evaluate_opentslm, run_statistical_analysis, save_confusion_matrix_png, save_other_outputs_histogram_png


def _decode_batch(batch):
    if "text" in batch:
        out = []
        for t in batch["text"]:
            try:
                out.append(json.loads(t))
            except Exception:
                out.append(t)
        batch["text"] = out
    return batch


def _run_opentslm_eval(args, folds, seeds):
    from datasets import load_dataset
    from elms.opentslm_wrapper import build_opentslm

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_opentslm(args, device)
    data_name = "_".join(args.data)
    results_file = f"./{args.opentslm_model}_{data_name}.json"
    all_metrics = []

    for fold in folds:
        for seed in seeds:
            print(f"Evaluating fold {fold} with seed {seed}")
            args.fold = fold
            args.seed = seed
            set_seed(seed)
            for dn in args.data:
                dataset = load_dataset(f"willxxy/{dn}", split=f"fold{fold}_test").with_transform(_decode_batch)
                out = evaluate_opentslm(model, dataset, args)
                all_metrics.append(out)

    statistical_results = run_statistical_analysis(all_metrics)
    print(statistical_results)
    with open(results_file, "w") as f:
        json.dump(statistical_results, f, indent=2)
    print(f"Saved evaluation results to {results_file}")


def _run_elm_eval(args, folds, seeds):
    all_metrics = []

    if args.elm_ckpt:
        checkpoint_dir = os.path.dirname(args.elm_ckpt)
        ckpt_file_name = Path(args.elm_ckpt).stem
    else:
        checkpoint_dir = "./"
        ckpt_file_name = "no_ckpt"
    sys_prompt_name = Path(args.system_prompt).stem
    data_name = "_".join(args.data)
    results_file = os.path.join(checkpoint_dir, f"{ckpt_file_name}_{data_name}_{sys_prompt_name}_{args.perturb}.json")

    for fold in folds:
        for seed in seeds:
            print(f"Evaluating fold {fold} with seed {seed}")
            args.fold = fold
            args.seed = seed
            set_seed(args.seed)
            build_dataloader = BuildDataLoader(args)
            dataloader = build_dataloader.build_dataloader()
            build_elm = BuildELM(args)
            elm_components = build_elm.build_elm(dataloader.dataset.llm_tokenizer)
            gpu_setup = GPUSetup(args)
            elm = gpu_setup.setup_gpu(elm_components["elm"], elm_components["find_unused_parameters"])
            if args.dev:
                gpu_setup.print_model_device(elm, f"{args.llm}_{args.encoder}")
            out = evaluate(elm, dataloader, args)
            all_metrics.append(out)
        if "confusion_matrix" in out:
            cm_path = results_file.replace(".json", f"{fold}_{seed}.png")
            save_confusion_matrix_png(out["confusion_matrix"], cm_path)
            other_path = results_file.replace(".json", f"{fold}_{seed}_other.png")
            save_other_outputs_histogram_png(out["other_output_counts"], other_path, top_k = 20)

    statistical_results = run_statistical_analysis(all_metrics)
    print(statistical_results)

    with open(results_file, "w") as f:
        json.dump(statistical_results, f, indent=2)
    print(f"Saved evaluation results to {results_file}")


def main():
    gc.collect()
    torch.cuda.empty_cache()
    mode = "eval"
    args = get_args(mode)
    args.mode = mode
    # folds = ["1", "2", "3", "4", "5"]
    # seeds = [1337, 1338, 1339, 1340, 1341]
    folds = ["1"]
    seeds = [1337, 1338]

    if args.elm == "opentslm":
        _run_opentslm_eval(args, folds, seeds)
    else:
        _run_elm_eval(args, folds, seeds)


if __name__ == "__main__":
    main()
