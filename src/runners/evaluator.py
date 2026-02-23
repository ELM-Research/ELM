import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import torch

from utils.gpu_manager import is_main, train_dev_break

from runners.helper import batch_to_device

def calculate_acc(references, hypotheses):
    return np.mean([ref == hyp for ref, hyp in zip(references, hypotheses)])

def evaluate_strings(references, hypotheses):
    if len(references) != len(hypotheses):
        raise ValueError("The number of references and hypotheses must be the same.")
    valid_pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses) if ref and hyp]
    if not valid_pairs:
        return {
            "ACC": 0.0,
        }
    valid_refs, valid_hyps = zip(*valid_pairs)
    return {
        "ACC": calculate_acc(valid_refs, valid_hyps),
    }


def run_statistical_analysis(all_seeds_results):
    metrics = list(all_seeds_results[0]["metrics"].keys())
    statistical_results = {}

    for metric in metrics:
        metric_values = [result["metrics"][metric] for result in all_seeds_results]

        if isinstance(metric_values[0], dict):
            statistical_results[metric] = {}
            for sub_metric in metric_values[0].keys():
                if isinstance(metric_values[0][sub_metric], list):
                    mean = np.mean(metric_values[0][sub_metric]) * 100
                    values = [np.mean(result["metrics"][metric][sub_metric]) * 100 for result in all_seeds_results]
                else:
                    mean = np.mean(metric_values[0][sub_metric]) * 100
                    values = [np.mean(result["metrics"][metric][sub_metric]) * 100 for result in all_seeds_results]

                mean = np.mean(values)
                std = np.std(values, ddof=1)
                confidence = 0.95
                degrees_of_freedom = len(values) - 1
                t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
                margin_of_error = t_value * (std / np.sqrt(len(values)))
                conf_interval = (mean - margin_of_error, mean + margin_of_error)
                statistical_results[metric][sub_metric] = {
                    "mean": mean,
                    "std": std,
                    "conf_interval": conf_interval,
                }
        else:
            values = [np.mean(result["metrics"][metric]) * 100 for result in all_seeds_results]
            mean = np.mean(values)
            std = np.std(values, ddof=1)

            confidence = 0.95
            degrees_of_freedom = len(values) - 1
            t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
            margin_of_error = t_value * (std / np.sqrt(len(values)))

            conf_interval = (mean - margin_of_error, mean + margin_of_error)

            statistical_results[metric] = {
                "mean": mean,
                "std": std,
                "conf_interval": conf_interval,
            }

    return statistical_results

def index_nested(encoder_tokenizer_out, batch):
    return {k: index_nested(v, batch) if isinstance(v, dict) else v[batch:batch+1] for k, v in encoder_tokenizer_out.items()}

def evaluate(elm, dataloader, args):
    show_progress = is_main()
    elm.eval()
    progress = tqdm(
        dataloader,
        desc=f"LLM: {args.llm} ENCODER: {args.encoder}",
        disable=not show_progress,
        leave=False,
    )
    dataset = dataloader.dataset
    device = next(elm.parameters()).device
    all_refs, all_hyps = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            B = batch["elm_input_ids"].shape[0]
            for b in range(B):
                full_ids = batch["elm_input_ids"][b].tolist()
                full_attn = batch["elm_attention_mask"][b].tolist()
                if args.elm:
                    signal_indices = batch["signal_id_indices"][b]
                    full_encoder_tokenizer_out = index_nested(batch["encoder_tokenizer_out"], b)
                ranges = dataset.get_response_ranges(full_ids)
                gt_texts = dataset.get_ground_truth_responses(full_ids, ranges)
                if getattr(args, "dev", False):
                    print(f"\n--- Batch {batch_idx}, Sample {b} ---")
                    print(f"Total turns: {len(ranges)}")
                    dataset.assert_range_alignment(full_ids, ranges)
                for turn_idx, ((s, _), gt) in enumerate(zip(ranges, gt_texts)):
                    sub_ids = full_ids[:s]
                    sub_attn = full_attn[:s]
                    gen_batch = {
                        "elm_input_ids": torch.tensor(sub_ids, dtype=torch.int64).unsqueeze(0),
                        "elm_attention_mask": torch.tensor(sub_attn, dtype=torch.float32).unsqueeze(0),
                    }
                    if args.elm:
                        gen_batch["encoder_tokenizer_out"] = full_encoder_tokenizer_out
                        gen_batch["signal_id_indices"] = signal_indices
                    gen_batch = {k: batch_to_device(v, device) for k, v in gen_batch.items()}
                    gen_out = elm.generate(**gen_batch)[0].cpu().tolist()
                    gen_txt = dataset.get_generated_response_for_turn(sub_ids, gen_out)
                    if getattr(args, "dev", False):
                        print(f"\nTurn {turn_idx + 1}:")
                        print(f"\nGround Truth:\n{gt}")
                        print(f"\nGenerated:\n{gen_txt}")
                        print("-" * 100)
                    if gt and gen_txt:
                        all_refs.append(gt)
                        all_hyps.append(gen_txt)
            if train_dev_break(getattr(args, "dev", False), batch, 0):
                break
    results = evaluate_strings(all_refs, all_hyps)
    print("\n=== N-Turn Evaluation (generated vs. gold response only) ===")
    print(f"Pairs: {len(all_refs)}")
    print(f"ACC: {results['ACC']:.4f}")
    return {
        "num_pairs": len(all_refs),
        "metrics": results,
        "references": all_refs,
        "hypotheses": all_hyps,
    }