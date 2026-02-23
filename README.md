# ELM

A clean research framework for training and evaluating ECG-language models (ELMs): ECG encoder + LLM, with optional multimodal connectors.

> **Status:** Active research.

## Setup

```bash
git clone https://github.com/ELM-Research/ELM.git
cd ELM
uv sync
```

Optional: compile the Rust ECG-Byte tokenizer (needed for `symbolic` representation).

```bash
cd src/dataloaders/data_representation/bpe
maturin develop --release
```

## What this repo supports

### Data

Datasets are loaded from Hugging Face (`willxxy/<dataset-key>`). Pass one or more keys via `--data`.

Common keys include:

- `ecg-qa-mimic-iv-ecg-250-{500,1250,2500}`
- `ecg-qa-ptbxl-250-{500,1250,2500}`
- `pretrain-mimic-250-{500,1250,2500}`
- `ecg-instruct-45k-250-{500,1250,2500}`
- `ecg-bench-pulse-250-{500,1250,2500}`
- `ecg-instruct-pulse-250-{500,1250,2500}`
- `ecg-grounding-250-2500`

### Representations (`--data_representation`)

- `signal` (raw ECG matrix)
- `symbolic` (ECG-Byte token stream)
- `stacked_signal` (paired encoder + LLM flow)
- `rgb` (image-style ECG input)

### ECG encoders (`--encoder`)

- ECG-native: `merl`, `mlae`, `mtae`, `st_mem`
- Vision backbones: `clip-vit-base-patch32`, `siglip2-so400m-patch16-naflex`, `vit-base-patch16-224-in21k`

### LLMs (`--llm`)

- `llama-3.2-1b-instruct`
- `llama-3.2-3b-instruct`
- `qwen2.5-1.5b-instruct`
- `qwen2.5-7b-instruct`
- `gemma-2-2b-it`

### ELM connectors (`--elm`)

- `llava`
- `fuyu`
- omit `--elm` to run the LLM alone (no encoder connector)

## Usage

### Train (single GPU)

```bash
uv run src/main_trainer.py \
  --data pretrain-mimic-250-2500 \
  --data_representation signal \
  --llm qwen2.5-1.5b-instruct \
  --encoder merl \
  --elm llava \
  --batch_size 4 \
  --epochs 1
```

### Train (multi-GPU / DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run torchrun --standalone --nproc_per_node=4 \
  src/main_trainer.py \
  --data pretrain-mimic-250-2500 \
  --data_representation signal \
  --llm qwen2.5-1.5b-instruct \
  --encoder merl \
  --elm llava \
  --distributed \
  --batch_size 8 \
  --epochs 3
```

### Evaluate

```bash
uv run src/main_evaluator.py \
  --data ecg-qa-mimic-iv-ecg-250-2500 \
  --data_representation signal \
  --llm qwen2.5-1.5b-instruct \
  --encoder merl \
  --elm llava \
  --elm_ckpt src/runs/pretrain/<run>/best.pt
```

Evaluation runs across configured folds/seeds and writes `evaluation_results.json` next to the checkpoint.

## High-value flags

| Flag | Purpose |
|---|---|
| `--distributed` | Enable DDP training |
| `--peft` | Enable LoRA PEFT for the LLM |
| `--torch_compile` | Compile model with `torch.compile` |
| `--wandb` | Weights & Biases logging |
| `--data_subset` | Use dataset fraction for quick runs |
| `--augment_ecg` / `--augment_rgb` | Enable augmentations |
| `--perturb` | `noise`, `zeros`, or `only_text` |
| `--ecg_tokenizer` | Path to ECG tokenizer (symbolic pipeline) |

## Project layout

- `src/main_trainer.py` — training entrypoint
- `src/main_evaluator.py` — evaluation entrypoint
- `src/dataloaders/` — dataset loading + representation pipelines
- `src/elms/` — model builders (LLMs, encoders, connectors)
- `src/runners/` — train/eval loops
- `scripts/` — example shell scripts

## Related

If you want encoder pretraining recipes first, see: [ecg_nn](https://github.com/ELM-Research/ecg_nn).

## License

MIT, except model files with upstream non-commercial licenses (see source headers / upstream references).
