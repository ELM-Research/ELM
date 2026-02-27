uv run src/main_analyze_results.py \
--json_dirs src/runs/llama-3.2-1b-instruct_merl/ecg-instruct-45k-250-2500/0/checkpoints src/runs/llama-3.2-1b-instruct_None/ecg-instruct-45k-250-2500/0/checkpoints \
--ckpt_type epoch_epoch_0_step_-1


uv run src/main_analyze_results.py \
--json_dirs src/runs/llama-3.2-1b-instruct_merl/ecg-qa-ptbxl-250-2500/0/checkpoints src/runs/llama-3.2-1b-instruct_None/ecg-qa-ptbxl-250-2500/0/checkpoints \
--ckpt_type epoch_epoch_0_step_-1