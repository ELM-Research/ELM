CUDA_VISIBLE_DEVICES=3 uv run src/main_evaluator.py \
--data_representation signal \
--data ecg-qa-ptbxl-250-2500 \
--llm llama-3.2-1b-instruct \
--elm fuyu \
--elm_ckpt src/runs/pretrain/llama-3.2-1b-instruct_None/3/checkpoints/epoch_epoch_1_step_-1.pt