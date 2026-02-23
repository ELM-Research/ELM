CUDA_VISIBLE_DEVICES=0 uv run src/main_evaluator.py \
--data_representation signal \
--data ecg-qa-ptbxl-250-2500 \
--llm llama-3.2-1b-instruct \
--elm fuyu \
--elm_ckpt src/runs/pretrain/llama-3.2-1b-instruct_None/2/checkpoints/epoch_best.pt