CUDA_VISIBLE_DEVICES=0 uv run src/main_chat.py \
--llm qwen2.5-0.5b-instruct \
--elm patch_elf \
--system_prompt src/dataloaders/system_prompts/system_prompt.txt \
--peft \
--elm_ckpt "src/runs/qwen2.5-0.5b-instruct_None/ecg-r1-no-rl/6/checkpoints/epoch_best.pt" \
--num_encoder_tokens 100 \
--data_representation signal
