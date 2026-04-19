# CUDA_VISIBLE_DEVICES=3 uv run scripts/erb_minimal.py \
#   --erb-dir ./ecg-reasoning-benchmark \
#   --llm qwen2.5-1.5b-instruct \
#   --encoder st_mem \
#   --elm mlp_llava \
#   --elm-ckpt src/runs/mlp_llava_qwen2.5-1.5b-instruct_st_mem/rl-ecg-r1/0/checkpoints/epoch_best.pt \
#   -- ./ecg-reasoning-benchmark/data \
#   --dataset ptbxl \
#   --ecg-base-dir ../data/ptb_xl/ \
#   --output-dir ./inference_results


  uv run ecg-reasoning-benchmark/evaluation.py ./results \
    --dataset ptbxl \
    --model ecglm \
    --evaluator heuristic \
    --save-dir ./eval_results


    # python evaluation.py /path/to/results/ \
    # --dataset ptbxl \
    # --model ecglm \
    # --evaluator gemini \
    # --gemini-model gemini-3-flash-preview \
    # --use-cache \
    # --save-cache \
    # --load-cache \
    # --save-cache-interval 1 \
    # --save-dir results
