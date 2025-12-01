TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file config_zero2.yaml --main_process_port 29390 ../grpo_main.py \
    --env-name maze-sample-5x5-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 100 \
    --num-env-steps 2000 \
    --num-steps 64 \
    --grad-accum-steps 16 \
    --max-new-tokens 64 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path liuhaotian/llava-v1.6-mistral-7b \
    --use-lora \
    --train-vision all \
    --use-curriculum \
    --curriculum-start-size 5 \
    --curriculum-end-size 20 \
    --curriculum-progression success_rate \
    --curriculum-success-threshold 0.7 \
    --curriculum-min-episodes 100 \
    --wandb-project VLM_GRPO \
    --wandb-run MAZE_GRPO \
    --use-wandb \
    # --q4


