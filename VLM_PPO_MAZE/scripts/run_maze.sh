TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file config_zero2.yaml --main_process_port 29380 ../main.py \
    --env-name custom-maze-5x5 \
    --init-lr 1e-4 \
    --end-lr 1e-7 \
    --lr_max_steps 25 \
    --eval-num-per-episode 100 \
    --num-env-steps 50000 \
    --num-steps 32 \
    --grad-accum-steps 32 \
    --max-new-tokens 64 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 6 \
    --mini-batch-size 1 \
    --value-loss-coef 0.25 \
    --max-grad-norm 0.5 \
    --model-path mvideet1/llava-mistral-7b-finetuned \
    --use-lora \
    --train-vision all \
    --wandb-project VLM_GRPO \
    --wandb-run MAZE_PPO_SFT \
    --use-wandb \
    # --use-curriculum \
    # --curriculum-start-size 5 \
    #--curriculum-end-size 100 \
    #--curriculum-progression success_rate \
    #--curriculum-success-threshold 0.7 \
    #--curriculum-min-episodes 100 \
    # --q4


