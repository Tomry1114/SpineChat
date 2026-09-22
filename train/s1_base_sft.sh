#!/bin/bash
# SpineChat Stage 1 — base multimodal SFT (LoRA; language-modeling loss only) via ms-swift.
# Produces the base adapter that Stage 3 (CurveToken) continues from, and whose frozen mid-layer
# features Stage 2 (LSEM) reads. Fill in MODEL and DATA (a multi-turn screening->grounding->
# diagnosis jsonl: messages / images / objects). No data ships with this repo.
set -e
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
DATA=${DATA:-data/train.jsonl}
OUT=${OUT:-outputs/s1_base_sft}

swift sft \
  --model "$MODEL" \
  --train_type lora --target_modules all-linear \
  --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
  --freeze_vit false --vit_lr 2e-5 \
  --learning_rate 1e-4 --num_train_epochs 3 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --lr_scheduler_type cosine --weight_decay 0.1 \
  --max_length 4096 --max_pixels 1003520 --gradient_checkpointing true \
  --loss_scale default \
  --dataset "$DATA" \
  --output_dir "$OUT"
