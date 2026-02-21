EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
NO_COMPILE="${NO_COMPILE:-0}"
COMPILE_ARGS=(--compile-mode "${COMPILE_MODE}")
if [[ "${NO_COMPILE}" == "1" ]]; then
  COMPILE_ARGS=(--no-compile)
fi

python flow_train_discrete_ARC.py \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --eval-split "evaluation" \
  --num-demos 3 \
  --image-size 30 \
  --num-colors 12 \
  --embed-dim 512 \
  --depth 10 \
  --num-heads 8 \
  --mlp-ratio 4.0 \
  --dropout 0.0 \
  --no-framewise-causal-attention \
  --attention-backend "auto" \
  --batch-size 4 \
  --eval-batch-size 8 \
  --log-every-steps 5 \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --epochs 20 \
  --learning-rate 2e-4 \
  "${COMPILE_ARGS[@]}" \
  --weight-decay 0 \
  --include-rearc \
  --bf16-autocast \
  --discrete-rate 5.0 \
  --reverse-sampler "sample" \
  --sample-steps 40 \
  --save-path "saves/flow_context_vit_discrete/checkpoint_last.pt" \
  --best-save-path "saves/flow_context_vit_discrete/checkpoint_best.pt" \
  --use-wandb \
  --wandb-project "VisionARC" \
  --wandb-num-vis-samples 8 \
  --wandb-vis-scale 8 \
  --wandb-run-name "flow-context-vit-discrete"
