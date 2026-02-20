# Vision ARC

<img src="assets/teaser.png" alt="Teaser" width="90%" />

This is the project webpage for the paper [ARC Is a Vision Problem!](https://arxiv.org/abs/2511.14761)

We formulate ARC with a vision paradigm, casting it as an image-to-image translation task.

## Visualization Galleries ✨

### VARC on ARC-1 🎉
[🌟🌟Explore the full gallery on ARC-1 VARC ↗🌟🌟](https://lillian039.github.io/assets/html/varc/arc_agi_1_VARC.html)

[🌟🌟Explore the full gallery on ARC-1 VARC-ensemble ↗🌟🌟](https://lillian039.github.io/assets/html/varc/arc_agi_1_ensemble.html)

![ARC1 Success](assets/arc1_success.png)

### VARC on ARC-2 🎉
[🌟🌟Explore the full gallery on ARC-2 VARC ↗🌟🌟](https://lillian039.github.io/assets/html/varc/arc_agi_2_VARC.html)

[🌟🌟Explore the full gallery on ARC-2 VARC-emsemble ↗🌟🌟](https://lillian039.github.io/assets/html/varc/arc_agi_2_ensemble.html)


![ARC2 Success](assets/arc2_success.png)

### Pixel-to-Pixel Attention 🔍
Attention heatmaps highlighting how VARC aligns pixels between inputs and outputs.  
[🌟🌟Try interactive attention demos ↗🌟🌟](https://lillian039.github.io/assets/html/varc/attention_heatmap.html)

![Heatmap](assets/heatmap.png)

### Task Token t-SNE Demonstrations 🔍
t-SNE of task embeddings, on the 400 task tokens learned from the ARC-1 training set. Each point represents a single task.

[🌟🌟Try interactive task token demos ↗🌟🌟](https://lillian039.github.io/assets/html/varc/task_embedding_tsne.html)

![tSNE](assets/tsne_task_token.png)

# Training and Inference Code

## Environment setup
```
conda create -n visarc python==3.10
conda activate visarc
pip install -r requirements.txt
hf auth login
wandb login
```

## Download trained checkpoints & predictions

### Links to checkpoints after offline training
#### Checkpoint: VARC-ViT-18M
```
https://huggingface.co/VisionARC/offline_train_ViT/tree/main
```
#### Checkpoint: VARC-Unet-55M
```
https://huggingface.co/VisionARC/offline_train_Unet/tree/main
```

### Download checkpoints
These checkpoints are from offline training, so they still need to be evaluated with TTT.
```
mkdir saves/
mkdir saves/offline_train_ViT
mkdir saves/offline_train_Unet
hf download VisionARC/offline_train_ViT --local-dir saves/offline_train_ViT
hf download VisionARC/offline_train_Unet --local-dir saves/offline_train_Unet
```
ARC-1 and ARC-2 test-time training (TTT) use the same checkpoint from ARC-1 training dataset + [RE-ARC](https://github.com/michaelhodel/re-arc) dataset
### Download VARC's predictions
These predictions are made after TTT on each task.
```
hf download VisionARC/VARC_predictions --local-dir . --repo-type dataset
unzip VARC_predictions.zip
```

After downloading the predictions, run analysis to get results and generate HTML visualizations.

ARC-1 results:
```
bash script/analysis/arc_1_vit.sh
bash script/analysis/arc_1_ensemble.sh
```
The HTML visualizations will be at ``arc_agi_1_vit.html`` and ``arc_agi_1_ensemble.html``.

ARC-2 results:
```
bash script/analysis/arc_2_vit.sh
bash script/analysis/arc_2_ensemble.sh
```
The HTML visualizations will be at ``arc_agi_2_vit.html`` and ``arc_agi_2_ensemble.html``.

## Reproduce our result from scratch

### 1. Build augmented TTT dataset
This also includes commands to sanity check the augmented versions.
```
# Build augmented data
python augment_data.py
# Run the following command
bash script/sanity_ARC1.sh
bash script/sanity_ARC2.sh
```

### 2. Offline training
Train VARC-ViT-18M (5h 12m 42s on 8 x H200):
```
bash script/offline_train_VARC_ViT.sh 
```
Train VARC-Unet-55M (7h 21m 29s on 8 x H200):
```
bash script/offline_train_VARC_Unet.sh
```

### 3. Test-time training
#### ARC-1
Train VARC-ViT-18M with TTT:
```
bash script/test_time_training_VARC_ViT_ARC1.sh
```
Each run's result can range between [52, 56].

Train VARC-Unet-55M with TTT:
```
bash script/test_time_training_VARC_Unet_ARC1.sh
```
Each run's result can range between [47, 49].
#### ARC-2
Train VARC-ViT-18M with TTT:
```
bash script/test_time_training_VARC_ViT_ARC2.sh
```
Each run's result can range between [6, 10].

Train VARC-Unet-55M with TTT:
```
bash script/test_time_training_VARC_Unet_ARC2.sh
```
Each run's result can range between [3, 6].

### 4. Run analysis
Run analysis to get final results and generate HTML visualizations (same as with downloaded predictions) by modifying `--output-root` to the path you save your predictions (e.g., `outputs/ARC_1_eval_ViT_attempt_0`).
```
bash script/analysis/arc_1_vit.sh
bash script/analysis/arc_1_ensemble.sh
```
The HTML visualizations will be at ``arc_agi_1_vit.html`` and ``arc_agi_1_ensemble.html``.

### 5. RL stage on top of pretrained VARC (GRPO + Qwen3-VL yes/no reward)
This repo includes `rl_train_ARC.py`, which adds a reinforcement learning stage after loading a pretrained VARC checkpoint (`--resume-checkpoint`).

The stage works as:
1. Sample multiple output candidates from VARC for each query example.
2. Build a judge prompt from task demonstrations + query input + candidate output.
3. Score each candidate with a VLM reward model (`--reward-model-id`, default `Qwen/Qwen3-VL-2B-Instruct`) using `logit("yes") - logit("no")`.
4. Optimize VARC with GRPO (`--rl-group-size`, `--rl-clip-eps`, `--rl-beta`).
5. Log training metrics and sample visualizations with Weights & Biases (`--use-wandb`) and optional local PNG dumps (`--rl-vis-dir`).
6. Optional reward sanity checks compare sampled-candidate reward vs ground-truth-candidate reward (`--rl-reward-sanity-every`, `--rl-reward-sanity-samples`).

Example command (single ARC task):
```
bash script/rl_train_VARC_ViT.sh
```

Direct command skeleton:
```
python rl_train_ARC.py \
  --resume-checkpoint saves/offline_train_ViT/checkpoint_best.pt \
  --data-root raw_data/ARC-AGI \
  --train-split eval_color_permute_ttt_9/<task_id> \
  --eval-split eval_color_permute_ttt_9/<task_id> \
  --architecture vit \
  --reward-model-id Qwen/Qwen3-VL-2B-Instruct \
  --rl-group-size 4 \
  --rl-action-mask target \
  --rl-vis-every 10 \
  --rl-vis-samples 4 \
  --rl-reward-sanity-every 10 \
  --rl-reward-sanity-samples 2 \
  --rl-vis-dir outputs/rl_vis/<task_id> \
  --use-wandb \
  --wandb-project VisionARC \
  --wandb-run-name varc-rl-<task_id> \
  --rl-max-steps 200 \
  --rl-save-path saves/rl_stage/<task_id>_rl.pt
```

### 6. Flow matching from scratch with frame-context ViT
This repo includes `flow_train_ARC.py` for a from-scratch training setup where each sample is:
1. `m` demonstration pairs: `{(x_1,y_1), ..., (x_m,y_m)}`
2. one query pair `(x_q, y_q)`, where `y_q` is the frame to denoise

Tokenized frame sequence:
- Demo-only context length: `2 * m * s`, where `s = H * W`
- Full sequence used by the model: `(2 * m + 2) * s` (demos + query input + query output frame)

Training objective:
- Flow matching with per-frame independent noise levels `t_f`
- Path: `x_t = (1 - t_f) * x_0 + t_f * eps`
- Target velocity: `v* = eps - x_0`
- Default loss applies to the final solution frame only (`--loss-on-target-only`)

Evaluation:
- Keep demo frames and query input clean
- Initialize only the last solution frame from noise
- Denoise only that last frame via Euler integration
- Report exact-match sample/task accuracy

Run:
```
bash script/flow_train_context_vit.sh
```

Direct command:
```
python flow_train_ARC.py \
  --data-root raw_data/ARC-AGI \
  --train-split training \
  --eval-split evaluation \
  --num-demos 3 \
  --image-size 30 \
  --num-colors 12 \
  --epochs 20 \
  --learning-rate 2e-4 \
  --loss-on-target-only \
  --sample-steps 40 \
  --save-path saves/flow_context_vit/checkpoint_last.pt \
  --best-save-path saves/flow_context_vit/checkpoint_best.pt
```

### Important hyperparameters

#### Training hyperparmeters
| Parameter | Meaning |
|----------|-------|
| `epochs` | training epochs |
| `batch-size` | batch size |
| `learning-rate` | learning rate |
| `lr-scheduler` | default to cosine |
| `num-attempts` | how many random perspectives for per pseudo task in ttt |
| `ttt-num-each` | how many individual ttt runs per task (for ensemble)|

#### Model hyperparameters
| Parameter | Meaning |
|----------|-------|
| `architecture` | `vit` or `unet` |
| `image-size` | fix canvas size |
| `patch-size` | ViT patch size |
| `depth` | ViT transformer block num |
| `embed-dim` | ViT embedding dimension |
| `num-heads` | ViT number of attention heads |
| `unet-size`  | `big` or `medium` or `small`, only for Unet | 
| `num-colors` | 10 ARC pixel color + 1 background color + 1 border color for shape prediction |

#### Data settings
| Parameter | Meaning |
|----------|-------|
| `data-root` | `raw_data/ARC-AGI` or `raw_data/ARC-AGI-2`|
| `train-split` | path to test-time training augmented demonstration pairs for each task |
| `eval-split` | path to test-time training augmented infer inputs for each task|

#### Saving and loading
| Parameter | Meaning |
|----------|-------|
| `resume-checkpoint` | path to the offline-trained checkpoint |
| `resume-skip-task-token` | discard task token from offline training |
| `eval-save-name` | the dir to save ttt predictions under `outputs` folder |
|`save-path`| the dir to save the final offline training checkpoint |
| `best-save-path` | the dir to save the best offline training checkpoint with validation pairs |
## Other

### Verified results
Below is our verified result using VARC-ViT-18M (no ensembling) on the ARC-2 private test set from the Kaggle competition, using a slightly smaller computation setting with 4 color permutations instead of 9.

Leaderboard: https://www.kaggle.com/competitions/arc-prize-2025/leaderboard

![Kaggle results](assets/kaggle.png)

### Offline training logs

#### Offline training curve for ViT
We use test pairs from the original ARC-1 training tasks as our validation.
![ViT train curve](assets/offline_train_curve_ViT.png)


#### Offline training curve for Unet
![Unet train curve](assets/offline_train_curve_Unet.png)

## Citation
If you find our method or models helpful, please kindly cite our paper :)
```
@misc{hu2025arcvisionproblem,
      title={{ARC} Is a Vision Problem!}, 
      author={Keya Hu and Ali Cy and Linlu Qiu and Xiaoman Delores Ding and Runqian Wang and Yeyin Eva Zhu and Jacob Andreas and Kaiming He},
      year={2025},
      eprint={2511.14761},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.14761}, 
}
```
