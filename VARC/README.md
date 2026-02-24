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

### Optional: Prepare BARC as extra training data
If you want to add BARC to training, convert it into VARC's expected extra-data layout:
`raw_data/BARC/tasks/<task_id>.json`, where each file is a list of `{input, output}` pairs.

From a local BARC dump (directory or json/jsonl file):
```
python script/prepare_barc.py \
  --input-root /path/to/BARC_source \
  --output-root raw_data/BARC \
  --overwrite
```

Directly from a Hugging Face dataset:
```
python script/prepare_barc.py \
  --hf-dataset <org_or_user>/<dataset_name> \
  --hf-split train \
  --output-root raw_data/BARC \
  --overwrite
```

Then enable it in training:
- Offline / RL: add `--include-barc --barc-path raw_data/BARC`
- Flow: add `--include-barc --barc-path raw_data/BARC`

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
- Default loss applies to all frames; pass `--loss-on-target-only` to train only on the final solution frame.
- Train data augmentation is off by default in this flow pipeline.
  - enable random translation + random resolution scaling per `(input, output)` pair with:
  - `--flow-train-translation-aug --flow-train-resolution-aug`

Evaluation:
- Keep demo frames and query input clean
- Initialize only the last solution frame from noise
- Denoise only that last frame via Euler integration
- Report exact-match sample/task accuracy
- Optional attention mode:
  - `--framewise-causal-attention` restricts attention so frame `f` only sees frames `<= f`
  - `--attention-backend flex|sdpa|auto` picks backend (`auto` uses flex on CUDA when available)

Run:
```
bash script/flow_train_context_vit.sh
```
DDP script:
```
bash script/flow_train_context_vit_ddp.sh
```

DDP run (`torchrun`, 4 GPUs example):
```
torchrun --nproc_per_node=4 flow_train_ARC.py \
  --ddp \
  --data-root raw_data/ARC-AGI \
  --train-split training \
  --eval-split evaluation \
  --num-demos 3 \
  --image-size 30 \
  --num-colors 12 \
  --epochs 20 \
  --learning-rate 2e-4 \
  --bf16-autocast
```
In DDP mode, evaluation is sharded across GPUs and metrics are aggregated globally; only rank 0 logs/checkpoints.

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
  --bf16-autocast \
  --framewise-causal-attention \
  --attention-backend flex \
  --sample-steps 40 \
  --save-path saves/flow_context_vit/checkpoint_last.pt \
  --best-save-path saves/flow_context_vit/checkpoint_best.pt
```

### 7. Discrete flow matching from scratch (logit-based)
This repo also includes `flow_train_discrete_ARC.py`, which keeps the same context construction but uses a strict discrete flow-matching setup aligned with [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching/tree/main).

Discrete training path:
- Sample per-frame noise levels `t_f` independently.
- Uses the same default train-time pair augmentation as above (translation + resolution scaling).
- Build a mixture discrete path with a uniform source token and clean ARC token target:
  - `sigma_t = exp(-beta * t)` (`--discrete-rate = beta`)
  - sample `x_t = x_0` with probability `sigma_t`, else `x_t = x_1`
- Model predicts logits for `p_theta(x_1 | x_t, t)` over colors.
- Loss uses the generalized KL form from `flow_matching` (`MixturePathGeneralizedKL`) on valid pixels:
  - `-beta * [ p_theta(x_t|x_t,t) - delta_{x_t,x_1} + (1-delta_{x_t,x_1}) log p_theta(x_1|x_t,t) ]`
  - default trains on all frames; pass `--loss-on-target-only` to train only on the final solution frame
- Sampling uses a discrete Euler update equivalent to `MixtureDiscreteEulerSolver`:
  - choose proposal `x_1` from logits (`--reverse-sampler sample|argmax`)
  - apply jump probability `1 - exp(-beta * dt)` per step on the target frame

Evaluation:
- Keep demos + query input clean.
- Initialize only the last solution frame as random tokens.
- Run discrete Euler updates for that frame only, measure exact-match sample/task accuracy.
- Optional attention mode:
  - `--framewise-causal-attention` restricts attention so frame `f` only sees frames `<= f`
  - `--attention-backend flex|sdpa|auto` picks backend (`auto` uses flex on CUDA when available)

Run:
```
bash script/flow_train_context_vit_discrete.sh
```
DDP script:
```
bash script/flow_train_context_vit_discrete_ddp.sh
```

DDP run (`torchrun`, 4 GPUs example):
```
torchrun --nproc_per_node=4 flow_train_discrete_ARC.py \
  --ddp \
  --data-root raw_data/ARC-AGI \
  --train-split training \
  --eval-split evaluation \
  --num-demos 3 \
  --image-size 30 \
  --num-colors 12 \
  --epochs 20 \
  --learning-rate 2e-4 \
  --bf16-autocast \
  --discrete-rate 5.0
```
In DDP mode, evaluation is sharded across GPUs and metrics are aggregated globally; only rank 0 logs/checkpoints.

Direct command:
```
python flow_train_discrete_ARC.py \
  --data-root raw_data/ARC-AGI \
  --train-split training \
  --eval-split evaluation \
  --num-demos 3 \
  --image-size 30 \
  --num-colors 12 \
  --epochs 20 \
  --learning-rate 2e-4 \
  --bf16-autocast \
  --discrete-rate 5.0 \
  --framewise-causal-attention \
  --attention-backend flex \
  --reverse-sampler sample \
  --sample-steps 40 \
  --wandb-num-vis-samples 8 \
  --wandb-vis-scale 8 \
  --save-path saves/flow_context_vit_discrete/checkpoint_last.pt \
  --best-save-path saves/flow_context_vit_discrete/checkpoint_best.pt
```

When `--use-wandb` is enabled, eval logs also include generated sample panels (`eval/generations`) with:
- demo input/output pairs
- query input
- predicted output
- ground-truth output

### 8. Hyperparameter search
You can run automatic hyperparameter search for `flow_train_discrete_ARC.py` with the included launcher:

Random search:
```bash
python script/hparam_search_flow_discrete.py \
  --mode random \
  --num-trials 12 \
  --workspace . \
  --output-dir sweeps/flow_discrete_hparam \
  --include-rearc \
  --include-barc \
  --bf16-autocast \
  --use-wandb
```
Default ranking is `--metric-key eval_loss --metric-goal minimize`.

Grid search:
```bash
python script/hparam_search_flow_discrete.py \
  --mode grid \
  --search-space-json '{"learning-rate":[1e-4,2e-4],"depth":[10,14],"discrete-rate":[3.0,5.0],"sample-steps":[20,40]}' \
  --workspace . \
  --output-dir sweeps/flow_discrete_grid \
  --include-rearc \
  --include-barc \
  --bf16-autocast \
  --use-wandb
```

The script writes:
- trial logs under `sweeps/.../trial_xxxx/train.log`
- checkpoints under each trial folder
- `results.jsonl` and `results.csv` summary files

You can also use existing packages/services:
- Weights & Biases Sweeps (template: `script/wandb_sweep_flow_discrete.yaml`)
- Optuna
- Ray Tune

The provided W&B sweep template uses:
- Bayesian search
- Hyperband early termination (ASHA-style multi-fidelity pruning)
- `eval/loss` minimization objective

W&B sweep example:
```bash
wandb sweep script/wandb_sweep_flow_discrete.yaml
wandb agent <entity>/<project>/<sweep_id>
```

Run multiple agents across GPUs:
```bash
script/run_wandb_sweep_agents.sh <entity>/<project>/<sweep_id>
```

Note:
- The sweep template launches each trial with DDP (`torch.distributed.run`) on `8` GPUs and passes `--ddp`.
- Therefore, run one W&B agent per node for this sweep template.

### 9. Pretraining discrete flow matching on 2D cellular automata (CA-2D)
This repo now includes a CA-2D pretraining path aligned with the data-generation strategy from [Edge of Chaos: Text Generation with One-Dimensional Cellular Automata](https://arxiv.org/abs/2410.02536), adapted from 1D CA language modeling to 2D grid prediction:

- on-the-fly generated CA trajectories from random initial states
- long rollouts (`1000` steps)
- random spatiotemporal windows (random time + random spatial crop)
- mixed prediction horizons (`1` and `5` steps)
- heterogeneous dynamics (ordered / complex / chaotic life-like rule families)

#### Generate CA-2D dataset (ARC JSON format)
The generator writes tasks under `raw_data/CA-2D-DFM/data/{training,evaluation}/*.json`, compatible with `flow_train_discrete_ARC.py`.
```bash
python script/generate_ca_2d_dataset.py \
  --output-root raw_data/CA-2D-DFM \
  --num-train-tasks 2048 \
  --num-eval-tasks 256 \
  --train-examples-per-task 12 \
  --test-examples-per-task 4 \
  --sim-size 96 \
  --crop-size 31 \
  --rollout-steps 1000 \
  --warmup-steps 60 \
  --densities "0.15,0.25,0.35,0.45,0.55" \
  --horizons "1,5" \
  --rule-profile paper_like \
  --overwrite
```

#### Local pretraining launcher
```bash
bash script/pretrain_discrete_ca2d.sh
```

#### UBELIX batch launcher (testbed and full runs)
`script/pretrain_discrete_ca2d_ubelix.sbatch` uses explicit GPU partition/type/count:
- `#SBATCH --partition=gpu`
- `#SBATCH --gpus-per-node=h100:1` (max high-end config allowed for current `job_gratis` QoS)

Submit:
```bash
ssh ss24i671@submit03.unibe.ch \
  "cd ~/Documents/solve_agi/VARC && sbatch script/pretrain_discrete_ca2d_ubelix.sbatch"
```

Optional quick testbed override (short run):
```bash
ssh ss24i671@submit03.unibe.ch \
  "cd ~/Documents/solve_agi/VARC && TRAIN_TASKS=128 EVAL_TASKS=32 TRAIN_EPOCHS=1 sbatch script/pretrain_discrete_ca2d_ubelix.sbatch"
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
