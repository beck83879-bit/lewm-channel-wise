# LeWM Channelwise Cube Experiment

This is an additive port of the channel selection mechanism from
`ZitongWang018/flow-jepa@f5763aa` onto
`lucas-maes/le-wm@8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
The original root `module.py`, `jepa.py`, `train.py`, and `eval.py` are unchanged.
`channel_routing.py` is copied byte-for-byte from the Flow-JEPA experiment;
its source SHA-256 is recorded in the tests.

## Preserved Mechanism

- 336 residual coordinates: 144 always active plus one of four 48-D groups.
- A separate router for each of six blocks, binary top-1 selection.
- Genuine selected-submatrix AdaLN, attention, feed-forward and final readout.
- Random routing for 100 optimizer steps; then learned scores plus usage bias.
- Usage EMA 0.95, acceptable load 20%-30%, bias LR 0.05 and bound +/-2.
- Matched selected/unselected swaps every eight optimizer steps after warmup.
- Ranking loss weight 0.01, margins 1e-5 absolute plus 1e-3 relative.
- Frozen probe routes, matched dropout RNG, detached router inputs for credit.
- Inactive coordinates retain their residual values within a predictor call.

## Necessary LeWM Adaptations

LeWM retains its original next-latent MSE, SIGReg, target gradient, ViT-Tiny/14,
projectors, causal attention, action-conditioned AdaLN and autoregressive MPC.
There is no flow matching, Euler integration or additional cross-attention.
LeWM's own inner attention/MLP LayerNorms are preserved in selected form.

The router receives current hidden-state summary, visible latent history,
visible history difference, visible action summary and a zero time feature.
LeWM has no flow time, so inventing one would change the original model.
Every target position uses only its visible prefix; all three one-step targets
are still supervised. A clip-wide choice shared across all three targets would
allow future observations to influence earlier predictions.

The prediction projector contains BatchNorm. The two credit probes use cloned
buffers through `torch.func.functional_call`, preserving training-mode batch
statistics without modifying the main forward's running buffers or autograd
versions. Main-loss gradients still reach the target encoder. The probe uses
the post-projector next-latent error, replacing Flow-JEPA's velocity error.

The active matrices are dispatched by selected group using ordinary PyTorch
operations. This preserves the selected-coordinate math without reusing the
Flow-specific cross-attention, nine-way modulation, Triton kernel or CUDA graph.
No claim of identical runtime, FLOPs or speedup is made.

## Four Matched Arms

| GPU | Arm | Predictor width | Active width | Router credit |
|---|---|---:|---:|---|
| 0 | original192 | 192 | 192 | off |
| 1 | dense336 | 336 | 336 | off |
| 2 | random | 336 | 192 | off |
| 3 | learned | 336 | 192 | on |

These are four independent single-GPU runs, each batch128, seed3072, 10 epochs,
history3, one-step prediction, four input frames and frameskip5. All use AdamW
LR5e-5, weight decay1e-3, SIGReg0.1 with 17 knots/1024 projections, BF16 mixed
precision and gradient clipping1.0. They reuse the same 90/10 random split.
Single-GPU batch128 avoids changing SIGReg's local-batch statistic to batch32.

The original repository's `LinearWarmupCosineAnnealingLR` configuration is
retained. The installed stable-pretraining resolves its warmup/total duration
in optimizer steps and steps it after updates, as does the previous LeWM
environment. The YAML's interval label alone is not evidence of epoch stepping.
Smoke checkpoint inspection verifies configured LR, weight decay and scheduler
state. Training manifests and full resumable checkpoints record actual state.

The Cube HDF5 already on A100 is byte-identical to the previous LeWM data on
A6000. Both complete files have SHA-256:
`0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625`.
The A100 file is `/home/wangtz/.lewam/datasets/cube.h5` (101942558720 bytes).
No new dataset download or copy is required; the filename does not select the
LeWAM model. All model imports point to this LeWM extension directory.

## Run and Verify

Existing A100 interpreter: `/home/wangtz/anaconda3/envs/lewam/bin/python`.
This is reused for dependencies only; no package is changed. It provides
PyTorch2.8.0+cu128, stable-worldmodel0.1.1, stable-pretraining0.1.7,
Lightning2.6.6 and transformers4.57.6. Future installation should use NJU's
mirror; the runner sets `PIP_INDEX_URL=https://mirror.nju.edu.cn/pypi/web/simple`.

From the repository root:

```bash
python -m unittest discover -s tests -v
export CUBE_DATASET=/home/wangtz/.lewam/datasets/cube.h5
export STABLEWM_HOME=/home/wangtz/.lewm-channelwise
CUDA_VISIBLE_DEVICES=3 python -m channelwise.smoke --arm learned --batch 128
bash channelwise/run_arm.sh learned 3 UNIQUE_RUN_ID
```

The runner refuses existing run directories or logs and refuses an occupied
GPU. Each completed training run automatically evaluates epoch10 on seeds
42,43,44, 50 starts each. Native repository CEM settings are 300 candidates,
30 iterations, top30; planning horizon5, execute5 blocks, action_block5,
history3, goal offset25 and interaction budget50. This is explicitly CEM30,
not the CEM10 used in some earlier local LeWM reports. All four arms use the
same evaluation protocol; do not merge old CEM10 scores into this comparison.

Evaluation saves model/config hashes, exact sampled starts, reset seeds and
per-case success flags. It preserves the upstream start-selection rule and
uses a seeded reset. These are native dataset-replay tests, not a newly held-out
trajectory generalization benchmark. Clean evaluation is queued; no Gaussian
condition is silently substituted into LeWM's native protocol.

Training writes CSV metrics, each epoch's model/config, and a resumable
`training_state/last.ckpt` containing optimizer, scheduler and routing buffers.
Offline logs do not rely on the laptop or a proxy tunnel. The source is run in
tmux. Original data, earlier runs and other GPUs are not modified.

## Validation Boundaries

Tests cover original-disabled output equality; explicit reduced dense reference
versus selected block outputs and input gradients; full fixed-route predictor
equivalence; prefix causality; credit-only router gradients; original target
gradients; untouched BatchNorm/RNG; route counts; replay; checkpoint reload and
upstream source hashes. Full-size GPU checks cover batch128 forward/backward,
credit activation after warmup and 300 candidate inference. Real-data smoke
training additionally checks save/resume and environment evaluation plumbing.

Synthetic and short smoke results are not benchmark results. Passing checks
does not establish performance superiority or the absence of every possible
bug. No full DDP check is claimed: the experiment uses independent single GPUs.
