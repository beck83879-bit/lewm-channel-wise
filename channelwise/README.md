# LeWM Channelwise Cube Experiment

## Status: Corrected Four-GPU Comparison Completed

On 2026-10-27, fresh Cube baseline (`original192`) and channelwise (`learned`)
runs were launched. Each uses four DDP ranks on GPUs 0-3; the two jobs share
these GPUs, with distinct rendezvous ports. Each rank uses batch32, giving
global batch128. Both use seed3072 and 10 epochs. No historical or smoke
checkpoint initializes these runs.

The dedicated source snapshot is `/home/wangtz/LeWM-channelwise-ddp-20261027`;
outputs are under `/home/wangtz/.lewm-channelwise-ddp-20261027`.
The tmux sessions are `lewm-ddp-cube-baseline` and
`lewm-ddp-cube-channelwise`. `run_ddp.sh` queues epoch10 evaluation with
seeds42/43/44, 50 starts each. Both training and all three evaluations completed
normally with pipeline exit code 0. Baseline scores were 70%, 82%, 74%;
channel-wise scores were 78%, 84%, 70%. The means are 75.3% and 77.3%.
All paired start manifests match. Eight cases improved, five regressed and
137 were unchanged. These are three evaluation seeds of one training seed,
not three independent training repetitions.

The four runs launched on 2026-10-26 were stopped at the user's request on
2026-10-27. Their checkpoints and logs are retained. They used grouped training
dispatch, which changed per-sample dropout masks when swap probes changed a
sample's group. Do not use those runs as validation of a faithful port.

The corrected code uses the source's batch-order-preserving selected linear
operation and explicit attention operation. Their function source is checked
against `f5763aa`; `sparse_linear.py` and `channel_routing.py` are byte copies.
LeWM parameter-bank mapping still needs six-way AdaLN and its inner LayerNorms.
Tests cover sample-wise dropout masks under changed routes, actual 192-D
matrix operands, inactive residual preservation, and reduced-LeWM equivalence.

Design reference: the "correct experiment" section of the user's
[Channel-Wise report](https://ycnj8ybm4mcw.feishu.cn/wiki/Zwz9ww0JGixsP0krBLdcQlHGnCc).
It specifies 336 total / 192 active dimensions. The earlier full-width
compute-then-mask experiment is explicitly marked incorrect in that report.

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
are still supervised. This is the recommended LeWM interface: the source
`FlowTrajectoryPredictor` itself states that causal routing requires
"separate per-prefix decisions". A clip-wide choice shared across all three
targets would allow future observations to influence earlier predictions.
This preserves LeWM's supervised targets, but does not make LeWM's predictor
identical to Flow-JEPA's predictor.

The prediction projector contains BatchNorm. The two credit probes use cloned
buffers through `torch.func.functional_call`, preserving training-mode batch
statistics without modifying the main forward's running buffers or autograd
versions. Main-loss gradients still reach the target encoder. The probe uses
the post-projector next-latent error, replacing Flow-JEPA's velocity error.

The active matrices use the source's gather, selected batched matrix multiply,
and scatter sequence, without regrouping samples between dropout operations.
The source's optional Triton helpers are retained as dependencies of the
copied primitives; this adapter uses the ordinary batched operation in both
training and inference. It does not claim the source's optimized inference
latency. No Flow-specific cross-attention or nine-way modulation is introduced.

## Fairness Boundaries

The migrated route is the source's default independent candidate route. The
optional shared-candidate route and CUDA-graph acceleration are not implemented
here, and must not be reported as enabled.

Equal active width is not equal total parameters, active parameters, FLOPs,
wall-clock time, or memory. Include the router and full-width input/condition
projections in accounting. Prefix routing also evaluates histories of lengths
1, 2, and 3, whereas the original dense predictor can evaluate all three targets
in one causal pass. No equal-compute claim is made.

Before launching a new comparison, both LeWM baseline and channelwise must use
the same data/split, optimizer, seed, epochs, per-rank batch, world size, precision
and evaluation starts/budget. Four ranks of batch32 and one rank of batch128
have different local SIGReg/BatchNorm statistics despite equal global batch.
Do not compare their scores as if training conditions were identical. The
previous four-arm arrangement is historical, not a new approved launch plan.

## Stopped Historical Runs

| GPU | Arm | Predictor width | Active width | Router credit |
|---|---|---:|---:|---|
| 0 | original192 | 192 | 192 | off |
| 1 | dense336 | 336 | 336 | off |
| 2 | random | 336 | 192 | off |
| 3 | learned | 336 | 192 | on |

These were four independent single-GPU runs, each batch128, seed3072, 10 epochs,
history3, one-step prediction, four input frames and frameskip5. All use AdamW
LR5e-5, weight decay1e-3, SIGReg0.1 with 17 knots/1024 projections, BF16 mixed
precision and gradient clipping1.0. They reuse the same 90/10 random split.
Single-GPU batch128 has different local SIGReg statistics from the source's
four-GPU batch32-per-rank training. These settings were not an exact copy of
the source experiment protocol.

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
```

The above are verification commands, not formal training commands. Do not
resume the historical stopped checkpoints with this code. The new comparison
uses `channelwise/run_ddp.sh` and `channelwise_cube_ddp.yaml`.
The historical `run_arm.sh` runner refuses existing run directories or logs and refuses an occupied
GPU. Each completed training run automatically evaluates epoch10 on seeds
42,43,44, 50 starts each. Native repository CEM settings are 300 candidates,
30 iterations, top30; planning horizon5, execute5 blocks, action_block5,
history3, goal offset25 and interaction budget50. This is explicitly CEM30,
not the CEM10 used in some earlier local LeWM reports. Both new arms use the
same evaluation protocol; do not merge old CEM10 scores into this comparison.

Evaluation saves model/config hashes, exact sampled starts, reset seeds and
per-case success flags. It preserves the upstream start-selection rule and
uses a seeded reset. These are native dataset-replay tests, not a newly held-out
trajectory generalization benchmark. Clean evaluation is complete; no Gaussian
condition is silently substituted into LeWM's native protocol.

Training writes CSV metrics, each epoch's model/config, and a resumable
`training_state/last.ckpt` containing optimizer, scheduler and routing buffers.
Offline logs do not rely on the laptop or a proxy tunnel. The historical tmux
sessions have been terminated. Original data and earlier results are retained.

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
bug. Real Cube DDP checks passed on four ranks: baseline three updates,
channelwise 106 updates (including warmup and swap credit), and restoration
from step106 to step107. All ranks had identical routing usage/bias buffers.
The step104 probe rate was 1, valid rate 0.9921875 and credit loss 0.69643.
These smoke runs shorten the scheduler; their terminal learning rate is not
the formal training schedule. Runtime JSON files record each formal rank's
sampler, initial step, batch size and initial learning rate.
