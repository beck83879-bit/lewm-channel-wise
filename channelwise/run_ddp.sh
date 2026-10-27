#!/usr/bin/env bash
set -euo pipefail

arm=${1:?original192 or learned}
stamp=${2:?unique experiment identifier}
port=${3:?unique torchrun master port}
case "$arm" in original192|learned) ;; *) exit 2 ;; esac
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export STABLEWM_HOME=/home/wangtz/.lewm-channelwise-ddp-20261027
export CUBE_DATASET=/home/wangtz/.lewam/datasets/cube.h5
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export PIP_INDEX_URL=https://mirror.nju.edu.cn/pypi/web/simple
python=/home/wangtz/anaconda3/envs/lewam/bin/python
run="cube_${arm}_ddp4_seed3072_${stamp}"
log="$STABLEWM_HOME/logs/$run.log"
mkdir -p "$STABLEWM_HOME/logs" "$STABLEWM_HOME/evaluations"
if [[ -e "$STABLEWM_HOME/checkpoints/$run" || -e "$log" ]]; then
  echo "Refusing to overwrite existing run $run" >&2
  exit 2
fi
exec > >(tee -a "$log") 2>&1
trap 'code=$?; printf "PIPELINE_EXIT_CODE=%s\n" "$code"; printf "%s\n" "$code" > "$STABLEWM_HOME/logs/$run.exit"' EXIT
printf 'RUN=%s GPUS=%s SOURCE=%s MASTER_PORT=%s\n' "$run" "$CUDA_VISIBLE_DEVICES" "$root" "$port"
"$python" -m torch.distributed.run --nproc_per_node=4 --master_port="$port" \
  --module channelwise.train --config-name=channelwise_cube_ddp \
  "arm=$arm" "subdir=$run" "output_model_name=$run" \
  +trainer.enable_progress_bar=false +trainer.log_every_n_steps=20
echo "TRAINING_STAGE_COMPLETE $run"
checkpoint="$STABLEWM_HOME/checkpoints/$run/weights_epoch_10.pt"
test -s "$checkpoint"
if [[ "$arm" == original192 ]]; then export CUDA_VISIBLE_DEVICES=0; else export CUDA_VISIBLE_DEVICES=1; fi
for seed in 42 43 44; do
  "$python" -u -m channelwise.evaluate "policy=$checkpoint" "seed=$seed" \
    "output.directory=$STABLEWM_HOME/evaluations/${run}_eval${seed}"
done
echo "PIPELINE_COMPLETE $run"
