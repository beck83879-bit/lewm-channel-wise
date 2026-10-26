#!/usr/bin/env bash
set -euo pipefail

arm=${1:?original192, dense336, random, or learned}
gpu=${2:?physical GPU index}
stamp=${3:?unique experiment identifier}
case "$arm" in
  original192|dense336|random|learned) ;;
  *) exit 2 ;;
esac
case "$gpu" in 0|1|2|3) ;; *) exit 2 ;; esac
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export CUDA_VISIBLE_DEVICES="$gpu"
export STABLEWM_HOME=/home/wangtz/.lewm-channelwise
export CUBE_DATASET=/home/wangtz/.lewam/datasets/cube.h5
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export PIP_INDEX_URL=https://mirror.nju.edu.cn/pypi/web/simple
python=/home/wangtz/anaconda3/envs/lewam/bin/python
run="cube_${arm}_seed3072_${stamp}"
log="$STABLEWM_HOME/logs/$run.log"
mkdir -p "$STABLEWM_HOME/logs" "$STABLEWM_HOME/evaluations"
if [[ -e "$STABLEWM_HOME/checkpoints/$run" || -e "$log" ]]; then
  echo "Refusing to overwrite existing run $run" >&2
  exit 2
fi
used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used > 1000 )); then
  echo "GPU $gpu has $used MiB in use; not starting." >&2
  exit 3
fi
exec > >(tee -a "$log") 2>&1
trap 'code=$?; printf "PIPELINE_EXIT_CODE=%s\n" "$code"; printf "%s\n" "$code" > "$STABLEWM_HOME/logs/$run.exit"' EXIT
printf 'RUN=%s GPU=%s SOURCE=%s\n' "$run" "$gpu" "$root"
"$python" -u -m channelwise.train "arm=$arm" "subdir=$run" "output_model_name=$run" \
  +trainer.enable_progress_bar=false +trainer.log_every_n_steps=20
checkpoint="$STABLEWM_HOME/checkpoints/$run/weights_epoch_10.pt"
test -s "$checkpoint"
for seed in 42 43 44; do
  "$python" -u -m channelwise.evaluate "policy=$checkpoint" "seed=$seed" \
    "output.directory=$STABLEWM_HOME/evaluations/${run}_eval${seed}"
done
echo "PIPELINE_COMPLETE $run"
