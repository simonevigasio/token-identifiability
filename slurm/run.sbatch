#!/bin/bash
# Token identifiability experiments on AAU AI-LAB / AI Cloud. From the repository root:
#
#   sbatch --array=0-2 slurm/run.sbatch          # one GPU job per model: 0=t5-small 1=t5-base 2=elf-b
#   DATASETS="mrpc cola mnli" sbatch --array=0-2 slurm/run.sbatch   # + Appendix D
#   MODELS="elf-b" sbatch --array=0 slurm/run.sbatch                # a single model
#
# AI-LAB allows at most 12 h per job. Shortly before the limit, the job requeues itself
# (up to MAX_RESTARTS times). The experiments resume from their .partial checkpoints and
# finished steps are skipped. Logs: logs/tokid_<jobid>_<task>.out (appended across restarts).
#
#SBATCH --job-name=tokid
#SBATCH --output=logs/tokid_%A_%a.out
#SBATCH --error=logs/tokid_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@600

set -uo pipefail
source "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}/slurm/common.sh"

read -r -a ALL_MODELS <<< "${MODELS:-t5-small t5-base elf-b}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  export MODELS=${ALL_MODELS[$SLURM_ARRAY_TASK_ID]:?array index out of range for MODELS}
else
  export MODELS="${ALL_MODELS[*]}"
fi
DATASETS=${DATASETS:-mrpc}
export DEVICE=${DEVICE:-cuda}
export FOLDS=${FOLDS:-10}
export EXTRA=${EXTRA:-}
MAX_RESTARTS=${MAX_RESTARTS:-6}

# Everything was downloaded by slurm/setup.sh; never touch the network from compute nodes.
if [ "${ONLINE:-0}" != 1 ]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

restarts=$(scontrol show job "$SLURM_JOB_ID" | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
restarts=${restarts:-0}
echo "=== $(date) job $SLURM_JOB_ID on $(hostname), restart $restarts/$MAX_RESTARTS"
echo "models=$MODELS datasets=$DATASETS folds=$FOLDS container=$CONTAINER"

term_handler() {
  echo "=== $(date) time limit approaching"
  if [ "$restarts" -lt "$MAX_RESTARTS" ]; then
    echo "requeueing $SLURM_JOB_ID (progress is checkpointed)"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  echo "maximum restarts reached; resubmit the same command to continue"
  exit 1
}
trap term_handler SIGTERM

# Run in the background and `wait`, so the SIGTERM trap fires immediately.
# shellcheck disable=SC2086
srun "${CONTAINER_CMD[@]}" bash -c 'source "$0/bin/activate" && exec bash scripts/run_all.sh "$@"' \
  "$VENV" $DATASETS &
wait $!
status=$?
echo "=== $(date) finished with status $status"
exit $status
