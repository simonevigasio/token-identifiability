#!/bin/bash
# Token identifiability experiments on AAU AI-LAB / AI Cloud. From the repository root:
#
#   sbatch --array=0-2 slurm/run.sbatch          # one GPU job per model: 0=t5-small 1=t5-base 2=elf-b
#   DATASETS="mrpc cola mnli" sbatch --array=0-2 slurm/run.sbatch   # + Appendix D
#   MODELS="elf-b" sbatch --array=0 slurm/run.sbatch                # a single model
#   MODELS="elf-inference" sbatch --array=0-5 slurm/run.sbatch      # ELF as in standard inference:
#       0=decoder t=1, 1-5=denoiser t=0.1,0.3,0.5,0.7,0.9 (see scripts/models.sh)
#   MODELS="t5-small elf-de-en-inference" DATASETS=wmt14 sbatch --array=0-7 slurm/run.sbatch
#       translation ELF-B, prompt rows: 0=t5-small (pipeline baseline), 1=t=0 first step,
#       2-6=denoiser t=0.1..0.9, 7=decoder t=1
#   MODELS="t5-small elf-xsum-inference" DATASETS=docmix sbatch --array=0-7 slurm/run.sbatch
#       summarization ELF-B, English prompts from 6 domains, real sampling trajectory:
#       0=t5-small, 1=t=0, 2-6=t=0.1..0.9, 7=decoder; identity with 3 splits (sentence,
#       leave-one-domain-out, unseen word types)
#   Model selection of g_l (17 sources, locked test set), two steps; the ELF extraction
#   writes all 7 trajectory points in one job, so it runs before the selection jobs:
#     jid=$(MODELS="t5-small elf-b-xsum_t0" DATASETS=multidoc STAGE=extract sbatch --parsable --array=0-1 slurm/run.sbatch)
#     MODELS="t5-small elf-xsum-inference" DATASETS=multidoc STAGE=select \
#       sbatch --dependency=afterok:$jid --array=0-7 slurm/run.sbatch
#       0=t5-small, 1=t=0, 2-6=t=0.1..0.9, 7=decoder
#   Shared probes across the XSum trajectory (after the extraction above; the hidden states
#   of all 7 points, ~26 GiB, are spread over the job's GPUs, so ask for 2). Up to ~7x / ~13x
#   a per-point selection, so allow many restarts:
#     MAX_RESTARTS=30 MODELS=elf-xsum-shared DATASETS=multidoc STAGE=select \
#       sbatch --gres=gpu:2 --mem=48G --array=0-1 slurm/run.sbatch
#       0=time-independent (one probe per layer), 1=layer-independent (one probe per point);
#       each compares with the per-point runs' results. On AI Cloud, 6-day jobs on the
#       prioritized partition need far fewer restarts (command-line options override #SBATCH):
#     MAX_RESTARTS=5 MODELS=elf-xsum-shared DATASETS=multidoc STAGE=select \
#       sbatch -p prioritized --time=6-00:00:00 --gres=gpu:2 --mem=48G --array=0-1 slurm/run.sbatch
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
# The V100 nodes (compute capability 7.0) are too old for the container's PyTorch
# (needs >= 7.5): jobs there fail with "no kernel image is available for execution".
#SBATCH --exclude=nv-ai-02,nv-ai-03
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@600

set -uo pipefail
source "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}/slurm/common.sh"

source scripts/models.sh
read -r -a ALL_MODELS <<< "$(expand_models "${MODELS:-t5-small t5-base elf-b}")"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  export MODELS=${ALL_MODELS[$SLURM_ARRAY_TASK_ID]:?array index out of range for MODELS}
else
  export MODELS="${ALL_MODELS[*]}"
fi
DATASETS=${DATASETS:-mrpc}
export DEVICE=${DEVICE:-cuda}
export FOLDS=${FOLDS:-10}
export EXTRA=${EXTRA:-}
export STAGE=${STAGE:-all} SEL_EXTRA=${SEL_EXTRA:-}   # multidoc (scripts/run_all.sh)
MAX_RESTARTS=${MAX_RESTARTS:-6}

# Everything was downloaded by slurm/setup.sh; never touch the network from compute nodes.
if [ "${ONLINE:-0}" != 1 ]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

restarts=$(scontrol show job "$SLURM_JOB_ID" | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
restarts=${restarts:-0}
echo "=== $(date) job $SLURM_JOB_ID on $(hostname), restart $restarts/$MAX_RESTARTS"
echo "models=$MODELS datasets=$DATASETS folds=$FOLDS stage=$STAGE container=$CONTAINER"

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
