# Shared settings for AAU AI-LAB / AI Cloud (Slurm + Singularity). Sourced, not executed.
#
# Overridable from the environment:
#   PROJECT_DIR  repository root (default: the directory you ran sbatch/srun from)
#   CONTAINER    PyTorch .sif (default: the one used at setup, else the newest
#                /ceph/container/pytorch/pytorch_*.sif (AI-LAB) or
#                /home/container/pytorch/pytorch_*.sif (AI Cloud))
#   VENV         venv with the extra packages (default: $PROJECT_DIR/.venv-aau)
#   HF_HOME      Hugging Face cache (default: $PROJECT_DIR/.cache/huggingface)

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
cd "$PROJECT_DIR" || exit 1
if [ ! -f scripts/run_all.sh ]; then
  echo "error: run this from the repository root (PROJECT_DIR=$PROJECT_DIR)" >&2
  exit 1
fi

VENV=${VENV:-$PROJECT_DIR/.venv-aau}
if [ -z "${CONTAINER:-}" ] && [ -f "$VENV/.container" ]; then
  CONTAINER=$(cat "$VENV/.container")   # the venv only works with the container it was built in
fi
if [ -z "${CONTAINER:-}" ]; then
  for d in /ceph/container/pytorch /home/container/pytorch; do
    if [ -d "$d" ]; then
      CONTAINER=$(ls -1 "$d"/pytorch_*.sif 2>/dev/null | sort -V | tail -n 1)
      break
    fi
  done
fi
if [ -z "${CONTAINER:-}" ] || [ ! -f "$CONTAINER" ]; then
  echo "error: no PyTorch container found; set CONTAINER=/path/to/pytorch_XX.YY.sif" >&2
  exit 1
fi

# $HOME is mounted in the container automatically; anything else must be bound.
BIND_ARGS=()
case "$PROJECT_DIR/" in
  "$HOME"/*) ;;
  *) BIND_ARGS=(--bind "$PROJECT_DIR") ;;
esac
CONTAINER_CMD=(singularity exec --nv ${BIND_ARGS[@]+"${BIND_ARGS[@]}"} "$CONTAINER")

export HF_HOME=${HF_HOME:-$PROJECT_DIR/.cache/huggingface}
export PYTHONNOUSERSITE=1          # ignore ~/.local; only container + venv packages
export PYTHONUNBUFFERED=1          # live progress in the Slurm log
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

# in_container CMD ARGS...  -> run inside the container with the venv activated
in_container() {
  "${CONTAINER_CMD[@]}" bash -c 'source "$0/bin/activate" && exec "$@"' "$VENV" "$@"
}
