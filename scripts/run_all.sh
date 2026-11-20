#!/usr/bin/env bash
# Full reproduction of Brunner et al. (2020), Sec. 4 + App. B token identifiability,
# for T5-small (ELF's frozen encoder), T5-base (BERT-base sized) and ELF-B.
#
#   bash scripts/run_all.sh                # MRPC (main text)
#   bash scripts/run_all.sh mrpc cola mnli # + Appendix D datasets
#
# Environment overrides:
#   MODELS="t5-small t5-base elf-b"  FOLDS=10  DEVICE=auto  PY=python
#   EXTRA="--lr-search"   (extra flags for every experiment)
#   HF_CACHE=/path        (HuggingFace cache dir; default: HF default)
# Finished steps are skipped, so the script can simply be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS=(mrpc)
if [ $# -gt 0 ]; then DATASETS=("$@"); fi
MODELS=${MODELS:-"t5-small t5-base elf-b"}
FOLDS=${FOLDS:-10}
DEVICE=${DEVICE:-auto}
PY=${PY:-python}
EXTRA=${EXTRA:-}
CACHE_ARGS=()
[ -n "${HF_CACHE:-}" ] && CACHE_ARGS=(--cache-dir "$HF_CACHE")

if [[ " $MODELS " == *" elf-b "* ]] && [ ! -d .deps/elf-pytorch ]; then
  $PY scripts/prepare_elf.py
fi

run_exp() {  # features experiment out [args...]
  local feats=$1 exp=$2 out=$3; shift 3
  if [ -f "$out" ]; then echo "skip $out"; return; fi
  $PY -m tokid.experiments --features "$feats" --experiment "$exp" --folds "$FOLDS" --device "$DEVICE" \
      --out "$out" $EXTRA "$@"
}

for ds in "${DATASETS[@]}"; do
  for model in $MODELS; do
    feats=features/${model}_${ds}.pt
    if [ ! -f "$feats" ]; then
      case $model in
        elf-b) $PY -m tokid.extract --model elf --dataset "$ds" --device "$DEVICE" ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} --out "$feats" ;;
        t5-*)  $PY -m tokid.extract --model t5 --t5-name "google-t5/$model" --dataset "$ds" --device "$DEVICE" \
                   ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} --out "$feats" ;;
        *) echo "unknown model $model"; exit 1 ;;
      esac
    fi
    res=results/${model}_${ds}
    run_exp "$feats" identity      "$res/identity_token.json"   --target token   # Fig. 2a, 2b, 7, 8-10
    run_exp "$feats" hidden2hidden "$res/hidden2hidden.json"                     # App. B.4, Fig. 11
    run_exp "$feats" neighbours    "$res/neighbours_token.json" --target token   # App. B.5, Fig. 14-17
    if [ "$model" = elf-b ]; then
      # Literal analogue of BERT's x_i: the rows entering ELF block 1.
      run_exp "$feats" identity "$res/identity_input.json" --target input
    fi
    $PY -m tokid.plots --results "$res" --out "figures/${model}_${ds}"
  done
  # Whenever both exist (also when each model ran in its own Slurm array task).
  if [ -f "results/t5-small_${ds}/identity_token.json" ] && [ -f "results/elf-b_${ds}/identity_token.json" ]; then
    $PY -m tokid.plots --pipeline "results/t5-small_${ds}" "results/elf-b_${ds}" --out "figures/pipeline_${ds}"
  fi
done
echo "All done. Tables: figures/*/summary.md"
