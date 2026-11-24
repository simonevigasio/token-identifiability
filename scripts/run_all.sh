#!/usr/bin/env bash
# Full reproduction of Brunner et al. (2020), Sec. 4 + App. B token identifiability,
# for T5-small (ELF's frozen encoder), T5-base (BERT-base sized) and ELF-B.
#
#   bash scripts/run_all.sh                # MRPC (main text)
#   bash scripts/run_all.sh mrpc cola mnli # + Appendix D datasets
#
# Environment overrides:
#   MODELS="t5-small t5-base elf-b"  FOLDS=10  DEVICE=auto  PY=python
#   MODELS="elf-inference"    (ELF as in standard inference: decoder t=1 + denoiser t<1,
#                              see scripts/models.sh; can be mixed with other models)
#   MODELS="t5-small elf-de-en-inference" bash scripts/run_all.sh wmt14
#                             (translation ELF-B, prompt rows, whole inference cycle)
#   MODELS="t5-small elf-xsum-inference" bash scripts/run_all.sh docmix
#                             (summarization ELF-B, prompt rows on the real sampling trajectory;
#                              identity only, with 3 splits: sentence, domain, unseen-token)
#   MODELS="t5-small elf-xsum-inference" bash scripts/run_all.sh multidoc
#                             (summarization ELF-B, 17 sources, prompts of 64-512 tokens: model
#                              selection of g_l with a locked test set, tokid.selection)
#   STAGE=extract|select|all  (multidoc only; default all. The ELF extraction writes all 7
#                              trajectory points at once, so on Slurm run STAGE=extract first,
#                              then the selection jobs; see slurm/run.sbatch)
#   SEL_EXTRA="--n-random 64" (extra flags for tokid.selection)
#   EXTRA="--lr-search"   (extra flags for every experiment)
#   HF_CACHE=/path        (HuggingFace cache dir; default: HF default)
# Finished steps are skipped, so the script can simply be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS=(mrpc)
if [ $# -gt 0 ]; then DATASETS=("$@"); fi
source scripts/models.sh   # ELF variant names, elf-inference
MODELS=$(expand_models "${MODELS:-t5-small t5-base elf-b}")
FOLDS=${FOLDS:-10}
DEVICE=${DEVICE:-auto}
PY=${PY:-python}
EXTRA=${EXTRA:-}
CACHE_ARGS=()
[ -n "${HF_CACHE:-}" ] && CACHE_ARGS=(--cache-dir "$HF_CACHE")

if [[ " $MODELS " == *" elf-b"* ]] && [ ! -d .deps/elf-pytorch ]; then
  $PY scripts/prepare_elf.py
fi

run_exp() {  # features experiment out [args...]
  local feats=$1 exp=$2 out=$3; shift 3
  if [ -f "$out" ]; then echo "skip $out"; return; fi
  $PY -m tokid.experiments --features "$feats" --experiment "$exp" --folds "$FOLDS" --device "$DEVICE" \
      --out "$out" $EXTRA "$@"
}

STAGE=${STAGE:-all}
SEL_EXTRA=${SEL_EXTRA:-}

run_multidoc() {  # model: extraction (all ELF points at once) and/or model selection
  local model=$1 feats=features/${1}_multidoc.pt res=results/${1}_multidoc done=features/${1}_multidoc.pt
  # The ELF extraction writes all 7 points and finishes with the decoder one.
  [[ $model == elf-b-xsum* ]] && done=features/elf-b-xsum_decoder_multidoc.pt
  if [ "$STAGE" != select ] && [ ! -f "$done" ]; then
    case $model in
      elf-b-xsum*) $PY -m tokid.extract --model elf --elf-weights xsum --dataset multidoc --device "$DEVICE" \
                     ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} ;;
      t5-*)        $PY -m tokid.extract --model t5 --t5-name "google-t5/$model" --dataset multidoc \
                     --device "$DEVICE" ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} ;;
      *) echo "multidoc runs t5-* and the elf-b-xsum_* variants, not $model"; exit 1 ;;
    esac
  fi
  [ "$STAGE" = extract ] && return
  if [ ! -f "$feats" ]; then echo "missing $feats: run STAGE=extract first"; exit 1; fi
  if [ -f "$res/selection.json" ]; then
    echo "skip $res/selection.json"
  else
    $PY -m tokid.selection --features "$feats" --device "$DEVICE" --out "$res/selection.json" $SEL_EXTRA
  fi
  $PY -m tokid.plots --results "$res" --out "figures/${model}_multidoc"
}

for ds in "${DATASETS[@]}"; do
  for model in $MODELS; do
    if [ "$ds" = multidoc ]; then
      run_multidoc "$model"
      continue
    fi
    feats=features/${model}_${ds}.pt
    if [ ! -f "$feats" ]; then
      case $model in
        elf-b*) $PY -m tokid.extract --model elf --variant "$model" --dataset "$ds" --device "$DEVICE" \
                   ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} --out "$feats" ;;
        t5-*)  $PY -m tokid.extract --model t5 --t5-name "google-t5/$model" --dataset "$ds" --device "$DEVICE" \
                   ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} --out "$feats" ;;
        *) echo "unknown model $model"; exit 1 ;;
      esac
    fi
    res=results/${model}_${ds}
    if [ "$ds" = docmix ]; then
      # How general is g_l: in-domain, leave-one-domain-out, unseen word types.
      run_exp "$feats" identity "$res/identity_token.json"              --target token
      run_exp "$feats" identity "$res/identity_token_domain.json"       --target token --split domain
      run_exp "$feats" identity "$res/identity_token_unseen-token.json" --target token --split unseen-token
      $PY -m tokid.plots --results "$res" --out "figures/${model}_${ds}"
      continue
    fi
    run_exp "$feats" identity      "$res/identity_token.json"   --target token   # Fig. 2a, 2b, 7, 8-10
    run_exp "$feats" hidden2hidden "$res/hidden2hidden.json"                     # App. B.4, Fig. 11
    run_exp "$feats" neighbours    "$res/neighbours_token.json" --target token   # App. B.5, Fig. 14-17
    if [[ $model == elf-b* ]]; then
      # Literal analogue of BERT's x_i: the rows entering ELF block 1.
      run_exp "$feats" identity "$res/identity_input.json" --target input
    fi
    $PY -m tokid.plots --results "$res" --out "figures/${model}_${ds}"
  done
  # Whenever both exist (also when each model ran in its own Slurm array task).
  for model in $MODELS; do
    [[ $model == elf-b* ]] || continue
    name=${model#elf-b}; name=${name/#-/_}   # "" for elf-b, else e.g. "_decoder_cfg3", "_de-en_t0"
    if [ -f "results/t5-small_${ds}/identity_token.json" ] && [ -f "results/${model}_${ds}/identity_token.json" ]; then
      $PY -m tokid.plots --pipeline "results/t5-small_${ds}" "results/${model}_${ds}" --out "figures/pipeline${name}_${ds}"
    fi
  done
done
echo "All done. Tables: figures/*/summary.md"
