#!/usr/bin/env bash
# (Re)generate all figures and summary tables from results/. From the repository root:
#
#   srun --mem=8G --cpus-per-task=2 --time=00:30:00 bash slurm/plots.sh
#
# results/<model>_<dataset>/      -> figures/<model>_<dataset>/ (PNG + PDF + summary.md)
# t5-small + elf-b of a dataset   -> figures/pipeline_<dataset>/
# t5-small + an ELF variant       -> figures/pipeline_<variant>_<dataset>/ (e.g. pipeline_decoder_cfg3_mrpc,
#                                    pipeline_de-en_t0.3_g2_scpred_wmt14)
# Unfinished experiments (only a .partial checkpoint, no .json yet) are skipped.
set -euo pipefail
source "$(dirname "$0")/common.sh"

shopt -s nullglob
found=0
for res in results/*_*/; do
  res=${res%/}
  name=$(basename "$res")
  jsons=("$res"/*.json)
  if [ ${#jsons[@]} -eq 0 ]; then
    echo "skip $res (no finished experiment yet)"
    continue
  fi
  in_container python -m tokid.plots --results "$res" --out "figures/$name"
  found=1
done
for t5 in results/t5-small_*/; do
  ds=${t5%/}; ds=${ds##*_}
  [ -f "results/t5-small_$ds/identity_token.json" ] || continue
  for elf in results/elf-b*_"$ds"/; do
    elf=${elf%/}
    [ -f "$elf/identity_token.json" ] || continue
    name=$(basename "$elf"); name=${name#elf-b}; name=${name%_"$ds"}; name=${name/#-/_}   # "", "_decoder_cfg3", "_de-en_t0"
    in_container python -m tokid.plots --pipeline "results/t5-small_$ds" "$elf" \
      --out "figures/pipeline${name}_$ds"
  done
done
[ "$found" = 1 ] || { echo "no results found in $PROJECT_DIR/results" >&2; exit 1; }
echo "Figures in $PROJECT_DIR/figures (tables: figures/*/summary.md)"
