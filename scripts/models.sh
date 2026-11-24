# Model lists shared by scripts/run_all.sh and slurm/run.sbatch. Sourced, not executed.
#
# ELF variants are named by their feature tag (tokid.extract.model_tag), which sets the
# --elf-* options of tokid.extract:
#   elf-b              clean input, t = 1, denoiser mode, zero self-cond., self-cond. CFG 1
#   _t<t>              flow time t (noisy input z_t = t*z + (1-t)*2*eps)
#   _decoder           decoder step (model-mode tokens on)
#   _cfg<s>            self-conditioning CFG token s
#   _g<s>              CFG on the prompt for the self-conditioning x-prediction (de-en, _scpred)
#   _scpred            self-conditioning on ELF's own x-prediction (denoiser only)
#   elf-b-de-en...     the WMT14 German->English checkpoint; probes the *prompt* rows (dataset wmt14)
#
# "elf-inference": ELF as in standard inference with the official unconditional sampling
# config (self-cond. CFG 3). The denoiser is only ever evaluated at t < 1, with
# self-conditioning; the final decoder step runs at t = 1 with zero self-conditioning.
ELF_INFERENCE="elf-b_decoder_cfg3 elf-b_t0.1_cfg3_scpred elf-b_t0.3_cfg3_scpred elf-b_t0.5_cfg3_scpred elf-b_t0.7_cfg3_scpred elf-b_t0.9_cfg3_scpred"

# "elf-de-en-inference": the whole conditional inference cycle of the translation
# checkpoint with the official cond_sampling_configs.yml (64-step ODE, CFG 2, self-cond. CFG 1):
# first step t = 0 (pure-noise target, zero self-cond.), denoiser steps t = 0.1 ... 0.9 with
# the guided x-prediction as self-cond., final decoder step at t = 1. Run with DATASETS=wmt14.
ELF_DEEN_INFERENCE="elf-b-de-en_t0 elf-b-de-en_t0.1_g2_scpred elf-b-de-en_t0.3_g2_scpred elf-b-de-en_t0.5_g2_scpred elf-b-de-en_t0.7_g2_scpred elf-b-de-en_t0.9_g2_scpred elf-b-de-en_decoder"

# expand_models "list" -> the list with the keywords elf-inference / elf-de-en-inference replaced
expand_models() {
  local out=() m
  for m in $1; do
    if [ "$m" = elf-inference ]; then
      read -r -a extra <<< "$ELF_INFERENCE"
      out+=("${extra[@]}")
    elif [ "$m" = elf-de-en-inference ]; then
      read -r -a extra <<< "$ELF_DEEN_INFERENCE"
      out+=("${extra[@]}")
    else
      out+=("$m")
    fi
  done
  echo "${out[*]}"
}
