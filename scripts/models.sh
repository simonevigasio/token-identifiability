# Model lists shared by scripts/run_all.sh and slurm/run.sbatch. Sourced, not executed.
#
# ELF variants are named by their feature tag (tokid.extract.model_tag), which sets the
# --elf-* options of tokid.extract:
#   elf-b              clean input, t = 1, denoiser mode, zero self-cond., self-cond. CFG 1
#   _t<t>              flow time t (noisy input z_t = t*z + (1-t)*2*eps)
#   _decoder           decoder step (model-mode tokens on)
#   _cfg<s>            self-conditioning CFG token s
#   _scpred            self-conditioning on ELF's own x-prediction (denoiser only)
#
# "elf-inference": ELF as in standard inference with the official unconditional sampling
# config (self-cond. CFG 3). The denoiser is only ever evaluated at t < 1, with
# self-conditioning; the final decoder step runs at t = 1 with zero self-conditioning.
ELF_INFERENCE="elf-b_decoder_cfg3 elf-b_t0.1_cfg3_scpred elf-b_t0.3_cfg3_scpred elf-b_t0.5_cfg3_scpred elf-b_t0.7_cfg3_scpred elf-b_t0.9_cfg3_scpred"

# expand_models "list" -> the list with the keyword elf-inference replaced
expand_models() {
  local out=() m
  for m in $1; do
    if [ "$m" = elf-inference ]; then
      read -r -a extra <<< "$ELF_INFERENCE"
      out+=("${extra[@]}")
    else
      out+=("$m")
    fi
  done
  echo "${out[*]}"
}
