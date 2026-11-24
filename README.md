# Token identifiability for T5 and ELF

Reproduces the **token identifiability** experiments of
Brunner et al., *On Identifiability in Transformers*, ICLR 2020 (arXiv:1908.04211),
Section 4 and Appendix B, with BERT replaced by:

| tag | model | layers | width | role |
|---|---|---|---|---|
| `t5-small` | `google-t5/t5-small` encoder | 6 | 512 | ELF's frozen text encoder |
| `t5-base`  | `google-t5/t5-base` encoder | 12 | 768 | same size as BERT-base in the paper |
| `elf-b`    | ELF-B (OWT, 105M, `embedded-language-flows/ELF-B-owt-torch`) | 12 | 768 | tokens → T5-small → ELF-B |
| `elf-b-de-en` | ELF-B (WMT14 De→En, 105M, `embedded-language-flows/ELF-B-de-en-torch`) | 12 | 768 | prompt rows of a conditional ELF |

## Method (as in the paper)

* Data: MRPC dev set, all 408 pairs (Sec. 4.1). With `cola` / `mnli` you get Appendix D:
  all 1043 CoLA dev sentences, and the first 500 MNLI-matched dev pairs.
* For every token i and layer l we store the input embedding `x_i` and the hidden state `e^l_i`.
* Token identifiability: `e^l_i` is identified if `NN(g_l(e^l_i)) = x_i`, where NN is a
  1-nearest-neighbour lookup **among the input tokens of the same sentence**. The reported
  number is the fraction of identified tokens.
* `g_l`: a linear perceptron without bias, or an MLP (1 hidden layer, 1000 units, GELU).
  Glorot-uniform init, Adam (lr 1e-4, β=(0.9, 0.999)), batch 256, early stopping after
  20 epochs without validation improvement. Loss and NN metric are both cosine or both L2.
  Naive baseline: `g = identity`.
* 10 folds, each a random 70/15/15 train/val/test split **by sentence** (App. B.1).
* Experiments:
  * `identity`: Fig. 2a (same layer), Fig. 2b / 8 / 9 / 10 (trained on layer l, tested on all
    layers), Fig. 7 (train vs. test).
  * `hidden2hidden`: App. B.4 / Fig. 11 (identify `e^{l-1}` from `e^l`).
  * `neighbours`: App. B.5 / Fig. 14–17 (recover `x_{i±k}`, k = 1, 2, 3).
  * Optional `--lr-search`: the App. B.1 grid {3e-3 … 3e-6}, picked on fold 0's validation set.

## What has to change for T5 and ELF

* **Input embedding `x_i`.** BERT's `x_i` = token + segment + position embedding. T5 has
  no absolute position embedding (it uses relative attention biases) and ELF uses RoPE, so
  `x_i` is the T5 token-embedding row (512-d). The same word twice in a sentence therefore
  gives two identical candidates. Retrieving either one counts as correct: the check is
  "retrieved vector == true `x_i`" (`tokid/identify.py`).
* **Input text.** Pairs are joined as plain text, `"<s1> <s2>"`, followed by T5's `</s>`
  (the analogue of BERT's `[CLS] … [SEP] … [SEP]` tokens being counted). T5 and ELF see the
  exact same token ids. `--pair-format t5` uses T5's own GLUE prompt instead, and
  `--no-eos` drops `</s>`.
* **T5 layers.** `hidden[0]` = token embeddings, `hidden[l]` = output of encoder block l.
  These are HF `output_hidden_states`, so the last one includes T5's final RMSNorm.
* **ELF.** Observed with a clean input, like the paper observes BERT on clean text:
  tokens → T5-small → `z = (h − 0)/0.2` → ELF-B at **t = 1** (clean), zero
  self-conditioning, CFG-scale token 1, denoiser mode. Only the language rows are kept
  (the 12 time / CFG / mode prefix rows are dropped). `hidden[0]` is the text rows going
  into block 1, after the self-conditioning and 128-d bottleneck projections.
  `hidden[1..12]` are the outputs of the 12 blocks. Two targets are run for ELF:
  * `--target token` (main result): recover the T5 token embedding (512-d) from ELF's
    768-d states. The dimensions differ, so **there is no naive baseline** for this target.
    Layer "in" (`hidden[0]`) is a real data point here: it shows how much identity
    survives T5 plus the rank-128 bottleneck.
  * `--target input`: recover ELF's own block-1 input rows (768-d). This is the literal
    analogue of BERT's `x_i`, and the naive baseline is defined.
  * Options: `--elf-t <t>` (noisy input `z_t = t·z + (1−t)·2·ε`, noise seeded per
    sentence), `--elf-mode decoder`, `--elf-cfg-scale <s>` and `--elf-self-cond pred`.
    They are not used by default.
* **ELF as in standard inference** (`MODELS=elf-inference`). The default `elf-b` (denoiser
  at t = 1) is a setting ELF never meets when generating. Official sampling
  (`.deps/elf-pytorch/src/utils/sampling_utils.py`) evaluates the denoiser only at t < 1:
  every ODE/SDE step calls the network at the step's *start* time, so the last step goes
  from `t_steps[-2]` to 1 without a call at t = 1. Then one decoder step at t = 1
  (model-mode tokens on, zero self-conditioning) maps the latent to tokens
  (`generation.py: _dlm_decode_batch`). `elf-inference` covers both, with self-cond. CFG
  token 3 as in `uncond_sampling_configs.yml`:

  | tag | mode | t | self-conditioning |
  |---|---|---|---|
  | `elf-b_decoder_cfg3` | decoder | 1 | zero |
  | `elf-b_t{0.1,0.3,0.5,0.7,0.9}_cfg3_scpred` | denoiser | 0.1 … 0.9 | ELF's own x̂ |

  `scpred`: during sampling the second half of the input is the previous step's
  x-prediction. Here a first forward pass with zero self-conditioning gives x̂, and the
  states are read from a second pass on `[z_t, x̂]`. This is exactly two sampler calls at
  the same t (`tests/test_models.py` checks it against upstream `_forward_sample_self_cond`).
  Note that sampling draws t from logit-normal(−1.5, 0.8) (median t ≈ 0.18, 90% below
  ≈ 0.38). The input `z_t` is a noised real sentence, as in training, not a sampler
  trajectory, so the true tokens are known.
  Variant names are feature tags: `--variant <tag>` sets all `--elf-*` options
  (`scripts/models.sh`).
* **Pipeline figure.** `figures/pipeline_<ds>/` puts T5-small blocks 1–6 and ELF-B
  in, 1–12 on one axis. It shows how token identity decays through ELF's whole stack.
* **Conditional ELF: prompt rows** (`MODELS="t5-small elf-de-en-inference"`, dataset
  `wmt14`). The OWT checkpoint is unconditional, so every row is generated. The
  translation checkpoint (`ELF-B-de-en-torch`, German → English) works on a 128-row
  sequence `[prompt (≤ 64) | target | </s> pad]` (`train_de-en_ELF-B.yml`). The prompt
  rows are the clean T5 latent of the German sentence and are kept fixed during
  sampling, while the target rows are generated. Here the queries and candidates are the
  **prompt rows** only: `x_i` = T5 embedding of prompt token i, `e^l_i` = ELF state of
  prompt row i. Everything follows upstream conditional generation
  (`generation.py: test_generation_cond`, `cond_sampling_configs.yml`: 64-step ODE,
  CFG 2, self-cond. CFG 1):
  * T5 latent with the upstream masks: prompt rows attend only to the prompt (so they
    equal T5 on the prompt alone); target/pad rows attend to every non-pad token.
  * Target/pad rows: `z_t = t·x0 + (1−t)·2·ε` (seeded); prompt rows: clean `x0`.
  * ELF gets no attention mask, as in sampling. The states are read from the
    conditional pass.

  | tag | step of inference | t | self-conditioning half |
  |---|---|---|---|
  | `elf-b-de-en_t0` | first step | 0 (target = pure noise) | zeros, prompt rows = clean prompt |
  | `elf-b-de-en_t{0.1,0.3,0.5,0.7,0.9}_g2_scpred` | later denoiser steps | 0.1 … 0.9 | guided x̂ (CFG 2), prompt restored |
  | `elf-b-de-en_decoder` | final decoder step | 1 (mode tokens on) | zeros |

  x̂ is upstream `_forward_sample` at the same t (`x_uncond + 2·(x_cond − x_uncond)`),
  i.e. what the next ODE step gets as `x_pred_prev`. The target rows hold the reference
  translation (`x0`), noised to t. Data: the first 500 WMT14 validation pairs whose
  prompt fits in 64 tokens (≈ 16.5k prompt tokens, mean length 33; `--max-examples`).
  The prompt ids are the dataset's `condition_input_ids`, which equal T5's tokenization
  with `</s>`. `t5-small` on `wmt14` encodes the same prompts and gives the pipeline
  figure `figures/pipeline_de-en_<variant>_wmt14/`.

Implementation notes:
* The G layers of a model are trained together as G independent probes: parameters have a
  leading layer dimension, the loss is the sum of per-layer losses, and early stopping and
  best-weight restore are per layer. Because gradients never mix across layers and Adam is
  element-wise, this is identical to training each layer on its own.
  `tests/test_probes.py` checks that weights and stopping epochs match exactly.
* "L2 loss" is the squared Euclidean distance. Validation *loss* is what early stopping monitors.
* The ELF code is the official `pytorch_elf` branch at a pinned commit (cloned to
  `.deps/elf-pytorch`, never modified). The weights are pinned to an exact HF revision and
  use the EMA parameters. The extractor gives bit-identical states to
  `../sipit-elf`'s independent wrapper.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch            # pick the CUDA build matching the machine
pip install -e ".[test]"
python scripts/prepare_elf.py   # clones github.com/lillian039/ELF @ b29d883 (pytorch_elf)
python -m pytest -q             # CPU, no downloads needed
```

## Running on AAU AI-LAB / AI Cloud (Slurm + Singularity)

Follows the CLAAUDIA docs (hpc.aau.dk). Jobs run in the cluster's PyTorch Singularity
container: `/ceph/container/pytorch/` on AI-LAB, `/home/container/pytorch/` on AI Cloud.
The newest one is picked automatically; override with `CONTAINER=...`. Extra packages go
into a venv (`.venv-aau`) built on top of the container, so torch/CUDA always come from
the container. Run everything from the repository root on the front-end node:

```bash
# 1. once, needs internet: venv + packages, ELF code, models + GLUE into .cache/huggingface, tests
srun --mem=24G --cpus-per-task=8 --gres=gpu:1 --time=01:00:00 bash slurm/setup.sh
#    (if the compute node has no internet, run `bash slurm/setup.sh` directly on the front-end)

# 2. experiments: one 1-GPU job per model (0=t5-small, 1=t5-base, 2=elf-b), MRPC
sbatch --array=0-2 slurm/run.sbatch
#    + Appendix D datasets:
DATASETS="mrpc cola mnli" sbatch --array=0-2 slurm/run.sbatch
#    ELF as in standard inference (0=decoder t=1, 1-5=denoiser t=0.1..0.9):
MODELS=elf-inference sbatch --array=0-5 slurm/run.sbatch
#    translation ELF-B, prompt rows (0=t5-small, 1=t=0, 2-6=t=0.1..0.9 scpred, 7=decoder):
MODELS="t5-small elf-de-en-inference" DATASETS=wmt14 sbatch --array=0-7 slurm/run.sbatch

# 3. monitor
squeue --me
tail -f logs/tokid_<jobid>_2.out
```

* Resources per job: 1 GPU, 8 CPUs, 24 GB RAM, 12 h. Those fit AI-LAB's limits: ≤24 GB and
  ≤15 CPUs per GPU, ≤12 h per job, ≤4 GPUs per job, ≤8 GPUs and 8 jobs per user.
* **12 h limit:** Slurm signals the job 10 min before the limit, and the job requeues itself
  (`scontrol requeue`, at most `MAX_RESTARTS=6` times). Experiments checkpoint after every
  trained probe (`results/.../*.json.partial`), so a restarted job resumes where it stopped;
  finished steps are skipped. If a job dies anyway, submit the same command again.
* Jobs run with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, since setup
  already downloaded everything. Set `ONLINE=1` to allow downloads.
* Other variables: `MODELS`, `DATASETS`, `FOLDS`, `EXTRA="--lr-search"`, `CONTAINER`, `VENV`,
  `HF_HOME`, `MAX_RESTARTS`. Also set: `PYTHONNOUSERSITE=1` (ignore `~/.local`),
  `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`, `PYTHONUNBUFFERED=1`.
* AI Cloud: add a partition if your account needs one, e.g. `sbatch -p prioritized --array=0-2 ...`.
* Figures are made automatically at the end of each job. To (re)generate all of them:
  `srun --mem=8G --cpus-per-task=2 --time=00:30:00 bash slurm/plots.sh`. Then copy them
  to your laptop: `rsync -av <user>@<front-end>:<repo>/figures ./`.

The first run downloads T5-small/base, the two ELF-B checkpoints (~0.8 GB each), GLUE and
the WMT14 De-En validation set from the Hugging Face Hub. If setup already ran before the
translation experiment existed, run `slurm/setup.sh` again (step 1). It is idempotent:
it keeps the venv and ELF code and only downloads what is missing. Set `HF_HOME` or `HF_CACHE=...` to choose where they are stored.

## Run

Everything (MRPC, all three models, all experiments, figures):

```bash
bash scripts/run_all.sh                 # MRPC (main text)
bash scripts/run_all.sh mrpc cola mnli  # + Appendix D
```

Environment knobs: `MODELS="t5-small elf-b"` (or `elf-inference`, `elf-de-en-inference`
with dataset `wmt14`, or any ELF tag such as `elf-b_t0.3_cfg3_scpred`), `FOLDS=10`, `DEVICE=cuda`, `EXTRA="--lr-search"`,
`PY=python`. Finished steps are skipped, so after an interruption just re-run the same command.
The `neighbours` experiment costs about 6× `identity`, since it trains one probe set per offset.

Single steps:

```bash
python -m tokid.extract --model elf --dataset mrpc                               # features/elf-b_mrpc.pt
python -m tokid.extract --model t5 --t5-name google-t5/t5-small --dataset mrpc   # features/t5-small_mrpc.pt
python -m tokid.extract --model elf --variant elf-b_decoder_cfg3 --dataset mrpc  # features/elf-b_decoder_cfg3_mrpc.pt
python -m tokid.extract --model elf --variant elf-b-de-en_t0.3_g2_scpred --dataset wmt14  # prompt rows
python -m tokid.experiments --features features/elf-b_mrpc.pt --experiment identity --target token
python -m tokid.experiments --features features/elf-b_mrpc.pt --experiment hidden2hidden
python -m tokid.experiments --features features/elf-b_mrpc.pt --experiment neighbours
python -m tokid.plots --results results/elf-b_mrpc --out figures/elf-b_mrpc
python -m tokid.plots --pipeline results/t5-small_mrpc results/elf-b_mrpc --out figures/pipeline_mrpc
```

## Outputs

* `features/<model>_<dataset>.pt`: token ids, sentence ids, positions, `tokens` (N×d_tok),
  `hidden` ((L+1)×N×d), texts, metadata (revisions, lengths).
* `results/<model>_<dataset>/*.json`: per-fold, per-layer rates (`test`, `train`, `val`, the
  `cross` layer matrix), best epochs, learning rates, splits.
* `figures/pipeline[_<variant>]_<dataset>/`: T5-small + ELF on one axis, for `elf-b` and every ELF variant.
* `figures/<model>_<dataset>/`: `summary.md` (all tables, mean ± std over folds),
  `identity_token/fig2a_identifiability.png`, `fig2b_crosslayer_linear_cos.png`,
  `fig7_generalization_l2.png`, `fig8–10_*`, `hidden2hidden/fig11_*`,
  `neighbours_token/fig14–17_*` (PNG + PDF).

## Layout

```
tokid/data.py         GLUE dev sets and text formatting
tokid/models.py       T5 / ELF-B hidden-state extractors
tokid/extract.py      CLI: features
tokid/splits.py       sentence-level 70/15/15 folds
tokid/probes.py       batched linear/MLP probes, training with per-layer early stopping
tokid/identify.py     within-sentence nearest-neighbour identification
tokid/experiments.py  CLI: identity / hidden2hidden / neighbours
tokid/plots.py        CLI: figures + summary tables
tokid/prefetch.py     CLI: download all models/datasets once
scripts/prepare_elf.py, scripts/run_all.sh, scripts/models.sh (ELF variants)
slurm/common.sh, slurm/setup.sh, slurm/run.sbatch   AAU AI-LAB / AI Cloud
```
