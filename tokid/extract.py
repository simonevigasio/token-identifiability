"""Extract input embeddings and all hidden layers for a GLUE dev set (paper Sec. 4.1).

Every sentence (pair) is processed on its own (batch size 1, no padding), so the
features are exactly the model's hidden states for that input.

For the conditional ELF-B De-En checkpoint (``--elf-weights de-en``, dataset wmt14) the
"sentence" is the German prompt: ELF sees [prompt | English target | pad] and only the
prompt rows are kept (tokid.models.ELFCondExtractor).

Example:
    python -m tokid.extract --model elf --dataset mrpc
    python -m tokid.extract --model t5 --t5-name google-t5/t5-small --dataset mrpc
    python -m tokid.extract --model elf --variant elf-b-de-en_t0.3_g2_scpred --dataset wmt14
"""

import argparse
import os
import time
from pathlib import Path

import torch

from tokid.data import DATASET_NAMES, TRANSLATION, load_texts, load_translation
from tokid.models import DEFAULT_ELF_SOURCE, ELF_WEIGHTS, ELFObservation, load_elf, load_t5


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


ELF_DEFAULTS = {"elf_weights": "owt", "elf_t": 1.0, "elf_mode": "denoiser", "elf_cfg_scale": 1.0,
                "elf_guidance": 1.0, "elf_self_cond": "zero"}
ELF_HEADS = {"owt": "elf-b", "de-en": "elf-b-de-en"}


def model_tag(args) -> str:
    """File tag of a feature set; for ELF it encodes the checkpoint and every non-default
    observation setting, e.g. ``elf-b``, ``elf-b_decoder_cfg3``, ``elf-b_t0.3_cfg3_scpred``,
    ``elf-b-de-en_t0.3_g2_scpred``."""
    if args.model == "t5":
        return args.t5_name.split("/")[-1]
    tag = ELF_HEADS[args.elf_weights]
    if args.elf_t != 1.0:
        tag += f"_t{args.elf_t:g}"
    if args.elf_mode != "denoiser":
        tag += f"_{args.elf_mode}"
    if args.elf_cfg_scale != 1.0:
        tag += f"_cfg{args.elf_cfg_scale:g}"
    if args.elf_guidance != 1.0:
        tag += f"_g{args.elf_guidance:g}"
    if args.elf_self_cond != "zero":
        tag += f"_sc{args.elf_self_cond}"
    return tag


def parse_elf_tag(tag: str) -> dict:
    """Inverse of model_tag for ELF: ``elf-b_t0.3_cfg3_scpred`` -> the --elf-* settings."""
    head, *parts = tag.split("_")
    weights = {h: w for w, h in ELF_HEADS.items()}.get(head)
    if weights is None:
        raise ValueError(f"not an ELF tag: {tag!r}")
    out = {**ELF_DEFAULTS, "elf_weights": weights}
    for part in parts:
        if part == "decoder":
            out["elf_mode"] = "decoder"
        elif part.startswith("cfg"):
            out["elf_cfg_scale"] = float(part[3:])
        elif part.startswith("g"):
            out["elf_guidance"] = float(part[1:])
        elif part.startswith("sc"):
            out["elf_self_cond"] = part[2:]
        elif part.startswith("t"):
            out["elf_t"] = float(part[1:])
        else:
            raise ValueError(f"unknown part {part!r} in ELF tag {tag!r}")
    return out


def extract(extractor, tokenizer, texts, add_eos=True, log_every=100, targets=None):
    """targets: per-text target token ids, for the conditional checkpoint (texts are prompts)."""
    token_ids, sentence_ids, positions, tokens, hidden, lengths = [], [], [], [], None, []
    start = time.time()
    for s, text in enumerate(texts):
        ids = torch.tensor(tokenizer(text, add_special_tokens=add_eos)["input_ids"], dtype=torch.long)
        out = extractor(ids, index=s) if targets is None else extractor(ids, index=s, target_ids=targets[s])
        n = ids.numel()
        token_ids.append(ids)
        sentence_ids.append(torch.full((n,), s, dtype=torch.long))
        positions.append(torch.arange(n))
        tokens.append(out["tokens"].float().cpu())
        if hidden is None:
            hidden = [[] for _ in out["hidden"]]
        for l, h in enumerate(out["hidden"]):
            hidden[l].append(h.float().cpu())
        lengths.append(n)
        if log_every and (s + 1) % log_every == 0:
            print(f"  {s + 1}/{len(texts)} sentences ({time.time() - start:.0f}s)", flush=True)
    offsets = torch.zeros(len(texts) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(torch.tensor(lengths), 0)
    return {
        "token_ids": torch.cat(token_ids), "sentence_ids": torch.cat(sentence_ids),
        "positions": torch.cat(positions), "offsets": offsets,
        "tokens": torch.cat(tokens),                                   # (N, d_tok)
        "hidden": torch.stack([torch.cat(h) for h in hidden]),         # (L+1, N, d)
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["t5", "elf"], required=True)
    p.add_argument("--dataset", choices=DATASET_NAMES, default="mrpc")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Default: all (MRPC, CoLA) / first 500 (MNLI), as in the paper; "
                        "first 500 prompts of <= 64 tokens (WMT14).")
    p.add_argument("--pair-format", choices=["concat", "t5"], default="concat",
                   help="concat: 's1 s2' plain text (default). t5: T5's GLUE prompt, e.g. 'mrpc sentence1: ...'.")
    p.add_argument("--no-eos", action="store_true", help="Do not append T5's </s> token.")
    p.add_argument("--t5-name", default="google-t5/t5-small")
    p.add_argument("--elf-source", type=Path, default=DEFAULT_ELF_SOURCE)
    p.add_argument("--elf-checkpoint", type=Path, default=None, help="Local checkpoint file instead of the official one.")
    p.add_argument("--elf-weights", choices=sorted(ELF_WEIGHTS), default="owt",
                   help="owt: unconditional ELF-B (default). de-en: WMT14 German->English ELF-B; "
                        "probes the prompt rows, needs --dataset wmt14.")
    p.add_argument("--elf-t", type=float, default=1.0, help="Flow time; 1.0 = clean input (default).")
    p.add_argument("--elf-mode", choices=["denoiser", "decoder"], default="denoiser")
    p.add_argument("--elf-cfg-scale", type=float, default=1.0)
    p.add_argument("--elf-self-cond", choices=["zero", "pred"], default="zero",
                   help="Self-conditioning input: zero (default) or ELF's own x-prediction (denoiser only).")
    p.add_argument("--elf-guidance", type=float, default=1.0,
                   help="CFG on the prompt for the x-prediction fed back as self-conditioning "
                        "(de-en with --elf-self-cond pred; official sampling uses 2).")
    p.add_argument("--elf-noise-seed", type=int, default=0)
    p.add_argument("--variant", default=None,
                   help="ELF tag that sets all --elf-* options at once, e.g. elf-b_t0.3_cfg3_scpred.")
    p.add_argument("--cache-dir", default=None, help="HuggingFace cache directory.")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", type=Path, default=Path("features"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if args.variant is not None:
        if args.model != "elf":
            p.error("--variant is only for --model elf")
        try:
            settings = parse_elf_tag(args.variant)
        except ValueError as e:
            p.error(str(e))
        set_by_hand = [k for k, v in ELF_DEFAULTS.items() if getattr(args, k) != v]
        if set_by_hand:
            p.error(f"--variant cannot be combined with {', '.join('--' + k.replace('_', '-') for k in set_by_hand)}")
        vars(args).update(settings)
        if model_tag(args) != args.variant:
            p.error(f"non-canonical variant {args.variant!r}; write it as {model_tag(args)!r}")
    if args.model == "elf":  # fail before loading anything
        if args.elf_self_cond == "pred" and args.elf_mode == "decoder":
            p.error("ELF's decoder step always uses zero self-conditioning (--elf-self-cond zero)")
        if not 0.0 <= args.elf_t <= 1.0:
            p.error("--elf-t must be in [0, 1]")
        conditional = ELF_WEIGHTS[args.elf_weights]["conditional"]
        if args.elf_guidance != 1.0 and not (conditional and args.elf_self_cond == "pred"):
            p.error("--elf-guidance only applies to a conditional checkpoint with --elf-self-cond pred")
        if conditional != (args.dataset in TRANSLATION):
            p.error(f"--elf-weights {args.elf_weights} needs a {'prompt/target' if conditional else 'GLUE'} "
                    f"dataset ({', '.join(TRANSLATION) if conditional else 'not ' + ', '.join(TRANSLATION)})")

    if args.dataset in TRANSLATION and args.no_eos:
        p.error("--no-eos: the WMT14 prompts are tokenized with </s>, as in ELF's training data")

    out = args.out or args.out_dir / f"{model_tag(args)}_{args.dataset}.pt"
    if out.exists() and not args.overwrite:
        p.error(f"{out} exists (use --overwrite)")
    device = resolve_device(args.device)
    torch.set_grad_enabled(False)

    if args.model == "t5":
        extractor, tokenizer, meta = load_t5(args.t5_name, args.cache_dir, args.local_files_only, device)
    else:
        obs = ELFObservation(t=args.elf_t, mode=args.elf_mode, cfg_scale=args.elf_cfg_scale,
                             noise_seed=args.elf_noise_seed, self_cond=args.elf_self_cond, guidance=args.elf_guidance)
        extractor, tokenizer, meta = load_elf(args.elf_source, obs, args.cache_dir, args.local_files_only,
                                              args.elf_checkpoint, device, weights=args.elf_weights)
    targets = None
    if args.dataset in TRANSLATION:
        texts, targets = load_translation(args.dataset, args.max_examples, args.cache_dir)
        if args.model == "t5":
            targets = None  # T5 encodes the prompt alone, as ELF's frozen encoder does for prompt rows
    else:
        texts = load_texts(args.dataset, args.max_examples, args.pair_format, args.cache_dir)
    print(f"[extract] {meta['model']} on {args.dataset}: {len(texts)} examples, device={device}", flush=True)
    feats = extract(extractor, tokenizer, texts, add_eos=not args.no_eos, targets=targets)
    n_tok, n_sent = feats["token_ids"].numel(), len(texts)
    lengths = feats["offsets"].diff()
    feats["texts"] = texts
    feats["meta"] = {**meta, "tag": model_tag(args), "dataset": args.dataset, "pair_format": args.pair_format,
                     "add_eos": not args.no_eos, "num_sentences": n_sent, "num_tokens": n_tok,
                     "min_len": int(lengths.min()), "max_len": int(lengths.max()),
                     "mean_len": float(lengths.float().mean()),
                     "layer_names": ["input"] + [f"block {l}" for l in range(1, feats["hidden"].shape[0])]}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")  # atomic: a killed job never leaves a truncated file
    torch.save(feats, tmp)
    os.replace(tmp, out)
    print(f"[extract] {n_sent} sentences, {n_tok} tokens (len {feats['meta']['min_len']}-"
          f"{feats['meta']['max_len']}, mean {feats['meta']['mean_len']:.1f}); "
          f"hidden {tuple(feats['hidden'].shape)} -> {out}")


if __name__ == "__main__":
    main()
