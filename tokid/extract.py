"""Extract input embeddings and all hidden layers for a GLUE dev set (paper Sec. 4.1).

Every sentence (pair) is processed on its own (batch size 1, no padding), so the
features are exactly the model's hidden states for that input.

Example:
    python -m tokid.extract --model elf --dataset mrpc
    python -m tokid.extract --model t5 --t5-name google-t5/t5-small --dataset mrpc
"""

import argparse
import os
import time
from pathlib import Path

import torch

from tokid.data import DATASETS, load_texts
from tokid.models import DEFAULT_ELF_SOURCE, ELFObservation, load_elf, load_t5


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def model_tag(args) -> str:
    if args.model == "t5":
        return args.t5_name.split("/")[-1]
    tag = "elf-b"
    if args.elf_t != 1.0:
        tag += f"_t{args.elf_t:g}"
    if args.elf_mode != "denoiser":
        tag += f"_{args.elf_mode}"
    return tag


def extract(extractor, tokenizer, texts, add_eos=True, log_every=100):
    token_ids, sentence_ids, positions, tokens, hidden, lengths = [], [], [], [], None, []
    start = time.time()
    for s, text in enumerate(texts):
        ids = torch.tensor(tokenizer(text, add_special_tokens=add_eos)["input_ids"], dtype=torch.long)
        out = extractor(ids, index=s)
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
    p.add_argument("--dataset", choices=sorted(DATASETS), default="mrpc")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Default: all (MRPC, CoLA) / first 500 (MNLI), as in the paper.")
    p.add_argument("--pair-format", choices=["concat", "t5"], default="concat",
                   help="concat: 's1 s2' plain text (default). t5: T5's GLUE prompt, e.g. 'mrpc sentence1: ...'.")
    p.add_argument("--no-eos", action="store_true", help="Do not append T5's </s> token.")
    p.add_argument("--t5-name", default="google-t5/t5-small")
    p.add_argument("--elf-source", type=Path, default=DEFAULT_ELF_SOURCE)
    p.add_argument("--elf-checkpoint", type=Path, default=None)
    p.add_argument("--elf-t", type=float, default=1.0, help="Flow time; 1.0 = clean input (default).")
    p.add_argument("--elf-mode", choices=["denoiser", "decoder"], default="denoiser")
    p.add_argument("--elf-cfg-scale", type=float, default=1.0)
    p.add_argument("--elf-noise-seed", type=int, default=0)
    p.add_argument("--cache-dir", default=None, help="HuggingFace cache directory.")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", type=Path, default=Path("features"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    out = args.out or args.out_dir / f"{model_tag(args)}_{args.dataset}.pt"
    if out.exists() and not args.overwrite:
        p.error(f"{out} exists (use --overwrite)")
    device = resolve_device(args.device)
    torch.set_grad_enabled(False)

    if args.model == "t5":
        extractor, tokenizer, meta = load_t5(args.t5_name, args.cache_dir, args.local_files_only, device)
    else:
        obs = ELFObservation(t=args.elf_t, mode=args.elf_mode, cfg_scale=args.elf_cfg_scale,
                             noise_seed=args.elf_noise_seed)
        extractor, tokenizer, meta = load_elf(args.elf_source, obs, args.cache_dir, args.local_files_only,
                                              args.elf_checkpoint, device)
    texts = load_texts(args.dataset, args.max_examples, args.pair_format, args.cache_dir)
    print(f"[extract] {meta['model']} on {args.dataset}: {len(texts)} examples, device={device}", flush=True)
    feats = extract(extractor, tokenizer, texts, add_eos=not args.no_eos)
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
