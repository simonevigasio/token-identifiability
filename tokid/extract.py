"""Extract input embeddings and all hidden layers for a GLUE dev set (paper Sec. 4.1).

Every sentence (pair) is processed on its own (batch size 1, no padding), so the
features are exactly the model's hidden states for that input.

For the conditional ELF-B De-En checkpoint (``--elf-weights de-en``, dataset wmt14) the
"sentence" is the German prompt: ELF sees [prompt | English target | pad] and only the
prompt rows are kept (tokid.models.ELFCondExtractor). For the XSum checkpoint
(``--elf-weights xsum``, dataset docmix) the prompts are English documents from six
domains and the states are read during a real sampling run (ELFTrajectoryExtractor).

Example:
    python -m tokid.extract --model elf --dataset mrpc
    python -m tokid.extract --model t5 --t5-name google-t5/t5-small --dataset mrpc
    python -m tokid.extract --model elf --variant elf-b-de-en_t0.3_g2_scpred --dataset wmt14
    python -m tokid.extract --model elf --variant elf-b-xsum_t0.3 --dataset docmix
    python -m tokid.extract --model elf --elf-weights xsum --dataset multidoc   # all 7 points
    python -m tokid.extract --model t5 --t5-name google-t5/t5-small --dataset multidoc
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from tokid.data import (DATASET_NAMES, DATASETS, DOCMIX, MULTIDOC, TRANSLATION, load_docmix, load_multidoc, load_texts,
                        load_translation)
from tokid.models import DEFAULT_ELF_SOURCE, ELF_WEIGHTS, ELFObservation, load_elf, load_t5


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


ELF_DEFAULTS = {"elf_weights": "owt", "elf_t": 1.0, "elf_mode": "denoiser", "elf_cfg_scale": 1.0,
                "elf_guidance": 1.0, "elf_self_cond": "zero"}
ELF_HEADS = {"owt": "elf-b", "de-en": "elf-b-de-en", "xsum": "elf-b-xsum"}
# Which datasets each checkpoint is run on.
ELF_DATASETS = {"owt": set(DATASETS), "de-en": set(TRANSLATION), "xsum": {*DOCMIX, *MULTIDOC}}


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


def extract(extractor, tokenizer, texts, add_eos=True, log_every=100, targets=None, ids_list=None):
    """targets: per-text target token ids, for the conditional checkpoint (texts are prompts).
    ids_list: pre-tokenized inputs (used instead of tokenizing ``texts``)."""
    token_ids, sentence_ids, positions, tokens, hidden, lengths = [], [], [], [], None, []
    start = time.time()
    for s, text in enumerate(texts):
        if ids_list is not None:
            ids = torch.tensor(ids_list[s], dtype=torch.long)
        else:
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


def extract_queries(run, labels, prompts, out_paths, embedding, base_meta, log_every=100):
    """Query-position features of many prompts, for several observation points at once.

    run(ids, index) -> {label: {"tokens", "hidden"}}. For every label, the hidden states
    of the prompts' ``queries`` positions go to a float16 .npy file (written in place via
    memmap); the token arrays, the T5 embedding matrix and the meta go to the .pt file,
    which is written last, so its existence means the extraction is complete. Progress
    is checkpointed every ``log_every`` prompts and an interrupted run resumes."""
    ids = [torch.tensor(p["ids"], dtype=torch.long) for p in prompts]
    lengths = torch.tensor([len(i) for i in ids])
    offsets = torch.zeros(len(ids) + 1, dtype=torch.long)
    offsets[1:] = lengths.cumsum(0)
    n_q = torch.tensor([len(p["queries"]) for p in prompts])
    q_off = torch.zeros(len(ids) + 1, dtype=torch.long)
    q_off[1:] = n_q.cumsum(0)
    query_index = torch.cat([offsets[i] + torch.tensor(p["queries"], dtype=torch.long) for i, p in enumerate(prompts)])
    query_doc = torch.arange(len(ids)).repeat_interleave(n_q)
    npy = {l: out_paths[l].with_name(out_paths[l].stem + ".hidden.npy") for l in labels}
    partial = {l: npy[l].with_name(npy[l].name + ".partial") for l in labels}
    progress = out_paths[labels[0]].with_name(out_paths[labels[0]].stem + ".progress.json")
    state = {"labels": labels, "num_prompts": len(ids), "num_queries": int(q_off[-1]), "done": 0}
    # A label's states live in .partial until finalized, then in .npy (and its .pt is written).
    stored = lambda l: partial[l] if partial[l].exists() else npy[l]  # noqa: E731
    if progress.exists() and all(stored(l).exists() for l in labels):
        saved = json.loads(progress.read_text())
        if {k: v for k, v in saved.items() if k != "done"} == {k: v for k, v in state.items() if k != "done"}:
            state["done"] = saved["done"]
            print(f"[extract] resuming at prompt {state['done']}/{len(ids)}", flush=True)
    mm = {}
    start = time.time()
    for i in range(state["done"], len(ids)):
        out = run(ids[i], i)
        if i == 0:
            torch.testing.assert_close(out[labels[0]]["tokens"].float().cpu(), embedding[ids[0]])
        for l in labels:
            h = torch.stack(out[l]["hidden"])[:, prompts[i]["queries"]]
            if l not in mm:
                shape = (h.shape[0], int(q_off[-1]), h.shape[-1])
                mode = "r+" if state["done"] > 0 else "w+"
                mm[l] = np.lib.format.open_memmap(partial[l], mode=mode, dtype=np.float16, shape=shape)
            h16 = h.to(torch.float16)
            if not bool(torch.isfinite(h16).all()):
                raise OverflowError(f"prompt {i}, {l}: hidden states exceed the float16 range")
            mm[l][:, q_off[i]:q_off[i + 1]] = h16.cpu().numpy()
        if (i + 1) % log_every == 0 or i + 1 == len(ids):
            for m in mm.values():
                m.flush()
            state["done"] = i + 1
            tmp = progress.with_name(progress.name + ".tmp")  # atomic: a kill never leaves a torn file
            tmp.write_text(json.dumps(state))
            os.replace(tmp, progress)
            print(f"  {i + 1}/{len(ids)} prompts ({time.time() - start:.0f}s)", flush=True)
    # Finalize label by label. Each step is idempotent, so a kill in the middle of this
    # loop is resumed too: finished labels are skipped, a renamed .npy is reused.
    for l in labels:
        if out_paths[l].exists():
            continue
        if l not in mm:  # nothing left to do in this session
            mm[l] = np.load(stored(l), mmap_mode="r")
        shape = mm[l].shape
        del mm[l]
        if partial[l].exists():
            os.replace(partial[l], npy[l])
        feats = {"format": "queries", "token_ids": torch.cat(ids), "offsets": offsets, "query_index": query_index,
                 "query_doc": query_doc, "query_offsets": q_off, "embedding": embedding,
                 "domains": [p["domain"] for p in prompts], "doc_lengths": torch.tensor([p["length"] for p in prompts]),
                 "texts": [p["text"] for p in prompts],
                 "meta": {**base_meta(l), "hidden_file": npy[l].name, "hidden_shape": list(shape),
                          "hidden_dtype": "float16", "num_sentences": len(ids), "num_tokens": int(offsets[-1]),
                          "num_queries": int(q_off[-1]), "min_len": int(lengths.min()), "max_len": int(lengths.max()),
                          "mean_len": float(lengths.float().mean()),
                          "layer_names": ["input"] + [f"block {k}" for k in range(1, shape[0])]}}
        tmp = out_paths[l].with_name(out_paths[l].name + ".tmp")
        torch.save(feats, tmp)
        os.replace(tmp, out_paths[l])
        print(f"[extract] {out_paths[l]}: {len(ids)} prompts, {int(q_off[-1])} queries, hidden {tuple(shape)} "
              f"float16 -> {npy[l]}", flush=True)
    progress.unlink(missing_ok=True)


def main_multidoc(args, p):
    """--dataset multidoc: query positions only; ELF XSum: all trajectory points at once."""
    from tokid.models import ELFTrajectoryExtractor

    if args.no_eos or args.max_examples or args.out is not None:
        p.error("multidoc: the prompts are fixed (tokid.data.MultiDocSpec); no --no-eos / --max-examples / --out")
    if args.model == "elf":
        if args.elf_weights != "xsum":
            p.error("multidoc is for --elf-weights xsum")
        if args.variant is not None or any(getattr(args, k) != v for k, v in ELF_DEFAULTS.items() if k != "elf_weights"):
            p.error("multidoc with ELF writes all trajectory points (t = 0, 0.1 ... 0.9, decoder) in one run; "
                    "no --variant / --elf-* options")
    device = resolve_device(args.device)
    torch.set_grad_enabled(False)
    spec = MULTIDOC[args.dataset]
    if args.model == "t5":
        extractor, tokenizer, meta = load_t5(args.t5_name, args.cache_dir, args.local_files_only, device)
        labels, tags = ["t5"], {"t5": model_tag(args)}

        def run(ids, index):
            return {"t5": extractor(ids, index)}
    else:
        obs = ELFObservation(t=1.0, mode="decoder", noise_seed=args.elf_noise_seed)  # mode/t unused by run_points
        extractor, tokenizer, meta = load_elf(args.elf_source, obs, args.cache_dir, args.local_files_only,
                                              args.elf_checkpoint, device, weights=args.elf_weights)
        if not isinstance(extractor, ELFTrajectoryExtractor):
            raise TypeError("expected the trajectory extractor")
        labels = extractor.all_points()
        tags = {l: f"{ELF_HEADS[args.elf_weights]}_{l}" for l in labels}
        run = extractor.run_points
    out_paths = {l: args.out_dir / f"{tags[l]}_{args.dataset}.pt" for l in labels}
    done = [str(o) for o in out_paths.values() if o.exists()]
    if len(done) == len(out_paths) and not args.overwrite:  # partly done: resume the rest
        p.error(f"{', '.join(done)} exist (use --overwrite)")
    if args.overwrite:  # start from scratch: drop outputs and any checkpoint
        for o in out_paths.values():
            for f in (o, o.with_name(o.stem + ".hidden.npy"), o.with_name(o.stem + ".hidden.npy.partial"),
                      o.with_name(o.stem + ".progress.json")):
                f.unlink(missing_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_multidoc(args.dataset, tokenizer, args.cache_dir)
    embedding = extractor.encoder.get_input_embeddings().weight.detach().float().cpu()
    print(f"[extract] {meta['model']} on {args.dataset}: {len(prompts)} prompts, points {labels}, device={device}",
          flush=True)

    def base_meta(label):
        m = {**meta, "tag": tags[label], "dataset": args.dataset, "add_eos": True, "rows": "prompt",
             "domain_names": list(spec.sources), "ood_domains": list(spec.ood), "lengths": list(spec.lengths),
             "queries_per_doc": spec.queries_per_doc}
        if args.model == "elf":
            m["observation"] = {**m["observation"], "t": 1.0 if label == "decoder" else float(label[1:]),
                                "mode": "decoder" if label == "decoder" else "denoiser"}
        return m

    extract_queries(run, labels, prompts, out_paths, embedding, base_meta)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["t5", "elf"], required=True)
    p.add_argument("--dataset", choices=DATASET_NAMES, default="mrpc")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Default: all (MRPC, CoLA) / first 500 (MNLI), as in the paper; "
                        "first 500 prompts of <= 64 tokens (WMT14); docmix: 64 documents per domain.")
    p.add_argument("--pair-format", choices=["concat", "t5"], default="concat",
                   help="concat: 's1 s2' plain text (default). t5: T5's GLUE prompt, e.g. 'mrpc sentence1: ...'.")
    p.add_argument("--no-eos", action="store_true", help="Do not append T5's </s> token.")
    p.add_argument("--t5-name", default="google-t5/t5-small")
    p.add_argument("--elf-source", type=Path, default=DEFAULT_ELF_SOURCE)
    p.add_argument("--elf-checkpoint", type=Path, default=None, help="Local checkpoint file instead of the official one.")
    p.add_argument("--elf-weights", choices=sorted(ELF_WEIGHTS), default="owt",
                   help="owt: unconditional ELF-B (default). de-en: WMT14 German->English ELF-B; "
                        "probes the prompt rows, needs --dataset wmt14. xsum: XSum summarization ELF-B; "
                        "prompt rows on the real sampling trajectory, needs --dataset docmix.")
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
    if args.dataset in MULTIDOC:
        return main_multidoc(args, p)
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
        spec = ELF_WEIGHTS[args.elf_weights]
        if args.dataset not in ELF_DATASETS[args.elf_weights]:
            p.error(f"--elf-weights {args.elf_weights} runs on --dataset {' / '.join(sorted(ELF_DATASETS[args.elf_weights]))}")
        if spec.get("trajectory"):
            if args.elf_self_cond != "zero" or args.elf_guidance != 1.0 or args.elf_cfg_scale != 1.0:
                p.error(f"--elf-weights {args.elf_weights} follows the official sampler (CFG 2, self-cond. CFG 1, "
                        "self-cond. from the previous step); only --elf-t / --elf-mode can be set")
            if args.elf_mode == "denoiser" and args.elf_t == 1.0:
                p.error("the denoiser is never evaluated at t = 1 in sampling; use --elf-mode decoder")
        elif args.elf_guidance != 1.0 and not (spec["conditional"] and args.elf_self_cond == "pred"):
            p.error("--elf-guidance only applies to a conditional checkpoint with --elf-self-cond pred")

    if args.dataset in TRANSLATION and args.no_eos:
        p.error("--no-eos: the WMT14 prompts are tokenized with </s>, as in ELF's training data")
    if args.dataset in DOCMIX and (args.no_eos or args.max_examples):
        p.error("docmix prompts are fixed (64 per domain, 255 tokens + </s>); no --no-eos / --max-examples")

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
    targets = ids_list = domains = None
    if args.dataset in DOCMIX:
        prompts = load_docmix(args.dataset, tokenizer, args.cache_dir)
        texts, ids_list, domains = ([q[k] for q in prompts] for k in ("text", "ids", "domain"))
    elif args.dataset in TRANSLATION:
        texts, targets = load_translation(args.dataset, args.max_examples, args.cache_dir)
        if args.model == "t5":
            targets = None  # T5 encodes the prompt alone, as ELF's frozen encoder does for prompt rows
    else:
        texts = load_texts(args.dataset, args.max_examples, args.pair_format, args.cache_dir)
    print(f"[extract] {meta['model']} on {args.dataset}: {len(texts)} examples, device={device}", flush=True)
    feats = extract(extractor, tokenizer, texts, add_eos=not args.no_eos, targets=targets, ids_list=ids_list)
    n_tok, n_sent = feats["token_ids"].numel(), len(texts)
    lengths = feats["offsets"].diff()
    feats["texts"] = texts
    if domains is not None:  # one label per sentence (prompt)
        feats["domains"] = domains
    feats["meta"] = {**meta, "tag": model_tag(args), "dataset": args.dataset, "pair_format": args.pair_format,
                     "add_eos": not args.no_eos, "num_sentences": n_sent, "num_tokens": n_tok,
                     "min_len": int(lengths.min()), "max_len": int(lengths.max()),
                     "mean_len": float(lengths.float().mean()),
                     "domain_names": list(DOCMIX[args.dataset].domains) if domains is not None else None,
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
