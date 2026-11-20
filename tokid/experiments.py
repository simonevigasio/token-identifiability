"""Token identifiability experiments of Brunner et al. (ICLR 2020), Sec. 4 and App. B.

Experiments (``--experiment``):

identity       Sec. 4.2 / Fig. 2a, 2b, 7, 8-10. For every layer l, train g_l on
               (e^l_i, x_i) pairs of that layer, report the test identifiability rate,
               the train rate (Fig. 7), and the rate of g_l applied to every other
               layer (Fig. 2b, 8-10). Also the naive baseline g = identity.
hidden2hidden  App. B.4 / Fig. 11. Identify e^{l-1}_i from e^l_i among the
               sentence's layer-(l-1) states (pairs (e^1, x), (e^l, e^{l-1})).
neighbours     App. B.5 / Fig. 14-17. Recover x_{i+k}, k in {-3..-1, 1..3}, from e^l_i.

Targets (``--target``, identity/neighbours):

token   x_i = T5 input token embedding (default). The discrete input of T5 and ELF.
input   x_i = hidden[0], the rows entering the first transformer block (literal
        analogue of BERT's input embedding). Same as ``token`` for T5; for ELF it is
        the projected T5 latent in ELF's 768-d space.

Example:
    python -m tokid.experiments --features features/elf-b_mrpc.pt --experiment identity
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tokid.extract import resolve_device
from tokid.identify import nn_rate, vector_keys
from tokid.probes import make_probe, predict, train_probe
from tokid.splits import sentence_splits


LR_GRID = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6]  # App. B.1


@dataclass
class Task:
    """Queries are tokens with a valid target; candidates are all tokens."""
    layers: list            # layer index (into hidden) of each of the G inputs
    x: torch.Tensor         # (G, Q, d_in) input hidden states of the queries
    y: torch.Tensor         # (Gy, Q, d_out) regression targets, Gy in {1, G}
    query_sent: torch.Tensor  # (Q,)
    target_key: torch.Tensor  # (Gy, Q)
    cand: torch.Tensor      # (Gy, N, d_out)
    cand_sent: torch.Tensor  # (N,)
    cand_key: torch.Tensor  # (Gy, N)

    @property
    def naive_ok(self):
        return self.x.shape[-1] == self.y.shape[-1]

    def rate(self, pred, idx, metric):
        tk = self.target_key[:, idx]
        return nn_rate(pred, self.query_sent[idx], tk, self.cand, self.cand_sent, self.cand_key, metric)


def load_features(path, device):
    feats = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("token_ids", "sentence_ids", "positions", "offsets", "tokens", "hidden"):
        feats[k] = feats[k].to(device)
    return feats


def target_vectors(feats, target):
    if target == "token":
        return feats["tokens"]
    if target == "input":
        return feats["hidden"][0]
    raise ValueError(target)


def identity_task(feats, target):
    hidden, sent = feats["hidden"], feats["sentence_ids"]
    tgt = target_vectors(feats, target)
    keys = vector_keys(tgt)
    return Task(list(range(hidden.shape[0])), hidden, tgt[None], sent, keys[None], tgt[None], sent, keys[None])


def neighbour_task(feats, target, offset):
    hidden, sent, pos = feats["hidden"], feats["sentence_ids"], feats["positions"]
    lengths = feats["offsets"].diff()[sent]
    q = torch.nonzero((pos + offset >= 0) & (pos + offset < lengths), as_tuple=True)[0]
    tgt = target_vectors(feats, target)
    keys = vector_keys(tgt)
    return Task(list(range(hidden.shape[0])), hidden[:, q], tgt[None, q + offset], sent[q], keys[None, q + offset],
                tgt[None], sent, keys[None])


def hidden2hidden_task(feats):
    hidden, sent = feats["hidden"], feats["sentence_ids"]
    prev = hidden[:-1]
    keys = vector_keys(prev)
    return Task(list(range(1, hidden.shape[0])), hidden[1:], prev, sent, keys, prev, sent, keys)


def split_indices(task, split):
    out = {}
    for name, sents in split.items():
        chosen = torch.as_tensor(sents, device=task.query_sent.device, dtype=task.query_sent.dtype)
        out[name] = torch.nonzero(torch.isin(task.query_sent, chosen), as_tuple=True)[0]
    return out


def fit(task, idx, probe, loss, lr, args, seed):
    model = make_probe(probe, task.x.shape[0], task.x.shape[-1], task.y.shape[-1], args.mlp_hidden, seed=seed)
    tr, va = idx["train"], idx["val"]
    return train_probe(model, task.x[:, tr], task.y[:, tr], task.x[:, va], task.y[:, va], loss, lr=lr,
                       batch_size=args.batch_size, patience=args.patience, max_epochs=args.max_epochs,
                       seed=seed, log_every=args.log_every)


def lr_search(task, idx, probe, loss, args):
    """App. B.1 learning-rate search; chosen by mean validation identifiability (fold 0)."""
    scores = {}
    for lr in args.lr_grid:
        model, _ = fit(task, idx, probe, loss, lr, args, seed=args.seed)
        scores[lr] = task.rate(predict(model, task.x[:, idx["val"]]), idx["val"], loss).mean().item()
        print(f"    lr search {probe}/{loss}: lr={lr:g} val rate={scores[lr]:.4f}", flush=True)
    best = max(scores, key=scores.get)
    return best, {str(k): v for k, v in scores.items()}


class Checkpoint:
    """Partial results, saved atomically after every trained probe.

    A job killed by the scheduler (e.g. AI-LAB's 12 h limit) resumes from here; at most
    the probe that was being trained is lost. The run settings are stored too, and a
    checkpoint written with different settings is refused instead of silently mixed in.
    """

    def __init__(self, path, settings):
        self.path, self.settings, self.units = path, settings, {}
        if path is not None and path.exists():
            data = json.loads(path.read_text())
            if data["settings"] != settings:
                raise SystemExit(f"{path} was written with different settings:\n  {data['settings']}\n"
                                 f"now:\n  {settings}\nDelete it to start over.")
            self.units = data["units"]
            print(f"[resume] {len(self.units)} finished units loaded from {path}", flush=True)

    def get(self, key):
        return self.units.get(key)

    def put(self, key, value):
        self.units[key] = value
        if self.path is not None:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({"settings": self.settings, "units": self.units}))
            os.replace(tmp, self.path)


def run_task(task, folds, args, cross_layer, label="", ckpt=None, prefix=""):
    ckpt = ckpt or Checkpoint(None, None)
    res = {"layers": task.layers, "configs": {}, "lr": {}, "lr_search": {}}
    configs = [(p, l) for p in args.probes for l in args.losses]
    lrs = {c: args.lr for c in configs}
    if args.lr_search:
        idx0 = split_indices(task, folds[0])
        for c in configs:
            key = f"{prefix}lr_search/{c[0]}_{c[1]}"
            if ckpt.get(key) is None:
                ckpt.put(key, lr_search(task, idx0, *c, args))
            lrs[c], res["lr_search"][f"{c[0]}_{c[1]}"] = ckpt.get(key)
    res["lr"] = {f"{p}_{l}": lrs[(p, l)] for p, l in configs}

    def put(name, key, value):
        res["configs"].setdefault(name, {}).setdefault(key, []).append(value)

    for k, split in enumerate(folds):
        idx = split_indices(task, split)
        te = idx["test"]
        if task.naive_ok:  # g = identity
            key = f"{prefix}fold{k}/naive"
            if ckpt.get(key) is None:
                ckpt.put(key, {m: task.rate(task.x[:, te], te, m).tolist() for m in ("cos", "l2")})
            for metric, rates in ckpt.get(key).items():
                put(f"naive_{metric}", "test", rates)
        for ci, (probe, loss) in enumerate(configs):
            name = f"{probe}_{loss}"
            key = f"{prefix}fold{k}/{name}"
            rec = ckpt.get(key)
            if rec is None:
                t0 = time.time()
                seed = args.seed * 100_000 + k * 100 + ci
                model, info = fit(task, idx, probe, loss, lrs[(probe, loss)], args, seed)
                rec = {
                    "test": task.rate(predict(model, task.x[:, te]), te, loss).tolist(),
                    "train": task.rate(predict(model, task.x[:, idx["train"]]), idx["train"], loss).tolist(),
                    "val": task.rate(predict(model, task.x[:, idx["val"]]), idx["val"], loss).tolist(),
                    "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                }
                if cross_layer:  # Fig. 2b: g trained on layer a, tested on every layer b
                    with torch.no_grad():
                        rec["cross"] = [task.rate(model.single(a, task.x[:, te]), te, loss).tolist()
                                        for a in range(task.x.shape[0])]
                del model
                ckpt.put(key, rec)
                print(f"  {label}fold {k + 1}/{len(folds)} {name:10s} test rate by layer: "
                      + " ".join(f"{v:.3f}" for v in rec["test"])
                      + f"  [{info['epochs_run']} ep, {time.time() - t0:.0f}s]", flush=True)
            for field, value in rec.items():
                put(name, field, value)
    for cfg in res["configs"].values():
        for key in ("test", "train", "val"):
            if key in cfg:
                arr = np.asarray(cfg[key])
                cfg[f"{key}_mean"], cfg[f"{key}_std"] = arr.mean(0).tolist(), arr.std(0).tolist()
        if "cross" in cfg:
            cfg["cross_mean"] = np.asarray(cfg["cross"]).mean(0).tolist()
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--experiment", choices=["identity", "hidden2hidden", "neighbours"], required=True)
    p.add_argument("--target", choices=["token", "input"], default="token")
    p.add_argument("--probes", nargs="+", choices=["linear", "mlp"], default=["linear", "mlp"])
    p.add_argument("--losses", nargs="+", choices=["cos", "l2"], default=["cos", "l2"])
    p.add_argument("--offsets", nargs="+", type=int, default=[-1, -2, -3, 1, 2, 3])
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-search", action="store_true", help="Search LR_GRID on fold 0 (App. B.1).")
    p.add_argument("--lr-grid", type=float, nargs="+", default=LR_GRID)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--mlp-hidden", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=0, help="Print validation loss every N epochs (0: off).")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    suffix = "" if args.experiment == "hidden2hidden" else f"_{args.target}"
    out = args.out or Path("results") / args.features.stem / f"{args.experiment}{suffix}.json"
    if out.exists() and not args.overwrite:
        p.error(f"{out} exists (use --overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    feats = load_features(args.features, device)
    meta = feats["meta"]
    folds = sentence_splits(meta["num_sentences"], args.folds, seed=args.seed)
    print(f"[{args.experiment}] {meta['tag']} / {meta['dataset']}: {meta['num_tokens']} tokens, "
          f"{meta['num_sentences']} sentences, hidden {tuple(feats['hidden'].shape)}, target={args.target}, "
          f"device={device}", flush=True)
    result = {"meta": meta, "experiment": args.experiment,
              "target": None if args.experiment == "hidden2hidden" else args.target,
              "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "folds": [{k: v.tolist() for k, v in f.items()} for f in folds]}
    settings = {k: v for k, v in result["args"].items() if k not in ("device", "log_every", "out", "overwrite")}
    partial = out.with_name(out.name + ".partial")
    if args.overwrite and partial.exists():
        partial.unlink()
    ckpt = Checkpoint(partial, settings)
    t0 = time.time()
    if args.experiment == "identity":
        result.update(run_task(identity_task(feats, args.target), folds, args, cross_layer=True, ckpt=ckpt))
    elif args.experiment == "hidden2hidden":
        result.update(run_task(hidden2hidden_task(feats), folds, args, cross_layer=False, ckpt=ckpt))
    else:
        result["offsets"] = {}
        for k in args.offsets:
            if k == 0:
                continue
            task = neighbour_task(feats, args.target, k)
            result["offsets"][str(k)] = run_task(task, folds, args, cross_layer=False, label=f"k={k:+d} ",
                                                 ckpt=ckpt, prefix=f"k{k}/")
            result["offsets"][str(k)]["num_queries"] = task.x.shape[1]
            del task
    result["elapsed_s"] = time.time() - t0  # of this (possibly resumed) session
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, out)
    partial.unlink(missing_ok=True)
    print(f"[{args.experiment}] done in {result['elapsed_s'] / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
