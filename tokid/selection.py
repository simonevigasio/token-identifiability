"""Model selection for the probes g_l, with an untouched test set (multidoc features).

Question: how well can *any* probe g_l recover the input token from layer l? The best
probe found gives a lower bound on token identifiability. Searching over many probes
and then reporting the best *test* score would overstate that bound, so the protocol
keeps every test document out of every decision:

1. Splits, fixed before any training (``protocol_splits``), by document:
   * test_ood: every document of the out-of-distribution sources (MultiDocSpec.ood),
   * test_id: 20% of the documents of every other source, stratified by prompt length,
   * dev: the remaining 80%; all training and all choices use only these.
2. Search (``successive_halving``) on one dev train/val split: ``--n-random`` random
   configurations plus the paper's four probes. Each round trains the survivors for a
   larger epoch budget and keeps the best 1/eta (a configuration survives if it ranks
   high for at least one layer). Cheap first rounds weed out bad configurations.
3. Selection (``cross_validate``): the finalists are trained on ``--cv-folds`` dev folds
   with early stopping on the validation identifiability rate. Each layer picks the
   configuration with the best mean validation rate (per-layer selection).
4. Final test, done once: each chosen configuration is retrained on all dev documents
   for the median best epoch of its CV runs (no validation set needed), with
   ``--seeds`` seeds, and evaluated on test_id and test_ood. Rates come with 95%
   bootstrap confidence intervals over test documents, and are broken down by source,
   by prompt length, and by whether the token type occurred as a training target.
   Besides the paper's test (nearest among the prompt's own distinct tokens), every
   test is also scored against the whole vocabulary (every row of the embedding
   matrix): true token nearest (vocab_top1) or among the 5 nearest (vocab_top5).
   ``--retest`` adds these to a finished run without redoing the search.

The paper's probes (linear / MLP-1000, cosine / L2, lr 1e-4, early stopping on the
validation loss with patience 20) are trained on the same dev split and tested on the
same documents, as a reference.

Shared probes (``--share``, several ``--features`` files of the same prompts, e.g. the 7
points of the ELF trajectory t0 ... decoder). A cell is one (file, layer) pair:
  none   one probe per cell (the protocol above, run per file)
  time   one probe per layer, trained on that layer of every file: time-independent
  layer  one probe per file, trained on every layer of it: layer-independent
  both   one probe for every cell: time- and layer-independent
Everything above stays the same with "probe" in place of "layer": a probe's training set
is every (query, cell) pair of its cells (an epoch passes over all of them), early
stopping and CV selection use its validation rate averaged over its cells, halving
promotes a configuration that ranks high for some probe, and input standardization
uses the probe's pooled training data. Test rates are reported per cell. The splits
are the per-file ones (same prompts, same seed), so the rates compare directly with
the per-file runs.

Search space (``sample_config``):
  arch          linear (no bias, paper) | affine (bias) | mlp (1-2 GELU layers, biases)
  width         512 | 1024 | 2048 (mlp)          dropout     0 | 0.1 | 0.2 (mlp)
  standardize   input z-scored per layer and dimension (train statistics) or not
  loss          cos | l2 | infonce (softmax over the prompt's own tokens, temperature tau)
  lr            log-uniform [1e-5, 3e-3]          weight decay 0 | 1e-3 | 1e-2 | 1e-1 (AdamW)
  batch size    256 | 1024
The nearest-neighbour metric of a probe is cosine for cos / infonce and L2 for l2.

Objective (``--objective``):
  prompt  (default, the paper's task) a probe is right if the nearest of its prompt's own
          distinct tokens is the true one; early stopping, halving and CV selection use
          this validation rate, and the contrastive loss is a softmax over those tokens.
  vocab   the task is the whole vocabulary: early stopping, halving and CV selection use
          the vocabulary-wide validation top-1 (true token nearest among every row of the
          embedding matrix), and the contrastive loss becomes ``infonce_vocab``, a softmax
          over every embedding row (frozen T5 embeddings as the classes: no learned
          classifier or bias, so token types never seen in training stay meaningful). The
          same random configurations are drawn (infonce -> infonce_vocab); cos / l2 do not
          depend on candidates. Paper probes keep the paper procedure. Results go to
          ``results/<run>_vocab/``. The test reports both rates, as always.

Weights: every refit of a selected configuration saves the weights of the probes that
selected it (``<out dir>/probes/<config>_seed<s>.pt``, listed under "weights" in
selection.json); ``load_probes`` / ``load_selected`` rebuild them.

Example:
    python -m tokid.selection --features features/elf-b-xsum_t0.3_multidoc.pt
"""

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tokid.extract import resolve_device
from tokid.probes import BatchedProbe, info_nce_loss, info_nce_vocab_loss, per_layer_loss

PAPER_CONFIGS = {
    f"paper_{arch}_{loss}": {"arch": arch, "depth": 1 if arch == "mlp" else 0, "width": 1000 if arch == "mlp" else 0,
                             "dropout": 0.0, "standardize": False, "loss": loss, "tau": None, "lr": 1e-4,
                             "weight_decay": 0.0, "batch_size": 256}
    for arch in ("linear", "mlp") for loss in ("cos", "l2")
}


# ------------------------------------------------------------------------------ data

def load_query_features(path, device):
    """multidoc features (tokid.extract.extract_queries) -> dict with ``hidden`` on device."""
    feats = torch.load(path, map_location="cpu", weights_only=False)
    if feats.get("format") != "queries":
        raise ValueError(f"{path}: not a query-position feature file (extract with --dataset multidoc)")
    mm = np.load(Path(path).with_name(feats["meta"]["hidden_file"]), mmap_mode="r")
    hidden = torch.empty(mm.shape, dtype=torch.float16, device=device)
    for l in range(mm.shape[0]):  # layer by layer: never two full copies in RAM
        hidden[l] = torch.from_numpy(np.array(mm[l])).to(device)
    feats["hidden"] = hidden
    return feats


# Probe sharing. A "cell" is one (variant, layer) pair: variant = one feature file, e.g.
# one point of the ELF sampling trajectory (t0, t0.1, ..., decoder). Cell index:
# variant * layers + layer. Every probe reads the cells of one row of ``probe_cells``.
SHARE_MODES = ("none", "time", "layer", "both")
VOCAB_KS = (1, 5)   # vocabulary-wide identifiability: true token among the k nearest of all tokens


def probe_cells(share: str, variants: int, layers: int) -> torch.Tensor:
    """(P, K) cell indices read by each of the P probes.

    none:  one probe per (variant, layer), P = V*L, K = 1 (the per-layer protocol);
    time:  one probe per layer, shared by every variant (time-independent), P = L, K = V;
    layer: one probe per variant, shared by every layer (layer-independent), P = V, K = L;
    both:  a single probe for every cell, P = 1, K = V*L."""
    grid = torch.arange(variants * layers).view(variants, layers)
    if share == "none":
        return grid.reshape(-1, 1)
    if share == "time":
        return grid.T.contiguous()
    if share == "layer":
        return grid.clone()
    if share == "both":
        return grid.reshape(1, -1)
    raise ValueError(f"unknown share mode {share!r}")


class CellStore:
    """Hidden states of C cells, each (Q, d) float16, held in one or more shards (one per
    device: the states of all 7 ELF trajectory points do not fit on one 24 GB GPU).

    ``gather(c, q)``: (P, B, d) float16 on the main device; row (p, b) is the state of
    query q[b] in cell c[p, b]."""

    def __init__(self, shards, main):
        """shards: list of (tensor (n_s, Q, d), the n_s cell ids it holds)."""
        self.main = torch.device(main)
        self.shards = [x for x, _ in shards]
        C = sum(len(ids) for _, ids in shards)
        shard_of, local = torch.full((C,), -1, dtype=torch.long), torch.zeros(C, dtype=torch.long)
        for s, (_, ids) in enumerate(shards):
            ids = torch.as_tensor(ids, dtype=torch.long)
            shard_of[ids] = s
            local[ids] = torch.arange(ids.numel())
        if (shard_of < 0).any():
            raise ValueError("every cell must be in exactly one shard")
        self.shard_of, self.local = shard_of.to(self.main), local.to(self.main)
        self.shape = (C, *self.shards[0].shape[1:])

    def gather(self, c, q):
        if len(self.shards) == 1:
            x = self.shards[0]
            return x[self.local[c].to(x.device), q.to(x.device)].to(self.main)
        out = torch.empty(*c.shape, self.shape[2], dtype=self.shards[0].dtype, device=self.main)
        which, qq = self.shard_of[c], q.expand_as(c)
        for s, x in enumerate(self.shards):
            m = which == s
            if bool(m.any()):
                out[m] = x[self.local[c[m]].to(x.device), qq[m].to(x.device)].to(self.main)
        return out


def _same_prompts(a, b):
    keys = ("token_ids", "offsets", "query_index", "query_doc", "query_offsets", "doc_lengths", "embedding")
    return a["domains"] == b["domains"] and all(torch.equal(a[k], b[k]) for k in keys)


def load_cells(paths, device, reserve_gb: float = 6.0):
    """Several multidoc feature files of the same prompts and queries (e.g. the 7 points
    of the ELF trajectory) -> the features of the first, with ``hidden`` a CellStore of
    all V*L cells (cell = variant * L + layer) and ``variant_meta`` the V metas.

    On CUDA the cells fill the other visible GPUs first and the main GPU, where the
    probes train, only with what is left, keeping ``reserve_gb`` free there (training
    peaks at a few GiB, e.g. the standardization pass). Cells are loaded one at a time:
    RAM never holds more than one (Q, d) array."""
    device = torch.device(device)
    feats = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    for p, f in zip(paths, feats):
        if f.get("format") != "queries":
            raise ValueError(f"{p}: not a query-position feature file (extract with --dataset multidoc)")
        if not _same_prompts(feats[0], f):
            raise ValueError(f"{p}: different prompts or queries than {paths[0]}")
    mms = [np.load(Path(p).with_name(f["meta"]["hidden_file"]), mmap_mode="r") for p, f in zip(paths, feats)]
    if len({m.shape for m in mms}) != 1:
        raise ValueError(f"hidden shapes differ: {[m.shape for m in mms]}")
    L, Q, d = mms[0].shape
    C, cell_bytes = len(mms) * L, Q * d * 2
    if device.type == "cuda":
        main = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count()) if i != main.index] + [main]
        caps = []
        for dv in devices:
            free, _ = torch.cuda.mem_get_info(dv)
            keep = (reserve_gb if dv == main else 0.5) * 2**30
            caps.append(max(0, int((free - keep) // cell_bytes)))
        if sum(caps) < C:
            raise SystemExit(f"the hidden states need {C * cell_bytes / 2**30:.1f} GiB ({C} cells); the "
                             f"{len(devices)} visible GPU(s) "
                             f"({', '.join(torch.cuda.get_device_name(dv) for dv in devices)}) hold {sum(caps)} "
                             f"cells. Request more GPUs (sbatch --gres=gpu:{len(devices) + 1}), GPUs with more "
                             f"memory (e.g. exclude the T4 nodes), or lower --reserve-gb.")
    else:
        main, devices, caps = device, [device], [C]
    shards, nxt = [], 0
    for dv, cap in zip(devices, caps):
        ids = list(range(nxt, min(C, nxt + cap)))
        if not ids:
            continue
        x = torch.empty(len(ids), Q, d, dtype=torch.float16, device=dv)
        for j, c in enumerate(ids):
            x[j] = torch.from_numpy(np.array(mms[c // L][c % L])).to(dv)
        shards.append((x, ids))
        nxt += len(ids)
    out = feats[0]
    out["hidden"] = CellStore(shards, main)
    out["variant_meta"] = [f["meta"] for f in feats]
    out["shard_cells"] = [(str(x.device), len(ids)) for x, ids in shards]
    return out


class QueryData:
    """Queries (stored positions) and candidates (every distinct token of the prompt).

    A query is identified if the nearest candidate of its own prompt is its token. Two
    occurrences of a token have the same T5 embedding, so candidates are the distinct
    token ids of the prompt: retrieving either occurrence counts as correct, as in
    tokid.identify.

    ``cells`` (P, K): probe p reads cells[p] (see probe_cells). Its training set is every
    (query, cell) pair of its K cells. Default: one probe per cell (P = C, K = 1)."""

    def __init__(self, feats, device, cells=None):
        self.device = device
        x = feats["hidden"]                                              # (C, Q, d) float16
        self.x = x if isinstance(x, CellStore) else CellStore([(x, list(range(x.shape[0])))], device)
        self.num_cells = self.x.shape[0]
        cells = torch.arange(self.num_cells)[:, None] if cells is None else torch.as_tensor(cells, dtype=torch.long)
        if not torch.equal(cells.flatten().sort().values, torch.arange(self.num_cells)):
            raise ValueError("every cell must be read by exactly one probe")
        self.cells = cells.to(device)
        self.probes, self.share_k = cells.shape
        probe_of = torch.empty(self.num_cells, dtype=torch.long)
        probe_of[cells.flatten()] = torch.arange(self.probes).repeat_interleave(self.share_k)
        self.probe_of = probe_of.tolist()
        tok = feats["token_ids"]
        self.q_tok = tok[feats["query_index"]].to(device)
        self.q_doc = feats["query_doc"].to(device)
        self.q_off = feats["query_offsets"]
        self.E = feats["embedding"].to(device)
        self.En = F.normalize(self.E, dim=-1)
        self.domains = list(feats["domains"])
        self.doc_lengths = feats["doc_lengths"].tolist()
        self.num_docs = len(self.domains)
        off = feats["offsets"]
        uniq = [torch.unique(tok[off[i]:off[i + 1]]) for i in range(self.num_docs)]
        counts = torch.tensor([u.numel() for u in uniq])
        self.c_off = torch.zeros(self.num_docs + 1, dtype=torch.long)
        self.c_off[1:] = counts.cumsum(0)
        self.c_tok = torch.cat(uniq).to(device)
        self.c_doc = torch.arange(self.num_docs).repeat_interleave(counts).to(device)
        U = int(counts.max())
        self.cand_pad = torch.zeros(self.num_docs, U, dtype=torch.long)
        self.cand_mask = torch.zeros(self.num_docs, U, dtype=torch.bool)
        slots = []
        for i, u in enumerate(uniq):
            self.cand_pad[i, :u.numel()] = u
            self.cand_mask[i, :u.numel()] = True
            q = tok[feats["query_index"][self.q_off[i]:self.q_off[i + 1]]]
            slots.append(torch.searchsorted(u, q))
        self.cand_pad, self.cand_mask = self.cand_pad.to(device), self.cand_mask.to(device)
        self.q_slot = torch.cat(slots).to(device)

    def queries(self, docs) -> torch.Tensor:
        docs = np.asarray(docs, dtype=np.int64)
        return torch.cat([torch.arange(int(self.q_off[d]), int(self.q_off[d + 1])) for d in docs]).to(self.device)

    def inputs(self, k, idx):
        """(P, B, d) float32: for each probe p, query idx[b] in its cell cells[p, k[b]]
        (``k`` a (B,) tensor, or an int: the same slot for every query)."""
        if torch.is_tensor(k):
            c = self.cells[:, k]
        else:
            c = self.cells[:, k:k + 1].expand(-1, idx.numel())
        return self.x.gather(c, idx).float()

    def per_probe(self, v):
        """Per-cell values (C, ...) -> per-probe means over the probe's cells (P, ...)."""
        return v[self.cells].mean(1)

    def targets(self, idx):
        return self.E[self.q_tok[idx]][None]

    def loss(self, pred, idx, cfg):
        if cfg["loss"] == "infonce":
            d = self.q_doc[idx]
            return info_nce_loss(pred, self.En[self.cand_pad[d]], self.cand_mask[d], self.q_slot[idx], cfg["tau"])
        if cfg["loss"] == "infonce_vocab":
            return info_nce_vocab_loss(pred, self.En, self.q_tok[idx], cfg["tau"])
        return per_layer_loss(pred, self.targets(idx).expand(pred.shape[0], -1, -1), cfg["loss"])

    @torch.no_grad()
    def correct(self, model, docs, metric, chunk_docs: int = 16):
        """Bool (C, queries of ``docs`` in order), per cell: nearest same-prompt candidate
        is right (cell c evaluated with the probe that reads it)."""
        out = []
        docs = np.asarray(docs, dtype=np.int64)
        for i in range(0, docs.size, chunk_docs):
            chunk = docs[i:i + chunk_docs]
            idx = self.queries(chunk)
            c = torch.cat([torch.arange(int(self.c_off[d]), int(self.c_off[d + 1])) for d in chunk]).to(self.device)
            cand = self.E[self.c_tok[c]]
            other_doc = self.q_doc[idx][:, None] != self.c_doc[c][None, :]
            ok = torch.empty(self.num_cells, idx.numel(), dtype=torch.bool, device=self.device)
            for k in range(self.share_k):
                pred = model(self.inputs(k, idx))                        # (P, n, d_tok)
                if metric == "cos":
                    score = F.normalize(pred, dim=-1) @ F.normalize(cand, dim=-1).T
                elif metric == "l2":
                    score = -torch.cdist(pred, cand[None].expand(pred.shape[0], -1, -1).contiguous(),
                                         compute_mode="donot_use_mm_for_euclid_dist")
                else:
                    raise ValueError(metric)
                score = score.masked_fill(other_doc, float("-inf"))
                ok[self.cells[:, k]] = self.c_tok[c][score.argmax(-1)] == self.q_tok[idx]
            out.append(ok)
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def vocab_correct(self, model, docs, metric, ks=VOCAB_KS, chunk=None):
        """Vocabulary-wide identifiability: dict k -> bool (C, queries of ``docs`` in order),
        per cell: the true token is among the k nearest rows of the whole embedding matrix
        (every token of the vocabulary, not only the prompt's). Same probe outputs and
        metric as ``correct``; the scores are computed in full float32 (no TF32), since
        neighbours among ~32k rows can be close."""
        idx = self.queries(docs)
        V = self.E.shape[0]
        chunk = chunk or max(64, (1 << 28) // (self.probes * V))   # score block of ~1 GiB
        E, En, e2 = self.E, self.En, (self.E * self.E).sum(-1)
        out = {k: torch.empty(self.num_cells, idx.numel(), dtype=torch.bool, device=self.device) for k in ks}
        kmax = max(ks)
        for i in range(0, idx.numel(), chunk):
            b = idx[i:i + chunk]
            true = self.q_tok[b][None, :, None]
            for k in range(self.share_k):
                pred = model(self.inputs(k, b))                          # (P, n, d_tok)
                tf32 = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = False
                try:
                    if metric == "cos":
                        score = F.normalize(pred, dim=-1) @ En.T
                    elif metric == "l2":
                        score = 2 * pred @ E.T - e2                      # -|p - e|^2 + |p|^2: same order
                    else:
                        raise ValueError(metric)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = tf32
                hit = score.topk(kmax, dim=-1).indices == true           # (P, n, kmax)
                for kk in ks:
                    out[kk][self.cells[:, k], i:i + b.numel()] = hit[..., :kk].any(-1)
        return out

    @torch.no_grad()
    def mean_loss(self, model, idx, cfg, chunk: int = 4096):
        """(P,) mean loss of each probe over all (query, cell) pairs of its cells."""
        total = torch.zeros(self.probes, device=self.device)
        for k in range(self.share_k):
            for i in range(0, idx.numel(), chunk):
                b = idx[i:i + chunk]
                total += self.loss(model(self.inputs(k, b)), b, cfg) * b.numel()
        return total / (self.share_k * idx.numel())


class Identity(torch.nn.Module):
    def forward(self, x):
        return x.float()


# ---------------------------------------------------------------------------- splits

def _stratified_order(docs, lengths, rng):
    """Documents sorted by prompt length, random within a length, so that taking every
    k-th one is stratified by length."""
    docs = np.asarray(docs)
    return docs[np.lexsort((rng.random(docs.size), np.asarray(lengths)[docs]))]


def protocol_splits(domains, lengths, ood, test_every: int = 5, val_every: int = 5, cv_folds: int = 4, seed: int = 0):
    """Document splits of the protocol, stratified by source and prompt length.

    test_id = every ``test_every``-th document (1/5 = 20%) of each in-distribution source,
    search val = every ``val_every``-th dev document, cv = ``cv_folds`` folds of dev."""
    rng = np.random.default_rng([seed, 7])
    domains = np.asarray(domains)
    test_ood = np.nonzero(np.isin(domains, list(ood)))[0]
    test_id, dev, val, fold_of = [], [], [], {}
    for name in dict.fromkeys(domains.tolist()):
        if name in ood:
            continue
        order = _stratified_order(np.nonzero(domains == name)[0], lengths, rng)
        off = rng.integers(test_every)
        is_test = (np.arange(order.size) + off) % test_every == 0
        test_id += order[is_test].tolist()
        rest = order[~is_test]  # still sorted by length
        dev += rest.tolist()
        off_v, off_f = rng.integers(val_every), rng.integers(cv_folds)
        val += rest[(np.arange(rest.size) + off_v) % val_every == 0].tolist()
        fold_of.update(zip(rest.tolist(), ((np.arange(rest.size) + off_f) % cv_folds).tolist()))
    dev = np.sort(dev)
    val = np.sort(val)
    folds = np.array([fold_of[d] for d in dev])
    return {
        "test_ood": test_ood.tolist(), "test_id": sorted(test_id), "dev": dev.tolist(),
        "search": {"train": np.setdiff1d(dev, val).tolist(), "val": val.tolist()},
        "cv": [{"train": dev[folds != k].tolist(), "val": dev[folds == k].tolist()} for k in range(cv_folds)],
    }


# ----------------------------------------------------------------------- configurations

OBJECTIVES = ("prompt", "vocab")


def sample_config(rng, objective: str = "prompt") -> dict:
    """A random configuration. With ``objective="vocab"`` the same random draws give the
    same configuration, with the contrastive loss over the whole vocabulary."""
    arch = str(rng.choice(["linear", "affine", "mlp"], p=[0.2, 0.2, 0.6]))
    mlp = arch == "mlp"
    loss = str(rng.choice(["cos", "l2", "infonce"]))
    cfg = {
        "arch": arch, "depth": int(rng.choice([1, 2])) if mlp else 0,
        "width": int(rng.choice([512, 1024, 2048])) if mlp else 0,
        "dropout": float(rng.choice([0.0, 0.1, 0.2])) if mlp else 0.0,
        "standardize": bool(rng.random() < 0.5), "loss": loss,
        "tau": float(math.exp(rng.uniform(math.log(0.02), math.log(0.2)))) if loss == "infonce" else None,
        "lr": float(math.exp(rng.uniform(math.log(1e-5), math.log(3e-3)))),
        "weight_decay": float(rng.choice([0.0, 1e-3, 1e-2, 1e-1])),
        "batch_size": int(rng.choice([256, 1024])),
    }
    if loss == "infonce" and objective == "vocab":   # after every draw: the random stream is unchanged
        cfg["loss"] = "infonce_vocab"
    return cfg


def metric_of(cfg):
    return "l2" if cfg["loss"] == "l2" else "cos"


def build_probe(cfg, data, train_idx, seed):
    """The P probes of a configuration. Input standardization uses each probe's own
    training data: all (query, cell) pairs of its cells (a shared probe gets pooled
    statistics, so no cell-specific scaling enters it)."""
    mean = std = None
    if cfg["standardize"]:
        s1 = torch.zeros(data.probes, data.x.shape[-1], dtype=torch.float64, device=data.device)
        s2 = torch.zeros_like(s1)
        for k in range(data.share_k):
            for i in range(0, train_idx.numel(), 16384):
                x = data.inputs(k, train_idx[i:i + 16384]).double()
                s1 += x.sum(1)
                s2 += (x * x).sum(1)
        n = data.share_k * train_idx.numel()
        mean = s1 / n
        std = (s2 / n - mean * mean).clamp_min(0).sqrt()
    gen = torch.Generator().manual_seed(seed)
    return BatchedProbe(data.probes, data.x.shape[-1], data.E.shape[-1], hidden=(cfg["width"],) * cfg["depth"],
                        bias=cfg["arch"] != "linear", dropout=cfg["dropout"], mean=mean, std=std,
                        generator=gen).to(data.device)


class UnitResume:
    """Training state of the unit in progress, saved every ``every_s`` seconds.

    The run checkpoint (``Checkpoint``) saves only between units, so a unit longer than
    one Slurm job (e.g. a shared probe on AI-LAB's 12 h limit) would restart from
    scratch on every requeue and never finish. This file holds the weights, optimizer,
    early-stopping state and random generators of the current unit, after a whole
    epoch, so the unit resumes where it stopped and ends exactly as without the break.
    A state is used only by the same unit of a run with the same settings."""

    def __init__(self, path, settings, every_s=900.0):
        self.path, self.settings, self.every_s = path, settings, every_s

    def load(self, key):
        if self.path is None or not self.path.exists():
            return None
        state = torch.load(self.path, map_location="cpu", weights_only=False)
        if state.get("key") != key or state.get("settings") != self.settings:
            return None
        return state

    def save(self, key, state):
        if self.path is None:
            return
        tmp = self.path.with_name(self.path.name + ".tmp")
        torch.save({"key": key, "settings": self.settings, **state}, tmp)
        os.replace(tmp, self.path)

    def clear(self):
        if self.path is not None:
            self.path.unlink(missing_ok=True)


class ProbeWeights:
    """Saves the trained weights of the selected probes.

    After the refit of configuration c with seed s, the probes that selected c (probe p
    with chosen[p] == c) are written to ``<root>/<c>_seed<s>.pt``: their slices of the
    model's parameters and buffers (leading probe dimension), the configuration, the
    cells each probe reads and their names, the refit epochs and the training seed. The
    file is written before the unit's checkpoint entry, so a finished unit always has its
    weights (unless it finished before weight saving existed: then a warning)."""

    def __init__(self, root, cell_names, objective):
        self.root, self.cell_names, self.objective = Path(root), list(cell_names), objective

    def path(self, c, s):
        return self.root / f"{c}_seed{s}.pt"

    def save(self, c, s, cfg, model, probes, data, epochs, seed):
        self.root.mkdir(parents=True, exist_ok=True)
        idx = torch.as_tensor(probes, dtype=torch.long)
        state = {k: v.detach()[idx.to(v.device)].cpu().clone() for k, v in model.state_dict().items()}
        cells = data.cells[idx.to(data.cells.device)].tolist()
        blob = {"format": "tokid-probes", "config_name": c, "config": dict(cfg), "objective": self.objective,
                "metric": metric_of(cfg), "seed_index": s, "train_seed": seed, "probes": list(probes),
                "cells": cells, "cell_names": [[self.cell_names[x] for x in row] for row in cells],
                "epochs": [int(epochs[p]) for p in probes], "d_in": int(data.x.shape[-1]),
                "d_out": int(data.E.shape[-1]), "state": state}
        tmp = self.path(c, s).with_name(self.path(c, s).name + ".tmp")
        torch.save(blob, tmp)
        os.replace(tmp, self.path(c, s))


def load_probes(path, device="cpu"):
    """A file written by ``ProbeWeights`` -> one dict per saved probe: ``model`` (a
    BatchedProbe with a single probe: input (1, B, d_in) or (B, d_in) after [None]),
    ``probe``, ``cells``, ``cell_names``, ``config``, ``metric`` (cos / l2 nearest
    neighbours), ``epochs``. Example::

        for e in load_probes("results/elf-b-xsum_t0.3_multidoc_vocab/probes/r07_seed0.pt"):
            pred = e["model"](hidden[None])[0]          # hidden: (B, 768) states of e["cells"]
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != "tokid-probes":
        raise ValueError(f"{path}: not a probe weight file")
    cfg, st = blob["config"], blob["state"]
    out = []
    for j, p in enumerate(blob["probes"]):
        one = {k: v[j:j + 1] for k, v in st.items()}
        std = torch.ones(1, blob["d_in"]) if "mean" in one else None
        model = BatchedProbe(1, blob["d_in"], blob["d_out"], hidden=(cfg["width"],) * cfg["depth"],
                             bias=cfg["arch"] != "linear", dropout=cfg["dropout"],
                             mean=torch.zeros(1, blob["d_in"]) if std is not None else None, std=std)
        model.load_state_dict(one)
        out.append({"model": model.to(device).eval(), "probe": p, "cells": blob["cells"][j],
                    "cell_names": blob["cell_names"][j], "config_name": blob["config_name"], "config": cfg,
                    "metric": blob["metric"], "epochs": blob["epochs"][j], "objective": blob["objective"],
                    "seed_index": blob["seed_index"]})
    return out


def load_selected(results_dir, seed=0, device="cpu"):
    """The selected probes of a finished run (``results/<run>/``), refit seed ``seed``:
    a list indexed by probe (per-layer runs: probe = layer; share-time: probe = layer,
    reading that layer at every point), each entry as in ``load_probes``."""
    results_dir = Path(results_dir)
    res = json.loads((results_dir / "selection.json").read_text())
    if "weights" not in res:
        raise ValueError(f"{results_dir}: no saved weights (run made before weight saving existed)")
    entries = {}
    for c, files in res["weights"]["files"].items():
        if seed < len(files) and files[seed] is not None:
            for e in load_probes(results_dir / files[seed], device):
                entries[e["probe"]] = e
    missing = [p for p in range(len(res["chosen"])) if p not in entries]
    if missing:
        raise ValueError(f"{results_dir}: no weights for probes {missing} (seed {seed})")
    return [entries[p] for p in range(len(res["chosen"]))]


def _rng_state(dev):
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None}


def _set_rng_state(state, dev):
    torch.set_rng_state(state["cpu"])
    if dev.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], dev)


def train(cfg, data, train_docs, val_docs=None, seed=0, max_epochs=200, patience=10, stop_on="rate",
          epochs_per_layer=None, log="", resume=None, key=None):
    """Train the P probes of one configuration (one per layer in the per-layer protocol).

    An epoch is one pass over the probe's training set: every (training query, cell)
    pair of its K cells, in a random order shared by all probes. With ``val_docs``:
    per-probe early stopping on the validation identifiability rate, averaged over the
    probe's cells (``stop_on="rate"``: among the prompt's tokens; ``"vocab"``: top-1 over
    the whole vocabulary), or loss (``"loss"``, the paper); each probe keeps
    its best epoch's weights. With ``epochs_per_layer`` (refit, no validation): probe p
    keeps its weights after epochs_per_layer[p] epochs. With ``resume`` (a UnitResume)
    and ``key``: the unit's state is saved periodically and a saved state is resumed."""
    G, dev = data.probes, data.device
    torch.manual_seed(seed)  # dropout draws from the global generator: every unit reproducible on its own
    tr = data.queries(train_docs)
    n_tr = tr.numel()
    model = build_probe(cfg, data, tr, seed)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    va = data.queries(val_docs) if val_docs is not None else None
    best = torch.full((G,), -float("inf"), device=dev)
    best_epoch = torch.zeros(G, dtype=torch.long, device=dev)
    snap = [p.detach().clone() for p in params]
    active = torch.ones(G, dtype=torch.bool, device=dev)
    if epochs_per_layer is not None:
        target = torch.as_tensor(epochs_per_layer, device=dev)
        max_epochs = int(target.max())
    bs, t0, epoch, start, done_s = cfg["batch_size"], time.time(), -1, 0, 0.0
    state = resume.load(key) if resume is not None else None
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        for s, v in zip(snap, state["snap"]):
            s.copy_(v)
        best, best_epoch, active = (state[k].to(dev) for k in ("best", "best_epoch", "active"))
        gen.set_state(state["gen"])
        _set_rng_state(state["rng"], dev)
        epoch = start = state["epoch"] + 1
        done_s = state["seconds"]
        print(f"  {log}: resumed after epoch {start}", flush=True)
    last_save = time.time()
    for epoch in range(start, max_epochs):
        model.train()
        # pair j = (slot j // n_tr, query tr[j % n_tr]); with K = 1 this is a plain permutation of tr
        perm = torch.randperm(data.share_k * n_tr, generator=gen).to(dev)
        for i in range(0, perm.numel(), bs):
            j = perm[i:i + bs]
            b = tr[j % n_tr]
            loss = data.loss(model(data.inputs(j // n_tr, b)), b, cfg).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        if epochs_per_layer is not None:
            take = target == epoch + 1
        else:
            if stop_on == "rate":
                score = data.per_probe(data.correct(model, val_docs, metric_of(cfg)).float().mean(-1))
            elif stop_on == "vocab":
                hit = data.vocab_correct(model, val_docs, metric_of(cfg), ks=(1,))[1]
                score = data.per_probe(hit.float().mean(-1))
            else:
                score = -data.mean_loss(model, va, cfg)
            take = (score > best) & active
            best = torch.where(take, score, best)
            best_epoch = torch.where(take, torch.full_like(best_epoch, epoch), best_epoch)
        if take.any():
            with torch.no_grad():
                for p, s in zip(params, snap):
                    s[take] = p.detach()[take]
        if epochs_per_layer is None:
            active &= (epoch - best_epoch) < patience
            if not bool(active.any()):
                break
        if resume is not None and epoch + 1 < max_epochs and time.time() - last_save >= resume.every_s:
            resume.save(key, {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                              "snap": snap, "best": best, "best_epoch": best_epoch, "active": active,
                              "gen": gen.get_state(), "rng": _rng_state(dev),
                              "seconds": done_s + time.time() - t0})
            last_save = time.time()
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.copy_(s)
    model.eval()
    if resume is not None:
        resume.clear()
    info = {"epochs_run": epoch + 1, "seconds": round(done_s + time.time() - t0, 1)}
    if epochs_per_layer is None:
        info.update(best_epoch=best_epoch.tolist(), best=best.tolist())
    if log:
        msg = f"best val {np.mean(info['best']):.4f}" if "best" in info else "refit"
        print(f"  {log}: {msg} ({info['epochs_run']} epochs, {info['seconds']:.0f}s)", flush=True)
    return model, info


def promote(scores: dict, k: int):
    """The k configurations ranking best for some layer: sorted by their best rank over
    layers, then by mean rank. scores: {config: per-layer rate}."""
    names = list(scores)
    s = np.array([scores[n] for n in names])                # (C, G)
    ranks = (-s).argsort(0).argsort(0)                      # 0 = best for that layer
    order = sorted(range(len(names)), key=lambda i: (ranks[i].min(), ranks[i].mean()))
    return [names[i] for i in order[:k]]


# ------------------------------------------------------------------------ evaluation

def doc_counts(data, model, docs, metric, seen_types):
    """Per test document and layer: correct queries, all and those whose token type
    never occurred as a training target ("unseen")."""
    ok = data.correct(model, docs, metric)                         # (G, n)
    vocab = data.vocab_correct(model, docs, metric)                # k -> (G, n)
    idx = data.queries(docs)
    q_doc = data.q_doc[idx]
    unseen = ~torch.isin(data.q_tok[idx], seen_types)
    pos = {int(d): i for i, d in enumerate(docs)}
    col = torch.tensor([pos[int(d)] for d in q_doc.tolist()], device=data.device)
    D, G = len(docs), data.num_cells

    def per_doc(mask, hits):
        c = torch.zeros(G, D, device=data.device).index_add_(1, col[mask], hits[:, mask].float())
        n = torch.zeros(D, device=data.device).index_add_(0, col[mask], torch.ones(int(mask.sum()), device=data.device))
        return c.tolist(), n.tolist()

    all_mask = torch.ones_like(unseen)
    c, n = per_doc(all_mask, ok)
    cu, nu = per_doc(unseen, ok)
    out = {"correct": c, "n": n, "correct_unseen": cu, "n_unseen": nu}
    for k, hits in vocab.items():
        out[f"correct_v{k}"] = per_doc(all_mask, hits)[0]
        out[f"correct_unseen_v{k}"] = per_doc(unseen, hits)[0]
    return out


def bootstrap(correct, n, groups, rng, reps: int):
    """correct (G, D), n (D,) -> rate and 95% CI per layer, over all docs and per group
    label (documents resampled with replacement within the selection)."""
    correct, n, groups = np.asarray(correct, float), np.asarray(n, float), np.asarray(groups)

    def one(sel):
        c, m = correct[:, sel], n[sel]
        if m.sum() == 0:
            return None
        rate = c.sum(1) / m.sum()
        w = rng.multinomial(sel.size, np.full(sel.size, 1.0 / sel.size), size=reps)   # (reps, docs)
        boot = (w @ c.T) / np.maximum(w @ m, 1e-12)[:, None]
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        return {"rate": rate.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "docs": int(sel.size),
                "queries": int(m.sum())}

    out = {"all": one(np.arange(n.size))}
    out["by_group"] = {str(g): one(np.nonzero(groups == g)[0]) for g in dict.fromkeys(groups.tolist())}
    return out


def summarize(counts, data, docs, rng, reps):
    doms = [data.domains[d] for d in docs]
    lens = [data.doc_lengths[d] for d in docs]
    start = copy.deepcopy(rng)   # the vocabulary-wide CIs reuse the document resamples of "all" (paired)
    by_dom = bootstrap(counts["correct"], counts["n"], doms, rng, reps)
    by_len = bootstrap(counts["correct"], counts["n"], lens, rng, reps)
    unseen = bootstrap(counts["correct_unseen"], counts["n_unseen"], doms, rng, reps)
    out = {"all": by_dom["all"], "by_source": by_dom["by_group"], "by_length": by_len["by_group"],
           "unseen_types": unseen["all"]}
    for k in VOCAB_KS:
        if f"correct_v{k}" in counts:   # absent in units computed before vocabulary-wide scoring existed
            out[f"vocab_top{k}"] = bootstrap(counts[f"correct_v{k}"], counts["n"], doms,
                                             copy.deepcopy(start), reps)["all"]
            out[f"vocab_top{k}_unseen"] = bootstrap(counts[f"correct_unseen_v{k}"], counts["n_unseen"], doms,
                                                    copy.deepcopy(start), reps)["all"]
    return out


def mean_counts(runs):
    return {k: np.mean([r[k] for r in runs], 0).tolist() for k in runs[0]}


def per_layer_counts(chosen, counts_by_cfg):
    """Counts of the selected probes: cell l from configuration chosen[l] (the
    configuration chosen for the probe that reads cell l)."""
    out = {}
    for k in counts_by_cfg[chosen[0]]:
        if k.startswith("correct"):
            out[k] = [counts_by_cfg[c][k][l] for l, c in enumerate(chosen)]
    for k in ("n", "n_unseen"):
        out[k] = counts_by_cfg[chosen[0]][k]
    return out


# ---------------------------------------------------------------------------- protocol

def run(data, splits, args, ckpt, rng_seed, resume=None, weights=None):
    """The protocol with one *probe* as the unit of every choice: halving promotes a
    configuration that ranks high for some probe, each probe picks its own configuration
    by CV, and gets its own refit epochs. Per-layer protocol: probe = layer. Rates are
    reported per cell, each from the probe that reads it. ``args.objective`` sets the
    validation score of every choice (see the module docstring); ``weights`` (a
    ProbeWeights) saves the refit probes."""
    G = data.probes
    objective = getattr(args, "objective", "prompt")
    stop = "vocab" if objective == "vocab" else "rate"
    rng = np.random.default_rng([args.seed, 11])
    pool = dict(PAPER_CONFIGS)
    pool.update({f"r{i:02d}": sample_config(rng, objective) for i in range(args.n_random)})
    res = {"pool": pool, "rungs": [], "cv": {}, "test": {}}
    seed_of = {c: args.seed * 1000 + i for i, c in enumerate(pool)}
    search = splits["search"]

    # 2. successive halving on the search split
    survivors = list(pool)
    for r, budget in enumerate(args.rungs):
        rung = {}
        for c in survivors:
            key = f"rung{r}/{c}"
            if ckpt.get(key) is None:
                _, info = train(pool[c], data, search["train"], search["val"], seed_of[c], max_epochs=budget,
                                patience=budget, stop_on=stop, log=f"rung {r} ({budget} ep) {c}", resume=resume,
                                key=key)
                ckpt.put(key, {"val": info["best"], "best_epoch": info["best_epoch"]})
            rung[c] = ckpt.get(key)
        res["rungs"].append(rung)
        last = r == len(args.rungs) - 1
        k = args.finalists if last else max(args.finalists, math.ceil(len(survivors) / args.eta))
        survivors = promote({c: v["val"] for c, v in rung.items()}, k)
    res["finalists"] = survivors

    # 3. cross-validated selection, per layer
    for c in survivors:
        folds = []
        for f, fold in enumerate(splits["cv"]):
            key = f"cv/{c}/fold{f}"
            if ckpt.get(key) is None:
                _, info = train(pool[c], data, fold["train"], fold["val"], seed_of[c] + 100 * (f + 1),
                                max_epochs=args.max_epochs, patience=args.patience, stop_on=stop,
                                log=f"cv {c} fold {f}", resume=resume, key=key)
                ckpt.put(key, {"val": info["best"], "best_epoch": info["best_epoch"]})
            folds.append(ckpt.get(key))
        res["cv"][c] = {"val": [f["val"] for f in folds], "best_epoch": [f["best_epoch"] for f in folds],
                        "val_mean": np.mean([f["val"] for f in folds], 0).tolist()}
    cv_mean = np.array([res["cv"][c]["val_mean"] for c in survivors])   # (C, G)
    chosen = [survivors[i] for i in cv_mean.argmax(0)]
    res["chosen"] = chosen
    res["chosen_cv_rate"] = cv_mean.max(0).tolist()  # optimistic: it is what was maximised
    res["final_epochs"] = {c: [int(np.median(res["cv"][c]["best_epoch"], 0)[l]) + 1 for l in range(G)]
                           for c in dict.fromkeys(chosen)}

    # 4. refit on all dev documents, test once
    seen = torch.unique(data.q_tok[data.queries(splits["dev"])])
    test_sets = {"id": splits["test_id"], "ood": splits["test_ood"]}
    counts, files = {}, {}
    for c in dict.fromkeys(chosen):
        runs = []
        mine = [p for p in range(G) if chosen[p] == c]
        files[c] = []
        for s in range(args.seeds):
            key = f"final/{c}/seed{s}"
            if ckpt.get(key) is None:
                seed = seed_of[c] + 10_000 * (s + 1)
                model, _ = train(pool[c], data, splits["dev"], seed=seed,
                                 epochs_per_layer=res["final_epochs"][c], log=f"final {c} seed {s}",
                                 resume=resume, key=key)
                if weights is not None:
                    weights.save(c, s, pool[c], model, mine, data, res["final_epochs"][c], seed)
                ckpt.put(key, {name: doc_counts(data, model, docs, metric_of(pool[c]), seen)
                               for name, docs in test_sets.items() if docs})
            runs.append(ckpt.get(key))
            if weights is not None:
                if weights.path(c, s).is_file():
                    files[c].append(f"{weights.root.name}/{weights.path(c, s).name}")
                else:
                    files[c].append(None)
                    print(f"  warning: no saved weights for {key} (finished before weight saving existed)",
                          flush=True)
        counts[c] = {name: mean_counts([r[name] for r in runs]) for name in runs[0]}

    # reference: the paper's probes, paper training procedure, same dev / test documents
    for name, cfg in PAPER_CONFIGS.items():
        key = f"paper/{name}"
        if ckpt.get(key) is None:
            model, _ = train(cfg, data, search["train"], search["val"], seed_of[name] + 50_000,
                             max_epochs=args.paper_max_epochs, patience=20, stop_on="loss", log=f"paper {name}",
                             resume=resume, key=key)
            ckpt.put(key, {n: doc_counts(data, model, docs, metric_of(cfg), seen)
                           for n, docs in test_sets.items() if docs})
    if data.x.shape[-1] == data.E.shape[-1]:  # g = identity (T5-small)
        for metric in ("cos", "l2"):
            key = f"naive/{metric}"
            if ckpt.get(key) is None:
                ckpt.put(key, {n: doc_counts(data, Identity(), docs, metric, seen)
                               for n, docs in test_sets.items() if docs})

    rng_b = np.random.default_rng([rng_seed, 13])
    selected = {}
    chosen_per_cell = [chosen[p] for p in data.probe_of]
    for name, docs in test_sets.items():
        if docs:
            selected[name] = summarize(per_layer_counts(chosen_per_cell, {c: counts[c][name] for c in counts}), data,
                                       docs, rng_b, args.bootstrap)
    res["test"]["selected"] = selected
    if weights is not None:
        res["weights"] = {"dir": weights.root.name, "files": files, "seeds": args.seeds,
                          "note": "probe p of seed s: the file of chosen[p], entry with probe == p (load_selected)"}
    for prefix in ("paper/", "naive/"):
        for key, val in ckpt.units.items():
            if key.startswith(prefix):
                label = key.replace("/", "_") if prefix == "naive/" else key[len(prefix):]
                res["test"][label] = {n: summarize(val[n], data, test_sets[n], rng_b, args.bootstrap) for n in val}
    return res


def search_units(res):
    """Checkpoint units of the search (halving rounds) and CV of a finished run."""
    units = {}
    for r, rung in enumerate(res["rungs"]):
        for c, v in rung.items():
            units[f"rung{r}/{c}"] = v
    for c, v in res["cv"].items():
        for f, (val, best) in enumerate(zip(v["val"], v["best_epoch"])):
            units[f"cv/{c}/fold{f}"] = {"val": val, "best_epoch": best}
    return units


def retest_compatible(prev, settings, splits):
    """Same settings (feature files compared by name) and the same documents."""
    a, b = dict(prev["args"]), dict(settings)
    fa, fb = a.pop("features"), b.pop("features")
    names = (lambda f: [Path(x).name for x in (f if isinstance(f, list) else [f])])
    return a == b and names(fa) == names(fb) and prev["splits"] == splits


def main(argv=None):
    from tokid.experiments import Checkpoint

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, nargs="+", required=True,
                   help="features/<tag>_multidoc.pt. Several files of the same prompts (e.g. the 7 points of the "
                        "ELF trajectory, in time order) for --share.")
    p.add_argument("--share", choices=SHARE_MODES, default="none",
                   help="none: one probe per (file, layer); time: one probe per layer for all files "
                        "(time-independent); layer: one probe per file for all layers (layer-independent); "
                        "both: one probe for everything.")
    p.add_argument("--tag", default=None, help="Run name with several files (default <common prefix>_share-<mode>).")
    p.add_argument("--reserve-gb", type=float, default=6.0,
                   help="GPU memory kept free for training on the main GPU; the hidden states fill the other "
                        "visible GPUs first and the main GPU only with what is left.")
    p.add_argument("--objective", choices=OBJECTIVES, default="prompt",
                   help="prompt: the paper's task (candidates = the prompt's distinct tokens) for early stopping, "
                        "halving and CV selection, and contrastive loss over those tokens; vocab: the whole "
                        "vocabulary (top-1 over every embedding row; contrastive loss over every row). vocab runs "
                        "write to results/<run>_vocab/ by default.")
    p.add_argument("--n-random", type=int, default=48, help="Random configurations (plus the 4 paper probes).")
    p.add_argument("--rungs", type=int, nargs="+", default=[5, 20], help="Epoch budget of each halving round.")
    p.add_argument("--eta", type=int, default=3, help="Keep 1/eta of the configurations after each round.")
    p.add_argument("--finalists", type=int, default=6)
    p.add_argument("--cv-folds", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=200, help="Cap for the CV runs (early stopping).")
    p.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs) of the CV runs.")
    p.add_argument("--seeds", type=int, default=3, help="Seeds of the final refit.")
    p.add_argument("--paper-max-epochs", type=int, default=300)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--unit-save-min", type=float, default=15.0,
                   help="Save the state of the training unit in progress every this many minutes (<out>.unit.pt), "
                        "so a unit longer than one job resumes after a requeue instead of restarting.")
    p.add_argument("--retest", action="store_true",
                   help="Re-test a finished run (--out exists) with the current test metrics, e.g. vocabulary-wide "
                        "identifiability: the search and CV results are taken from it, the chosen and paper probes "
                        "are retrained with the same seeds and tested again. The old file is kept as "
                        "<out stem>.prompt-only.json.")
    args = p.parse_args(argv)

    multi = len(args.features) > 1 or args.share != "none"
    device = resolve_device(args.device)
    if device.type == "cuda":  # probes only (features are fixed): TF32 matmuls, ~3x faster
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    feats = load_cells(args.features, device, args.reserve_gb)
    metas = feats["variant_meta"]
    meta = dict(feats["meta"])
    V, L = len(metas), feats["hidden"].shape[0] // len(metas)
    if multi:
        tags = [m["tag"] for m in metas]
        prefix = os.path.commonprefix(tags)
        prefix = prefix[:prefix.rfind("_")] if "_" in prefix else prefix
        meta.update(tag=args.tag or f"{prefix or 'multi'}_share-{args.share}", variants=tags, share=args.share,
                    layers_per_variant=L, hidden_file=[m["hidden_file"] for m in metas])
    suffix = "_vocab" if args.objective == "vocab" else ""
    out = args.out or Path("results") / ((f"{meta['tag']}_{meta['dataset']}" if multi else args.features[0].stem)
                                         + suffix) / "selection.json"
    if args.retest and not out.exists():
        p.error(f"--retest: {out} does not exist (run the selection first)")
    if out.exists() and not args.overwrite and not args.retest:
        p.error(f"{out} exists (use --overwrite, or --retest to re-test it)")
    out.parent.mkdir(parents=True, exist_ok=True)
    cells = probe_cells(args.share, V, L)
    data = QueryData(feats, device, cells)
    splits = protocol_splits(feats["domains"], feats["doc_lengths"].tolist(), meta["ood_domains"],
                             cv_folds=args.cv_folds, seed=args.seed)
    shards = feats["shard_cells"]
    del feats
    print(f"[selection] {meta['tag']} / {meta['dataset']}: {data.num_docs} prompts, {data.x.shape[1]} queries, "
          f"hidden {tuple(data.x.shape)} ({V} file(s) x {L} layers; cells per device {shards}); "
          f"share={args.share}: {data.probes} probe(s) x {data.share_k} cell(s); dev {len(splits['dev'])} / "
          f"test id {len(splits['test_id'])} / test ood {len(splits['test_ood'])} prompts; device={device}",
          flush=True)
    # Settings of a single-file per-layer run are unchanged, so its old checkpoints still resume.
    settings = {k: v for k, v in vars(args).items() if k not in ("device", "out", "overwrite", "tag", "reserve_gb",
                                                                  "unit_save_min", "retest")}
    settings["features"] = [str(f) for f in args.features] if multi else str(args.features[0])
    if not multi:
        settings.pop("share")
    if args.objective == "prompt":   # unchanged settings: runs and checkpoints from before --objective still match
        settings.pop("objective")
    prev = None
    if args.retest:
        prev = json.loads(out.read_text())
        if not retest_compatible(prev, settings, splits):
            raise SystemExit(f"--retest: {out} was made with other settings or documents:\n  {prev['args']}\n"
                             f"now:\n  {settings}")
        if all(f"vocab_top{VOCAB_KS[0]}" in v for v in prev["test"]["selected"].values()):
            print(f"[selection] {out} already has the vocabulary-wide test: nothing to do")
            return
    partial = out.with_name(out.name + (".retest.partial" if args.retest else ".partial"))
    unit = out.with_name(out.name + ".unit.pt")
    if args.overwrite:
        partial.unlink(missing_ok=True)
        unit.unlink(missing_ok=True)
    ckpt = Checkpoint(partial, settings)
    if prev is not None:   # search and CV from the finished run; final and paper units are redone
        ckpt.units.update({k: v for k, v in search_units(prev).items() if k not in ckpt.units})
    resume = UnitResume(unit, settings, every_s=60 * args.unit_save_min)
    if multi:
        cell_names = [f"{metas[c // L]['tag']}/layer{c % L}" for c in range(data.num_cells)]
    else:
        cell_names = [f"{meta['tag']}/layer{c}" for c in range(data.num_cells)]
    weights = ProbeWeights(out.parent / "probes", cell_names, args.objective)
    print(f"[selection] objective={args.objective}: early stopping, halving and CV selection on the "
          f"{'vocabulary-wide top-1' if args.objective == 'vocab' else 'same-prompt'} validation rate; "
          f"selected probes' weights -> {weights.root}", flush=True)
    t0 = time.time()
    res = run(data, splits, args, ckpt, args.seed, resume, weights)
    result = {"meta": meta, "experiment": "selection", "args": settings, "splits": splits,
              "layers": list(range(data.num_cells)), "elapsed_s": time.time() - t0, "objective": args.objective,
              **res}
    if multi:
        result.update(share=args.share, variants=meta["variants"], layers_per_variant=L,
                      cells=[[c // L, c % L] for c in range(data.num_cells)], probe_cells=cells.tolist())
    if prev is not None:
        for k in ("chosen", "final_epochs"):
            if result[k] != prev[k]:
                raise SystemExit(f"--retest: {k} differs from {out}; not overwriting it")
        diff = {t: float(np.max(np.abs(np.subtract(v["all"]["rate"], prev["test"]["selected"][t]["all"]["rate"]))))
                for t, v in result["test"]["selected"].items() if t in prev["test"]["selected"]}
        result["retest"] = {"elapsed_s_original": prev.get("elapsed_s"), "max_abs_diff_prompt_rate": diff}
        print(f"[selection] retest: same-prompt test rates differ from the original run by at most {diff} "
              "(0 up to GPU non-determinism)", flush=True)
        backup = out.with_name(out.stem + ".prompt-only.json")
        if not backup.exists():
            backup.write_text(json.dumps(prev))
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, out)
    partial.unlink(missing_ok=True)
    unit.unlink(missing_ok=True)
    sel = res["test"]["selected"]
    for name in sel:
        print(f"[selection] test {name}: " + " ".join(f"{r:.3f}" for r in sel[name]["all"]["rate"]))
        if "vocab_top1" in sel[name]:
            print(f"[selection] test {name}, vocabulary-wide top-1: "
                  + " ".join(f"{r:.3f}" for r in sel[name]["vocab_top1"]["rate"]))
    print(f"[selection] done in {result['elapsed_s'] / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
