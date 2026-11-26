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

Example:
    python -m tokid.selection --features features/elf-b-xsum_t0.3_multidoc.pt
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tokid.extract import resolve_device
from tokid.probes import BatchedProbe, info_nce_loss, per_layer_loss

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


def load_cells(paths, device, reserve_gb: float = 4.0):
    """Several multidoc feature files of the same prompts and queries (e.g. the 7 points
    of the ELF trajectory) -> the features of the first, with ``hidden`` a CellStore of
    all V*L cells (cell = variant * L + layer) and ``variant_meta`` the V metas.

    On CUDA the cells fill the main GPU (keeping ``reserve_gb`` free for training) and
    then the other visible GPUs. Cells are loaded one at a time: RAM never holds more
    than one (Q, d) array."""
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
        devices = [main] + [torch.device("cuda", i) for i in range(torch.cuda.device_count()) if i != main.index]
        caps = []
        for dv in devices:
            free, _ = torch.cuda.mem_get_info(dv)
            keep = (reserve_gb if dv == main else 0.5) * 2**30
            caps.append(max(0, int((free - keep) // cell_bytes)))
        if sum(caps) < C:
            raise SystemExit(f"the hidden states need {C * cell_bytes / 2**30:.1f} GiB ({C} cells); the "
                             f"{len(devices)} visible GPU(s) hold {sum(caps)} cells. Request more GPUs "
                             f"(e.g. sbatch --gres=gpu:2 --mem=48G) or lower --reserve-gb.")
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

def sample_config(rng) -> dict:
    arch = str(rng.choice(["linear", "affine", "mlp"], p=[0.2, 0.2, 0.6]))
    mlp = arch == "mlp"
    loss = str(rng.choice(["cos", "l2", "infonce"]))
    return {
        "arch": arch, "depth": int(rng.choice([1, 2])) if mlp else 0,
        "width": int(rng.choice([512, 1024, 2048])) if mlp else 0,
        "dropout": float(rng.choice([0.0, 0.1, 0.2])) if mlp else 0.0,
        "standardize": bool(rng.random() < 0.5), "loss": loss,
        "tau": float(math.exp(rng.uniform(math.log(0.02), math.log(0.2)))) if loss == "infonce" else None,
        "lr": float(math.exp(rng.uniform(math.log(1e-5), math.log(3e-3)))),
        "weight_decay": float(rng.choice([0.0, 1e-3, 1e-2, 1e-1])),
        "batch_size": int(rng.choice([256, 1024])),
    }


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
    probe's cells (``stop_on="rate"``), or loss (``"loss"``, the paper); each probe keeps
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
    idx = data.queries(docs)
    q_doc = data.q_doc[idx]
    unseen = ~torch.isin(data.q_tok[idx], seen_types)
    pos = {int(d): i for i, d in enumerate(docs)}
    col = torch.tensor([pos[int(d)] for d in q_doc.tolist()], device=data.device)
    D, G = len(docs), data.num_cells

    def per_doc(mask):
        c = torch.zeros(G, D, device=data.device).index_add_(1, col[mask], ok[:, mask].float())
        n = torch.zeros(D, device=data.device).index_add_(0, col[mask], torch.ones(int(mask.sum()), device=data.device))
        return c.tolist(), n.tolist()

    all_mask = torch.ones_like(unseen)
    c, n = per_doc(all_mask)
    cu, nu = per_doc(unseen)
    return {"correct": c, "n": n, "correct_unseen": cu, "n_unseen": nu}


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
    by_dom = bootstrap(counts["correct"], counts["n"], doms, rng, reps)
    by_len = bootstrap(counts["correct"], counts["n"], lens, rng, reps)
    unseen = bootstrap(counts["correct_unseen"], counts["n_unseen"], doms, rng, reps)
    return {"all": by_dom["all"], "by_source": by_dom["by_group"], "by_length": by_len["by_group"],
            "unseen_types": unseen["all"]}


def mean_counts(runs):
    return {k: np.mean([r[k] for r in runs], 0).tolist() for k in runs[0]}


def per_layer_counts(chosen, counts_by_cfg):
    """Counts of the selected probes: cell l from configuration chosen[l] (the
    configuration chosen for the probe that reads cell l)."""
    out = {}
    for k in ("correct", "correct_unseen"):
        out[k] = [counts_by_cfg[c][k][l] for l, c in enumerate(chosen)]
    for k in ("n", "n_unseen"):
        out[k] = counts_by_cfg[chosen[0]][k]
    return out


# ---------------------------------------------------------------------------- protocol

def run(data, splits, args, ckpt, rng_seed, resume=None):
    """The protocol with one *probe* as the unit of every choice: halving promotes a
    configuration that ranks high for some probe, each probe picks its own configuration
    by CV, and gets its own refit epochs. Per-layer protocol: probe = layer. Rates are
    reported per cell, each from the probe that reads it."""
    G = data.probes
    rng = np.random.default_rng([args.seed, 11])
    pool = dict(PAPER_CONFIGS)
    pool.update({f"r{i:02d}": sample_config(rng) for i in range(args.n_random)})
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
                                patience=budget, log=f"rung {r} ({budget} ep) {c}", resume=resume, key=key)
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
                                max_epochs=args.max_epochs, patience=args.patience, log=f"cv {c} fold {f}",
                                resume=resume, key=key)
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
    counts = {}
    for c in dict.fromkeys(chosen):
        runs = []
        for s in range(args.seeds):
            key = f"final/{c}/seed{s}"
            if ckpt.get(key) is None:
                model, _ = train(pool[c], data, splits["dev"], seed=seed_of[c] + 10_000 * (s + 1),
                                 epochs_per_layer=res["final_epochs"][c], log=f"final {c} seed {s}",
                                 resume=resume, key=key)
                ckpt.put(key, {name: doc_counts(data, model, docs, metric_of(pool[c]), seen)
                               for name, docs in test_sets.items() if docs})
            runs.append(ckpt.get(key))
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
    for prefix in ("paper/", "naive/"):
        for key, val in ckpt.units.items():
            if key.startswith(prefix):
                label = key.replace("/", "_") if prefix == "naive/" else key[len(prefix):]
                res["test"][label] = {n: summarize(val[n], data, test_sets[n], rng_b, args.bootstrap) for n in val}
    return res


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
    p.add_argument("--reserve-gb", type=float, default=4.0,
                   help="GPU memory kept free on the main GPU for training; the hidden states fill the rest and "
                        "then the other visible GPUs.")
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
    out = args.out or Path("results") / (f"{meta['tag']}_{meta['dataset']}" if multi else args.features[0].stem) \
        / "selection.json"
    if out.exists() and not args.overwrite:
        p.error(f"{out} exists (use --overwrite)")
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
                                                                  "unit_save_min")}
    settings["features"] = [str(f) for f in args.features] if multi else str(args.features[0])
    if not multi:
        settings.pop("share")
    partial = out.with_name(out.name + ".partial")
    unit = out.with_name(out.name + ".unit.pt")
    if args.overwrite:
        partial.unlink(missing_ok=True)
        unit.unlink(missing_ok=True)
    ckpt = Checkpoint(partial, settings)
    resume = UnitResume(unit, settings, every_s=60 * args.unit_save_min)
    t0 = time.time()
    res = run(data, splits, args, ckpt, args.seed, resume)
    result = {"meta": meta, "experiment": "selection", "args": settings, "splits": splits,
              "layers": list(range(data.num_cells)), "elapsed_s": time.time() - t0, **res}
    if multi:
        result.update(share=args.share, variants=meta["variants"], layers_per_variant=L,
                      cells=[[c // L, c % L] for c in range(data.num_cells)], probe_cells=cells.tolist())
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, out)
    partial.unlink(missing_ok=True)
    unit.unlink(missing_ok=True)
    sel = res["test"]["selected"]
    for name in sel:
        print(f"[selection] test {name}: " + " ".join(f"{r:.3f}" for r in sel[name]["all"]["rate"]))
    print(f"[selection] done in {result['elapsed_s'] / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
