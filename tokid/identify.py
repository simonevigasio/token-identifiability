"""Nearest-neighbour token identification (paper Sec. 4).

c_l(e^l_i) = NN(g_l(e^l_i)) is a 1-nearest-neighbour lookup among the candidates *of
the same input sentence*. e^l_i is identified if the retrieved candidate equals the
true target x_i.

Correctness is checked with *keys*: two candidates share a key iff their vectors are
identical. This matters for T5/ELF, whose token embeddings carry no position, so a
word that occurs twice in a sentence has two identical x vectors; retrieving either
one recovers the token (for BERT, x_i includes the position embedding, so all
candidates are distinct and keys reduce to positions).
"""

import torch
import torch.nn.functional as F


def vector_keys(vectors: torch.Tensor) -> torch.Tensor:
    """(..., N, d) -> (..., N) integer ids, equal iff rows are bitwise identical."""
    flat = vectors.reshape(-1, vectors.shape[-2], vectors.shape[-1])
    keys = [torch.unique(v, dim=0, return_inverse=True)[1] for v in flat]
    return torch.stack(keys).reshape(vectors.shape[:-1])


@torch.no_grad()
def nn_correct(pred, query_sent, target_key, cand, cand_sent, cand_key, metric: str, chunk_sentences: int = 32):
    """Boolean (G, Q) tensor: is the nearest same-sentence candidate the right one?

    pred:       (G, Q, d)   predictions g(e) for Q query tokens (sorted by sentence)
    query_sent: (Q,)        sentence id of each query
    target_key: (Q,) or (Gc, Q) key of each query's true target
    cand:       (C, d) or (Gc, C, d) candidate vectors (all tokens of the relevant sentences)
    cand_sent:  (C,)        sentence id of each candidate
    cand_key:   (C,) or (Gc, C) key of each candidate
    metric:     "cos" (max cosine similarity) or "l2" (min Euclidean distance)
    Gc is 1 or G (per-layer candidate sets, e.g. hidden -> hidden identification).
    """
    groups = pred.shape[0]
    dev = pred.device
    if cand.dim() == 2:
        cand = cand[None]
    if cand_key.dim() == 1:
        cand_key = cand_key[None]
    if target_key.dim() == 1:
        target_key = target_key[None]
    query_sent, cand_sent = query_sent.to(dev), cand_sent.to(dev)
    target_key, cand_key, cand = target_key.to(dev), cand_key.to(dev), cand.to(dev)
    correct = torch.zeros(groups, pred.shape[1], dtype=torch.bool, device=dev)
    sentences = torch.unique(query_sent)
    for i in range(0, sentences.numel(), chunk_sentences):
        chunk = sentences[i:i + chunk_sentences]
        q = torch.nonzero(torch.isin(query_sent, chunk), as_tuple=True)[0]
        c = torch.nonzero(torch.isin(cand_sent, chunk), as_tuple=True)[0]
        p = pred[:, q]                                   # (G, q, d)
        cc = cand[:, c]                                  # (Gc, c, d)
        if metric == "cos":
            score = torch.matmul(F.normalize(p, dim=-1), F.normalize(cc, dim=-1).transpose(1, 2))
        elif metric == "l2":
            score = -torch.cdist(p, cc.expand(groups, -1, -1).contiguous(), compute_mode="donot_use_mm_for_euclid_dist")
        else:
            raise ValueError(f"unknown metric {metric!r}")
        same = query_sent[q][:, None] == cand_sent[c][None, :]
        score = score.masked_fill(~same, float("-inf"))
        j = score.argmax(-1)                             # (G, q) index into c
        got = torch.gather(cand_key[:, c].expand(groups, -1), 1, j)
        correct[:, q] = got == target_key[:, q].expand(groups, -1)
    return correct


def nn_rate(*args, **kwargs) -> torch.Tensor:
    """Token identifiability rate per layer, shape (G,)."""
    return nn_correct(*args, **kwargs).float().mean(-1)
