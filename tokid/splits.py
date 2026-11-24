"""Sentence-level train/val/test splits (paper App. B.1).

"We split the data by sentences into train/validation/test according to a 70/15/15
split [...] we perform the experiments using 10-fold cross validation."
Ten disjoint 15% test sets cannot exist, so each fold is an independent random
70/15/15 partition of the sentences (repeated random sub-sampling). Tokens of one
sentence never end up in two sets.
"""

import numpy as np
import torch


def sentence_splits(num_sentences: int, num_folds: int = 10, fractions=(0.7, 0.15, 0.15), seed: int = 0):
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    folds = []
    for k in range(num_folds):
        perm = np.random.default_rng(seed + k).permutation(num_sentences)
        n_train = int(round(fractions[0] * num_sentences))
        n_val = int(round(fractions[1] * num_sentences))
        folds.append({
            "train": np.sort(perm[:n_train]),
            "val": np.sort(perm[n_train:n_train + n_val]),
            "test": np.sort(perm[n_train + n_val:]),
        })
    return folds


def token_index(sentence_ids: torch.Tensor, sentences) -> torch.Tensor:
    """Indices (sorted) of all tokens belonging to the given sentences."""
    chosen = torch.as_tensor(np.asarray(sentences), dtype=sentence_ids.dtype)
    return torch.nonzero(torch.isin(sentence_ids, chosen), as_tuple=True)[0]


def domain_splits(sentence_domains, domain_names, val_fraction: float = 0.15, seed: int = 0):
    """Leave-one-domain-out: fold k tests on every sentence of domain k; the other
    domains are split randomly into train / validation (``val_fraction`` of them)."""
    domains = np.asarray(sentence_domains)
    folds = []
    for k, name in enumerate(domain_names):
        test = np.nonzero(domains == name)[0]
        if test.size == 0:
            raise ValueError(f"no sentences of domain {name!r}")
        rest = np.random.default_rng(seed + k).permutation(np.nonzero(domains != name)[0])
        n_val = int(round(val_fraction * rest.size))
        folds.append({"train": np.sort(rest[n_val:]), "val": np.sort(rest[:n_val]), "test": test,
                      "held_out": name})
    return folds


def unseen_token_splits(num_sentences: int, token_ids, num_folds: int = 10, fractions=(0.7, 0.15, 0.15),
                        seed: int = 0):
    """Sentence splits as ``sentence_splits`` plus a token-type split per fold.

    A random set of token types covering ``fractions[2]`` of all token occurrences is
    held out (``test_types``): probes are trained and validated only on the other types
    and tested only on these, so every test token is a word the probe never saw as a
    target. Candidates for the nearest-neighbour lookup are still all tokens."""
    token_ids = np.asarray(token_ids)
    types, counts = np.unique(token_ids, return_counts=True)
    folds = sentence_splits(num_sentences, num_folds, fractions, seed)
    for k, fold in enumerate(folds):
        order = np.random.default_rng(seed + 10_000 + k).permutation(types.size)
        cum = np.cumsum(counts[order])
        take = (cum - counts[order]) < fractions[2] * token_ids.size  # up to and including the crossing type
        fold["test_types"] = np.sort(types[order][take])
    return folds
