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
