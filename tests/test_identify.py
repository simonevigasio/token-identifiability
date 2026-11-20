import torch

from tokid.identify import nn_correct, vector_keys


def test_vector_keys_detects_duplicates():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    k = vector_keys(x)
    assert k[0] == k[2] and k[0] != k[1]
    k2 = vector_keys(torch.stack([x, x.flip(0)]))
    assert k2.shape == (2, 3)


def test_duplicate_token_counts_as_identified():
    # Sentence 0 = tokens [a, b, a]; query 0's prediction is closest to the *second* a.
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    cand = torch.stack([a, b, a])
    keys = vector_keys(cand)
    sent = torch.zeros(3, dtype=torch.long)
    pred = torch.stack([a * 2, b, a])[None]
    ok = nn_correct(pred, sent, keys, cand, sent, keys, "cos")
    assert ok.all()


def test_candidates_restricted_to_same_sentence():
    cand = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    sent = torch.tensor([0, 0, 1])
    keys = torch.arange(3)
    # Query in sentence 1 whose target is row 2; row 0 (sentence 0) is an even closer match.
    pred = torch.tensor([[[1.0, 0.0]]])
    ok = nn_correct(pred, sent[2:], keys[2:], cand, sent, keys, "l2")
    assert ok.item()  # only row 2 is a valid candidate
    # A query in sentence 0 with target row 1 but prediction near row 0 -> wrong.
    ok = nn_correct(pred, sent[1:2], keys[1:2], cand, sent, keys, "l2")
    assert not ok.item()


def test_cos_and_l2_differ():
    cand = torch.tensor([[1.0, 0.0], [10.0, 1.0]])
    sent = torch.zeros(2, dtype=torch.long)
    keys = torch.arange(2)
    pred = torch.tensor([[[10.0, 0.0]]])  # direction of row 0, position near row 1
    tgt = torch.tensor([0])
    assert nn_correct(pred, sent[:1], tgt, cand, sent, keys, "cos").item()
    assert not nn_correct(pred, sent[:1], tgt, cand, sent, keys, "l2").item()


def test_per_layer_candidates_and_chunking():
    torch.manual_seed(0)
    n_sent, n_tok, d, groups = 7, 5, 16, 3
    cand = torch.randn(groups, n_sent * n_tok, d)
    sent = torch.arange(n_sent).repeat_interleave(n_tok)
    keys = vector_keys(cand)
    ok = nn_correct(cand.clone(), sent, keys, cand, sent, keys, "cos", chunk_sentences=2)
    assert ok.shape == (groups, n_sent * n_tok) and ok.all()
    # Shuffling predictions within sentences makes (almost) everything wrong.
    perm = torch.cat([torch.arange(s * n_tok, (s + 1) * n_tok).roll(1) for s in range(n_sent)])
    assert nn_correct(cand[:, perm], sent, keys, cand, sent, keys, "l2").float().mean() < 0.1
