import copy

import pytest
import torch

from tokid.identify import nn_rate, vector_keys
from tokid.probes import BatchedLinear, BatchedMLP, make_probe, predict, train_probe


def _data(groups=3, n=300, d_in=12, d_out=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(1, n, d_out, generator=g)
    maps = torch.randn(groups, d_out, d_in, generator=g)
    x = torch.bmm(y.expand(groups, -1, -1), maps) + 0.3 * torch.randn(groups, n, d_in, generator=g)
    return x, y


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("loss", ["cos", "l2"])
def test_joint_training_equals_separate_training(kind, loss):
    torch.manual_seed(0)
    x, y = _data()
    tr, va = slice(0, 200), slice(200, 300)
    joint = make_probe(kind, 3, x.shape[-1], y.shape[-1], mlp_hidden=16, seed=1)
    singles = []
    for g in range(3):
        m = make_probe(kind, 1, x.shape[-1], y.shape[-1], mlp_hidden=16, seed=1)
        with torch.no_grad():
            for p, pj in zip(m.parameters(), joint.parameters()):
                p.copy_(pj[g:g + 1])
        singles.append(m)
    kw = dict(loss=loss, lr=3e-3, batch_size=32, patience=3, max_epochs=40, seed=5)
    joint, info = train_probe(copy.deepcopy(joint), x[:, tr], y[:, tr], x[:, va], y[:, va], **kw)
    for g, m in enumerate(singles):
        m, info_g = train_probe(m, x[g:g + 1, tr], y[:, tr], x[g:g + 1, va], y[:, va], **kw)
        assert info_g["best_epoch"][0] == info["best_epoch"][g]
        assert info_g["epochs_run"] == info["layer_epochs"][g]
        for p, pj in zip(m.parameters(), joint.parameters()):
            torch.testing.assert_close(p[0], pj[g], rtol=1e-4, atol=1e-5)


def test_single_matches_batched_forward():
    torch.manual_seed(0)
    x = torch.randn(4, 10, 6)
    for model in (BatchedLinear(4, 6, 5), BatchedMLP(4, 6, 5, hidden=7)):
        full = model(x)
        for g in range(4):
            torch.testing.assert_close(model.single(g, x[g]), full[g])


def test_glorot_init_bounds():
    m = make_probe("linear", 2, 768, 512)
    bound = (6 / (768 + 512)) ** 0.5
    assert m.weight.abs().max() <= bound and m.weight.abs().max() > 0.9 * bound


def test_linear_probe_learns_rotation():
    # x = noisy random linear image of y -> a linear probe must identify ~all tokens.
    torch.manual_seed(0)
    x, y = _data(groups=2, n=1200, d_in=24, d_out=16)
    sent = torch.arange(1200) // 20
    keys = vector_keys(y[0])
    tr, va, te = slice(0, 800), slice(800, 1000), slice(1000, 1200)
    model = make_probe("linear", 2, 24, 16, seed=0)
    model, _ = train_probe(model, x[:, tr], y[:, tr], x[:, va], y[:, va], "cos", lr=3e-3, patience=10)
    rate = nn_rate(predict(model, x[:, te]), sent[te], keys[te], y, sent, keys[None], "cos")
    assert (rate > 0.95).all(), rate
