"""The function approximators g_l of Sec. 4.2 / App. B.1, trained for G layers at once.

* linear perceptron without bias:  g(e) = e W
* MLP: one hidden layer of 1000 units, GELU, biases.
* Glorot-uniform weights, zero biases; Adam(lr=1e-4, betas=(0.9, 0.999)); batch 256;
  early stopping when validation loss has not improved for 20 epochs.
* loss: cosine distance (1 - cos) or L2 (squared Euclidean distance).

All G layers share the same tokens (same sentence split), only their input features
differ. Each layer gets its own, independent parameters (leading G dimension); the
total loss is the sum of the per-layer losses, so gradients never mix across layers,
and Adam is element-wise. Training G probes together is therefore identical to
training G probes separately on the same batch sequence. Early stopping is tracked
per layer: each layer keeps the weights of its own best validation epoch.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def glorot_uniform_(w: torch.Tensor, fan_in: int, fan_out: int, generator=None):
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        return w.uniform_(-bound, bound, generator=generator)


class BatchedLinear(nn.Module):
    """G independent bias-free linear maps (d_in -> d_out)."""

    def __init__(self, groups: int, d_in: int, d_out: int, generator=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, d_in, d_out))
        glorot_uniform_(self.weight, d_in, d_out, generator)

    def forward(self, x):                        # x: (G, B, d_in)
        return torch.bmm(x, self.weight)

    def single(self, g: int, x):                 # apply probe g to x: (..., d_in)
        return x @ self.weight[g]


class BatchedMLP(nn.Module):
    """G independent MLPs d_in -> hidden (GELU) -> d_out."""

    def __init__(self, groups: int, d_in: int, d_out: int, hidden: int = 1000, generator=None):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(groups, d_in, hidden))
        self.b1 = nn.Parameter(torch.zeros(groups, 1, hidden))
        self.w2 = nn.Parameter(torch.empty(groups, hidden, d_out))
        self.b2 = nn.Parameter(torch.zeros(groups, 1, d_out))
        glorot_uniform_(self.w1, d_in, hidden, generator)
        glorot_uniform_(self.w2, hidden, d_out, generator)

    def forward(self, x):
        return torch.baddbmm(self.b2, F.gelu(torch.baddbmm(self.b1, x, self.w1)), self.w2)

    def single(self, g: int, x):
        return F.gelu(x @ self.w1[g] + self.b1[g, 0]) @ self.w2[g] + self.b2[g, 0]


def make_probe(kind: str, groups: int, d_in: int, d_out: int, mlp_hidden: int = 1000, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    if kind == "linear":
        return BatchedLinear(groups, d_in, d_out, gen)
    if kind == "mlp":
        return BatchedMLP(groups, d_in, d_out, mlp_hidden, gen)
    raise ValueError(f"unknown probe {kind!r}")


def per_layer_loss(pred, target, loss: str):
    """Mean over tokens, one value per layer. pred, target: (G, B, d)."""
    if loss == "cos":
        return (1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-8)).mean(-1)
    if loss == "l2":
        return (pred - target).pow(2).sum(-1).mean(-1)
    raise ValueError(f"unknown loss {loss!r}")


def _expand(y, groups):
    return y.expand(groups, -1, -1) if y.shape[0] == 1 else y


@torch.no_grad()
def predict(model, x, chunk: int = 8192):
    """x: (G, N, d_in) -> (G, N, d_out), in chunks of tokens."""
    return torch.cat([model(x[:, i:i + chunk]) for i in range(0, x.shape[1], chunk)], dim=1)


@torch.no_grad()
def eval_loss(model, x, y, loss, chunk: int = 8192):
    groups, n = x.shape[0], x.shape[1]
    y = _expand(y, groups)
    total = torch.zeros(groups, device=x.device)
    for i in range(0, n, chunk):
        xb, yb = x[:, i:i + chunk], y[:, i:i + chunk]
        total += per_layer_loss(model(xb), yb, loss) * xb.shape[1]
    return total / n


def train_probe(model, x_train, y_train, x_val, y_val, loss: str, lr: float = 1e-4, batch_size: int = 256,
                patience: int = 20, max_epochs: int = 1000, seed: int = 0, log_every: int = 0):
    """Train the G probes in ``model`` with per-layer early stopping.

    x_*: (G, N, d_in); y_*: (G or 1, N, d_out). Returns (model with best weights, info).
    """
    groups, n = x_train.shape[0], x_train.shape[1]
    y_train, y_val = _expand(y_train, groups), _expand(y_val, groups)
    device = x_train.device
    model = model.to(device)
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator().manual_seed(seed)

    best_val = torch.full((groups,), float("inf"), device=device)
    best_epoch = torch.zeros(groups, dtype=torch.long, device=device)
    best_params = [p.detach().clone() for p in params]
    active = torch.ones(groups, dtype=torch.bool, device=device)
    stop_epoch = torch.zeros(groups, dtype=torch.long, device=device)  # epochs each layer "ran"
    epoch = -1
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_loss = per_layer_loss(model(x_train[:, idx]), y_train[:, idx], loss).sum()
            opt.zero_grad(set_to_none=True)
            batch_loss.backward()
            opt.step()
        model.eval()
        val = eval_loss(model, x_val, y_val, loss)
        # Layers that already ran out of patience are finished: their snapshot is frozen,
        # exactly as if they had been trained (and stopped) on their own.
        improved = (val < best_val) & active
        if improved.any():
            best_val = torch.where(improved, val, best_val)
            best_epoch = torch.where(improved, torch.full_like(best_epoch, epoch), best_epoch)
            with torch.no_grad():
                for p, bp in zip(params, best_params):
                    bp[improved] = p.detach()[improved]
        stop_epoch = torch.where(active, torch.full_like(stop_epoch, epoch + 1), stop_epoch)
        active &= (epoch - best_epoch) < patience
        if log_every and epoch % log_every == 0:
            print(f"      epoch {epoch:4d} val {val.mean().item():.4f} active layers {int(active.sum())}/{groups}",
                  flush=True)
        if not bool(active.any()):
            break
    with torch.no_grad():
        for p, bp in zip(params, best_params):
            p.copy_(bp)
    model.eval()
    return model, {"best_epoch": best_epoch.tolist(), "best_val_loss": best_val.tolist(),
                   "layer_epochs": stop_epoch.tolist(), "epochs_run": epoch + 1,
                   "stopped_early": epoch + 1 < max_epochs}
