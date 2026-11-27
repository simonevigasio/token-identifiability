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


# ------------------------------------------------------------ model selection (tokid.selection)

class BatchedProbe(nn.Module):
    """G independent probes of a configurable architecture (the search space of
    tokid.selection): optional fixed input standardization, ``len(hidden)`` GELU layers
    of the given widths, dropout after each hidden layer, and an output layer. Biases
    are always used in hidden layers and optional in the output layer, so
    ``hidden=()`` / ``bias=False`` is the paper's linear probe and ``hidden=(1000,)`` /
    ``bias=True`` its MLP. Glorot-uniform weights, zero biases."""

    def __init__(self, groups: int, d_in: int, d_out: int, hidden=(), bias: bool = False, dropout: float = 0.0,
                 mean=None, std=None, generator=None):
        super().__init__()
        dims = [d_in, *hidden, d_out]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            w = nn.Parameter(torch.empty(groups, a, b))
            glorot_uniform_(w, a, b, generator)
            self.weights.append(w)
            if i < len(dims) - 2 or bias:
                self.biases.append(nn.Parameter(torch.zeros(groups, 1, b)))
        self.out_bias = bias
        self.dropout = dropout
        self.register_buffer("mean", None if mean is None else mean.reshape(groups, 1, d_in).float())
        self.register_buffer("inv_std", None if std is None else (1.0 / std.clamp_min(1e-6)).reshape(groups, 1, d_in).float())

    def forward(self, x):                        # x: (G, B, d_in)
        x = x.float()
        if self.mean is not None:
            x = (x - self.mean) * self.inv_std
        last = len(self.weights) - 1
        for i, w in enumerate(self.weights):
            has_bias = i < last or self.out_bias
            x = torch.baddbmm(self.biases[i], x, w) if has_bias else torch.bmm(x, w)
            if i < last:
                x = F.gelu(x)
                if self.dropout and self.training:
                    x = F.dropout(x, self.dropout, training=True)
        return x


def info_nce_loss(pred, cand, cand_mask, slot, tau: float):
    """Per-layer contrastive loss: softmax over the candidate tokens of each query's own
    prompt (cosine similarity / tau), cross-entropy with the true token.

    pred: (G, B, d); cand: (B, U, d) unit-norm candidates (padded); cand_mask: (B, U) True
    where a candidate exists; slot: (B,) index of the true token among them."""
    logits = torch.einsum("gbd,bud->gbu", F.normalize(pred, dim=-1), cand) / tau
    logits = logits.masked_fill(~cand_mask[None], float("-inf"))
    groups = pred.shape[0]
    return F.cross_entropy(logits.flatten(0, 1), slot.repeat(groups), reduction="none").view(groups, -1).mean(-1)


class _VocabCrossEntropy(torch.autograd.Function):
    """Cross-entropy over every row of a fixed matrix, computed in row chunks.

    Rows z (N, d) are scored against En (V, d): logits = z En^T / tau. The full (N, V)
    logits are never stored: each chunk's loss and its gradient with respect to z,
    (softmax - onehot) En / tau, are computed in the forward pass, so memory stays at
    one (chunk, V) block. Same arithmetic as autograd (one product for the logits, one
    for the gradient); En gets no gradient (frozen embeddings)."""

    @staticmethod
    def forward(ctx, z, En, target, tau, chunk):
        n = z.shape[0]
        loss = torch.empty(n, dtype=z.dtype, device=z.device)
        need_grad = ctx.needs_input_grad[0]
        grad = torch.empty_like(z) if need_grad else None
        for i in range(0, n, chunk):
            zi, ti = z[i:i + chunk], target[i:i + chunk]
            logits = (zi @ En.T).div_(tau)                                    # (c, V)
            lse = torch.logsumexp(logits, dim=-1)
            loss[i:i + chunk] = lse - logits.gather(1, ti[:, None])[:, 0]
            if need_grad:
                p = logits.sub_(lse[:, None]).exp_()                         # softmax, in place
                p[torch.arange(ti.numel(), device=z.device), ti] -= 1.0
                grad[i:i + chunk] = (p @ En).div_(tau)
        ctx.save_for_backward(grad if need_grad else torch.empty(0))
        return loss

    @staticmethod
    def backward(ctx, g):
        grad, = ctx.saved_tensors
        return grad * g[:, None], None, None, None, None


def info_nce_vocab_loss(pred, En, target, tau: float, chunk: int = 4096):
    """Per-layer contrastive loss over the **whole vocabulary**: softmax over every row of
    the embedding matrix (cosine similarity / tau), cross-entropy with the true token.

    pred: (G, B, d); En: (V, d) unit-norm embedding rows (frozen, no learned classifier);
    target: (B,) token ids. Returns (G,) mean losses. ``chunk`` rows of (G*B) at a time
    (a chunk of 4096 rows over T5's 32k tokens is a 0.5 GiB block)."""
    groups, b = pred.shape[0], pred.shape[1]
    z = F.normalize(pred, dim=-1).reshape(groups * b, -1)
    loss = _VocabCrossEntropy.apply(z, En.to(z.dtype), target.repeat(groups), float(tau), int(chunk))
    return loss.view(groups, b).mean(-1)
