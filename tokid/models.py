"""Per-layer hidden states e^l_i of T5 encoders and ELF-B (paper notation, Sec. 2/4).

For every model we return, for one tokenized sentence of length n:

* ``tokens``  -- T5 input token embeddings x_i (n, d_tok). This is the discrete input
                 of both models; T5 has no absolute position embedding, so x_i is just
                 the row of the shared embedding table.
* ``hidden``  -- list of L+1 tensors (n, d). ``hidden[0]`` is the input of the first
                 transformer block (the paper's x_i for BERT), ``hidden[l]`` is the
                 output of block l.

T5  : hidden[0] = token embeddings, hidden[1..L] = encoder block outputs, exactly
      ``output_hidden_states`` of HF ``T5EncoderModel`` (the last one includes T5's
      final RMS layer norm, i.e. it is the encoder output ELF consumes).
ELF : tokens -> frozen T5-small encoder -> (h - mean)/std = z -> ELF-B.
      hidden[0] = text rows entering block 1 (after self-cond projection and the
      128-d bottleneck projection), hidden[l] = text rows after ELF block l.
      Prefix rows (time / CFG / mode tokens) are dropped.
"""

from dataclasses import dataclass, asdict
import importlib
from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ELF_SOURCE = ROOT / ".deps/elf-pytorch"
ELF_REPO = "embedded-language-flows/ELF-B-owt-torch"
ELF_REVISION = "146f84133c1389bfd4ef47f14ec7a955da22faa7"
ELF_FILENAME = "checkpoint_95085"
ELF_ENCODER = "google-t5/t5-small"
T5_REVISIONS = {"google-t5/t5-small": "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"}


# --------------------------------------------------------------------------- T5

class T5Extractor:
    def __init__(self, encoder):
        self.encoder = encoder.eval().requires_grad_(False)

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @torch.no_grad()
    def __call__(self, ids: torch.Tensor, index: int = 0) -> dict:
        ids = ids.to(self.device)[None]
        out = self.encoder(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
        tokens = self.encoder.get_input_embeddings()(ids)[0]
        return {"tokens": tokens, "hidden": [h[0] for h in out.hidden_states]}


def load_t5(name: str, cache_dir=None, local_files_only=False, device="cpu"):
    from transformers import AutoTokenizer, T5EncoderModel

    revision = T5_REVISIONS.get(name)
    kw = dict(revision=revision, cache_dir=cache_dir, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(name, **kw)
    encoder = T5EncoderModel.from_pretrained(name, use_safetensors=True, **kw).to(device=device, dtype=torch.float32)
    meta = {"model": name, "revision": encoder.config._commit_hash, "num_layers": encoder.config.num_layers,
            "d_model": encoder.config.d_model}
    return T5Extractor(encoder), tokenizer, meta


# -------------------------------------------------------------------------- ELF

def import_elf(source: Path):
    src = Path(source).resolve() / "src"
    if not (src / "modules/model.py").is_file():
        raise FileNotFoundError(f"ELF source missing at {src}. Run: python scripts/prepare_elf.py")
    existing = sys.modules.get("modules.model")
    if existing is not None and Path(existing.__file__).resolve() != src / "modules/model.py":
        raise RuntimeError("Another `modules.model` is already imported; use a fresh Python process.")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))  # upstream uses absolute `modules.*` imports
    module = importlib.import_module("modules.model")
    if not hasattr(module.ELF, "forward"):
        raise ValueError("ELF source must be the pytorch_elf branch, not the JAX main branch.")
    return module


class VariableLengthRoPE(nn.Module):
    """Upstream precomputes RoPE tables for exactly prefix+1024 rows. Slice them to the
    actual length; prefix rows (cos=1, sin=0) and text positions 0..n-1 are unchanged."""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x):
        n = x.shape[-2]
        if n > self.original.freqs_cos.shape[0]:
            raise ValueError("Sequence exceeds ELF's rotary-position capacity.")
        cos = self.original.freqs_cos[:n].to(x.dtype)
        sin = self.original.freqs_sin[:n].to(x.dtype)
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return x * cos + rotated * sin


@dataclass(frozen=True)
class ELFObservation:
    t: float = 1.0               # flow time; 1 = clean data (ELF convention)
    mode: str = "denoiser"       # "denoiser" or "decoder" (model-mode tokens on)
    cfg_scale: float = 1.0       # self-conditioning CFG scale token
    latent_mean: float = 0.0     # train_owt_ELF-B.yml
    latent_std: float = 0.2      # train_owt_ELF-B.yml
    noise_scale: float = 2.0     # denoiser_noise_scale, used only if t < 1
    noise_seed: int = 0
    # Second half of the input. "zero": no self-conditioning (first sampling step, and the
    # decoder step). "pred": ELF's own x-prediction from a first forward pass with zero
    # self-conditioning, as in the later steps of sampling (_forward_sample_self_cond).
    self_cond: str = "zero"


class ELFExtractor:
    def __init__(self, encoder, elf, observation=ELFObservation()):
        if observation.mode not in {"denoiser", "decoder"}:
            raise ValueError("mode must be 'denoiser' or 'decoder'")
        if not 0.0 <= observation.t <= 1.0:
            raise ValueError("t must be in [0, 1]")
        if observation.self_cond not in {"zero", "pred"}:
            raise ValueError("self_cond must be 'zero' or 'pred'")
        if observation.self_cond == "pred" and observation.mode == "decoder":
            raise ValueError("ELF's decoder step always uses zero self-conditioning")
        self.encoder = encoder.eval().requires_grad_(False)
        self.elf = elf.eval().requires_grad_(False)
        if not isinstance(self.elf.feat_rope, VariableLengthRoPE):
            self.elf.feat_rope = VariableLengthRoPE(self.elf.feat_rope)
        self.obs = observation

    @property
    def device(self):
        return next(self.elf.parameters()).device

    @torch.no_grad()
    def __call__(self, ids: torch.Tensor, index: int = 0) -> dict:
        obs, dev = self.obs, self.device
        ids = ids.to(dev)[None]
        n = ids.shape[1]
        if n > self.elf.max_length:
            raise ValueError(f"sequence of {n} tokens exceeds ELF max_length {self.elf.max_length}")
        mask = torch.ones_like(ids)
        tokens = self.encoder.get_input_embeddings()(ids)[0]
        encoded = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        z = (encoded - obs.latent_mean) / obs.latent_std
        if obs.t < 1.0:
            # utils/sampling_utils.add_noise: z_t = t*x0 + (1-t)*noise*scale
            gen = torch.Generator(device="cpu").manual_seed(obs.noise_seed * 1_000_003 + index)
            noise = torch.randn(z.shape, generator=gen).to(dev)
            z = obs.t * z + (1.0 - obs.t) * noise * obs.noise_scale
        t = torch.full((1,), obs.t, device=dev)
        cfg = torch.full((1,), obs.cfg_scale, device=dev)
        decoder = obs.mode == "decoder"
        inputs = torch.cat([z, torch.zeros_like(z)], dim=-1)  # zero self-conditioning
        if obs.self_cond == "pred":
            x_pred, _ = self.elf(inputs, t, attention_mask=mask, deterministic=True,
                                 self_cond_cfg_scale=cfg, decoder_step_active=decoder)
            inputs = torch.cat([z, x_pred], dim=-1)

        captured = {}
        # The language rows are always the last n rows (prefix tokens are prepended).
        hooks = [self.elf.blocks[0].register_forward_pre_hook(
            lambda _m, args: captured.__setitem__(0, args[0][0, -n:].clone()))]
        for l, block in enumerate(self.elf.blocks, start=1):
            hooks.append(block.register_forward_hook(
                lambda _m, _i, out, l=l: captured.__setitem__(l, out[0, -n:].clone())))
        try:
            self.elf(inputs, t, attention_mask=mask, deterministic=True, self_cond_cfg_scale=cfg,
                     decoder_step_active=decoder)
        finally:
            for h in hooks:
                h.remove()
        hidden = [captured[l] for l in range(len(self.elf.blocks) + 1)]
        return {"tokens": tokens, "hidden": hidden}


def load_elf(source=DEFAULT_ELF_SOURCE, observation=ELFObservation(), cache_dir=None,
             local_files_only=False, checkpoint=None, device="cpu"):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, T5EncoderModel

    module = import_elf(source)
    kw = dict(revision=T5_REVISIONS[ELF_ENCODER], cache_dir=cache_dir, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(ELF_ENCODER, **kw)
    encoder = T5EncoderModel.from_pretrained(ELF_ENCODER, use_safetensors=True, **kw)
    official = checkpoint is None
    if official:
        checkpoint = hf_hub_download(ELF_REPO, ELF_FILENAME, revision=ELF_REVISION, cache_dir=cache_dir,
                                     local_files_only=local_files_only)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    key = "ema_params1" if payload.get("ema_params1") is not None else "params"
    # Architecture of train_owt_ELF-B.yml
    elf = module.ELF_B(text_encoder_dim=512, max_length=1024, bottleneck_dim=128, num_time_tokens=4,
                       num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
                       vocab_size=payload[key]["unembed_bias"].numel())
    elf.load_state_dict(payload[key], strict=True)
    step = payload.get("step")
    del payload
    encoder = encoder.to(device=device, dtype=torch.float32)
    elf = elf.to(device=device, dtype=torch.float32)
    meta = {
        "model": "ELF-B", "elf_repo": ELF_REPO if official else None,
        "elf_revision": ELF_REVISION if official else None, "checkpoint": str(checkpoint),
        "checkpoint_step": int(step) if step is not None else None, "weights_key": key,
        "encoder": ELF_ENCODER, "encoder_revision": encoder.config._commit_hash,
        "num_layers": len(elf.blocks), "d_model": elf.hidden_size, "observation": asdict(observation),
    }
    return ELFExtractor(encoder, elf, observation), tokenizer, meta
