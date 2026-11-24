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
ELF De-En (conditional, ``ELFCondExtractor``): the sequence is [prompt | target | </s> pad]
      (128 rows), as in upstream conditional generation. Only the *prompt* rows are
      returned: ``tokens`` = T5 embeddings of the prompt ids, ``hidden[l]`` = prompt rows.
"""

from dataclasses import dataclass, asdict
import importlib
from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ELF_SOURCE = ROOT / ".deps/elf-pytorch"
# Official PyTorch checkpoints, pinned. Architecture as in the matching training config.
ELF_WEIGHTS = {
    # train_owt_ELF-B.yml, unconditional
    "owt": dict(repo="embedded-language-flows/ELF-B-owt-torch", revision="146f84133c1389bfd4ef47f14ec7a955da22faa7",
                filename="checkpoint_95085", max_length=1024, conditional=False),
    # train_de-en_ELF-B.yml, WMT14 German prompt -> English target (eval: cond_sampling_configs.yml)
    "de-en": dict(repo="embedded-language-flows/ELF-B-de-en-torch", revision="93ce98315a2dec985cdefb8d8e62ab618e896ec4",
                  filename="checkpoint_880600", max_length=128, max_input_length=64, conditional=True),
}
ELF_REPO, ELF_REVISION, ELF_FILENAME = (ELF_WEIGHTS["owt"][k] for k in ("repo", "revision", "filename"))
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
    # Classifier-free guidance on the prompt (conditional checkpoint only). It only
    # changes x_pred: in sampling, the x-prediction fed back as self-conditioning is the
    # guided x_uncond + g * (x_cond - x_uncond). The probed states are always the
    # conditional pass.
    guidance: float = 1.0


def check_observation(obs: ELFObservation, conditional: bool):
    if obs.mode not in {"denoiser", "decoder"}:
        raise ValueError("mode must be 'denoiser' or 'decoder'")
    if not 0.0 <= obs.t <= 1.0:
        raise ValueError("t must be in [0, 1]")
    if obs.self_cond not in {"zero", "pred"}:
        raise ValueError("self_cond must be 'zero' or 'pred'")
    if obs.self_cond == "pred" and obs.mode == "decoder":
        raise ValueError("ELF's decoder step always uses zero self-conditioning")
    if obs.guidance != 1.0 and not (conditional and obs.self_cond == "pred"):
        raise ValueError("guidance only acts on the self-conditioning x_pred of a conditional checkpoint")


def capture_rows(elf, rows: slice) -> tuple[dict, list]:
    """Hooks storing ``rows`` of the block-1 input (key 0) and of every block output (1..L)."""
    captured = {}
    hooks = [elf.blocks[0].register_forward_pre_hook(
        lambda _m, args: captured.__setitem__(0, args[0][0, rows].clone()))]
    for l, block in enumerate(elf.blocks, start=1):
        hooks.append(block.register_forward_hook(
            lambda _m, _i, out, l=l: captured.__setitem__(l, out[0, rows].clone())))
    return captured, hooks


class ELFExtractor:
    def __init__(self, encoder, elf, observation=ELFObservation()):
        check_observation(observation, conditional=False)
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

        # The language rows are always the last n rows (prefix tokens are prepended).
        captured, hooks = capture_rows(self.elf, slice(-n, None))
        try:
            self.elf(inputs, t, attention_mask=mask, deterministic=True, self_cond_cfg_scale=cfg,
                     decoder_step_active=decoder)
        finally:
            for h in hooks:
                h.remove()
        hidden = [captured[l] for l in range(len(self.elf.blocks) + 1)]
        return {"tokens": tokens, "hidden": hidden}


class ELFCondExtractor:
    """Prompt rows of the conditional (translation) ELF-B, as in upstream conditional
    generation (generation.py: test_generation_cond, utils/generation_utils.py).

    Sequence (``max_length`` rows): [prompt (n <= max_input_length) | target | </s> pad].
    * T5 latent: prompt rows attend only to the prompt (so they are T5(prompt) alone);
      target and pad rows attend to every non-pad token (encoder_utils.build_self_attn_cond_masks).
    * Prompt rows are always the clean latent (add_noise / restore_cond keep them fixed);
      target and pad rows are z_t = t*x0 + (1-t)*2*eps, i.e. pure noise at t = 0.
    * Self-conditioning half: "zero" = zeros with the clean prompt restored, which is the
      first sampling step (restore_cond(zeros, cond_seq)); "pred" = the guided x-prediction
      of upstream _forward_sample at the same t, prompt rows restored.
    * Decoder (t = 1): [x0, zeros] with mode tokens on, as _dlm_decode_batch (whose
      self-conditioning half has no prompt restored).
    * Like upstream sampling, ELF itself gets no attention mask (all rows visible).
    The probed pass is the conditional one; with guidance the extra unconditional pass
    only enters through x_pred.
    """

    def __init__(self, encoder, elf, observation=ELFObservation(), max_length=128, max_input_length=64,
                 pad_id=1, t_eps=0.05):
        check_observation(observation, conditional=True)
        if elf.max_length != max_length:
            raise ValueError(f"ELF was built for {elf.max_length} rows, expected {max_length}")
        self.encoder = encoder.eval().requires_grad_(False)
        self.elf = elf.eval().requires_grad_(False)
        self.obs = observation
        self.max_length, self.max_input_length, self.pad_id = max_length, max_input_length, pad_id
        from types import SimpleNamespace
        import importlib

        self.sampling = importlib.import_module("utils.sampling_utils")  # upstream (import_elf put it on sys.path)
        self.sampler_config = SimpleNamespace(t_eps=t_eps, self_cond_prob=0.5,
                                              num_self_cond_cfg_tokens=elf.num_self_cond_cfg_tokens)

    @property
    def device(self):
        return next(self.elf.parameters()).device

    def sequence(self, ids: torch.Tensor, target_ids: torch.Tensor):
        """-> (input_ids (1, S), encoder mask (1, S, S), prompt mask (1, S, 1)), as upstream collate_fn."""
        n, S = ids.numel(), self.max_length
        if n > self.max_input_length:
            raise ValueError(f"prompt of {n} tokens exceeds max_input_length {self.max_input_length}")
        seq = torch.cat([ids, target_ids])[:S]
        valid = seq.numel()
        seq = torch.cat([seq, torch.full((S - valid,), self.pad_id, dtype=seq.dtype)])
        pos = torch.arange(S)
        is_cond, is_valid = pos < n, pos < valid
        enc_mask = (is_cond[:, None] & is_cond[None, :]) | (~is_cond[:, None] & is_valid[None, :])
        return seq[None], enc_mask[None].float(), is_cond[None, :, None].float()

    @torch.no_grad()
    def __call__(self, ids: torch.Tensor, index: int = 0, target_ids=None) -> dict:
        if target_ids is None:
            raise ValueError("the conditional checkpoint needs the target ids of every prompt")
        obs, dev = self.obs, self.device
        ids = ids.cpu()
        n = ids.numel()
        seq, enc_mask, m = (x.to(dev) for x in self.sequence(ids, torch.as_tensor(target_ids, dtype=ids.dtype)))
        tokens = self.encoder.get_input_embeddings()(ids.to(dev))
        x0 = self.encoder(input_ids=seq, attention_mask=enc_mask).last_hidden_state
        x0 = (x0 - obs.latent_mean) / obs.latent_std
        t = torch.full((1,), obs.t, device=dev)
        cfg = torch.full((1,), obs.cfg_scale, device=dev)
        decoder = obs.mode == "decoder"
        if decoder:
            inputs = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
        else:
            gen = torch.Generator(device="cpu").manual_seed(obs.noise_seed * 1_000_003 + index)
            noise = torch.randn(x0.shape, generator=gen).to(dev)
            z = obs.t * x0 + (1.0 - obs.t) * noise * obs.noise_scale
            z = m * x0 + (1 - m) * z                 # add_noise(..., cond_seq_mask): prompt stays clean
            self_cond = m * x0                       # restore_cond(zeros, cond_seq, cond_seq_mask)
            if obs.self_cond == "pred":
                _, self_cond = self.sampling._forward_sample(
                    model=self.elf, z=z, t_batch=t, x_pred_prev=self_cond, config=self.sampler_config,
                    cfg_scale=obs.guidance, self_cond_cfg_scale=obs.cfg_scale, cond_seq=x0, cond_seq_mask=m[..., 0])
            inputs = torch.cat([z, self_cond], dim=-1)

        # Text rows are the last max_length rows; the prompt is their first n.
        S = self.max_length
        captured, hooks = capture_rows(self.elf, slice(-S, -S + n))
        try:
            self.elf(inputs, t, deterministic=True, self_cond_cfg_scale=cfg, decoder_step_active=decoder)
        finally:
            for h in hooks:
                h.remove()
        return {"tokens": tokens, "hidden": [captured[l] for l in range(len(self.elf.blocks) + 1)]}


def load_elf(source=DEFAULT_ELF_SOURCE, observation=ELFObservation(), cache_dir=None,
             local_files_only=False, checkpoint=None, device="cpu", weights="owt"):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, T5EncoderModel

    spec = ELF_WEIGHTS[weights]
    module = import_elf(source)
    kw = dict(revision=T5_REVISIONS[ELF_ENCODER], cache_dir=cache_dir, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(ELF_ENCODER, **kw)
    encoder = T5EncoderModel.from_pretrained(ELF_ENCODER, use_safetensors=True, **kw)
    official = checkpoint is None
    if official:
        checkpoint = hf_hub_download(spec["repo"], spec["filename"], revision=spec["revision"], cache_dir=cache_dir,
                                     local_files_only=local_files_only)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    key = "ema_params1" if payload.get("ema_params1") is not None else "params"
    # Architecture of train_owt_ELF-B.yml / train_de-en_ELF-B.yml (identical except max_length)
    elf = module.ELF_B(text_encoder_dim=512, max_length=spec["max_length"], bottleneck_dim=128, num_time_tokens=4,
                       num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
                       vocab_size=payload[key]["unembed_bias"].numel())
    elf.load_state_dict(payload[key], strict=True)
    step = payload.get("step")
    del payload
    encoder = encoder.to(device=device, dtype=torch.float32)
    elf = elf.to(device=device, dtype=torch.float32)
    meta = {
        "model": "ELF-B" if weights == "owt" else f"ELF-B ({weights})", "weights": weights,
        "elf_repo": spec["repo"] if official else None,
        "elf_revision": spec["revision"] if official else None, "checkpoint": str(checkpoint),
        "checkpoint_step": int(step) if step is not None else None, "weights_key": key,
        "encoder": ELF_ENCODER, "encoder_revision": encoder.config._commit_hash,
        "num_layers": len(elf.blocks), "d_model": elf.hidden_size, "observation": asdict(observation),
    }
    if spec["conditional"]:
        meta.update(rows="prompt", max_length=spec["max_length"], max_input_length=spec["max_input_length"])
        extractor = ELFCondExtractor(encoder, elf, observation, spec["max_length"], spec["max_input_length"],
                                     pad_id=tokenizer.eos_token_id)  # train_de-en_ELF-B.yml: pad_token eos
    else:
        extractor = ELFExtractor(encoder, elf, observation)
    return extractor, tokenizer, meta
