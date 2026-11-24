"""Extractor wiring on tiny randomly initialised models (no downloads)."""

import pytest
import torch
import torch.nn.functional as F
from transformers import T5Config, T5EncoderModel

from tokid.models import (DEFAULT_ELF_SOURCE, ELFExtractor, ELFObservation, T5Extractor, VariableLengthRoPE,
                          encode_masked)


def tiny_t5(d_model=32):
    torch.manual_seed(0)
    cfg = T5Config(vocab_size=100, d_model=d_model, d_kv=8, d_ff=64, num_layers=3, num_heads=4)
    return T5EncoderModel(cfg).eval()


def test_t5_extractor_layers():
    enc = tiny_t5()
    ids = torch.tensor([5, 7, 9, 5, 1])
    out = T5Extractor(enc)(ids)
    assert len(out["hidden"]) == 4
    torch.testing.assert_close(out["hidden"][0], out["tokens"])
    last = enc(input_ids=ids[None]).last_hidden_state[0]
    torch.testing.assert_close(out["hidden"][-1], last)


elf_available = (DEFAULT_ELF_SOURCE / "src/modules/model.py").is_file()


@pytest.fixture(scope="module")
def tiny_elf():
    if not elf_available:
        pytest.skip("ELF source not prepared (python scripts/prepare_elf.py)")
    from tokid.models import import_elf

    module = import_elf(DEFAULT_ELF_SOURCE)
    torch.manual_seed(0)
    elf = module.ELF(text_encoder_dim=32, max_length=64, hidden_size=48, depth=3, num_heads=4,
                     bottleneck_dim=8, num_time_tokens=4, num_self_cond_cfg_tokens=4,
                     num_model_mode_tokens=4, vocab_size=100).eval()
    return module, elf


def test_rope_adapter_matches_upstream_at_full_length(tiny_elf):
    _, elf = tiny_elf
    rope = elf.feat_rope.original if isinstance(elf.feat_rope, VariableLengthRoPE) else elf.feat_rope
    x = torch.randn(1, 4, rope.freqs_cos.shape[0], rope.freqs_cos.shape[1])
    torch.testing.assert_close(VariableLengthRoPE(rope)(x), rope(x))


@pytest.mark.parametrize("mode", ["denoiser", "decoder"])
def test_elf_extractor_captures_language_rows(tiny_elf, mode):
    _, elf = tiny_elf
    enc = tiny_t5()
    ext = ELFExtractor(enc, elf, ELFObservation(mode=mode))
    ids = torch.tensor([5, 7, 9, 5, 1])
    out = ext(ids)
    assert len(out["hidden"]) == 4 and all(h.shape == (5, 48) for h in out["hidden"])
    with torch.no_grad():
        z = enc(input_ids=ids[None]).last_hidden_state / 0.2
        inp = torch.cat([z, torch.zeros_like(z)], -1)
        # hidden[0] = bottleneck projection of the self-conditioning projection
        torch.testing.assert_close(out["hidden"][0], elf.text_proj(elf.self_cond_proj(inp))[0])
        # the last captured state must reproduce ELF's own output head
        ref, _ = elf(inp, torch.ones(1), attention_mask=torch.ones(1, 5, dtype=torch.long),
                     self_cond_cfg_scale=torch.ones(1), decoder_step_active=mode == "decoder")
        torch.testing.assert_close(elf.final_layer(out["hidden"][-1][None]), ref)


def test_elf_noise_is_seeded(tiny_elf):
    _, elf = tiny_elf
    ext = ELFExtractor(tiny_t5(), elf, ELFObservation(t=0.5, noise_seed=3))
    ids = torch.tensor([5, 7, 9])
    a, b, c = ext(ids, index=0), ext(ids, index=0), ext(ids, index=1)
    torch.testing.assert_close(a["hidden"][2], b["hidden"][2])
    assert not torch.allclose(a["hidden"][2], c["hidden"][2])


def _noisy_latent(enc, ids, obs, index=0):
    z = enc(input_ids=ids[None]).last_hidden_state / obs.latent_std
    gen = torch.Generator().manual_seed(obs.noise_seed * 1_000_003 + index)
    return obs.t * z + (1 - obs.t) * torch.randn(z.shape, generator=gen) * obs.noise_scale


def test_elf_self_cond_pred_matches_upstream_sampler(tiny_elf):
    """--elf-self-cond pred = upstream _forward_sample_self_cond fed its own x-prediction."""
    from types import SimpleNamespace
    import importlib

    _, elf = tiny_elf
    enc = tiny_t5()
    obs = ELFObservation(t=0.3, cfg_scale=3.0, self_cond="pred", noise_seed=1)
    out = ELFExtractor(enc, elf, obs)(torch.tensor([5, 7, 9, 5, 1]), index=2)
    su = importlib.import_module("utils.sampling_utils")  # upstream, on sys.path via import_elf
    cfg = SimpleNamespace(t_eps=0.05, self_cond_prob=0.5, num_self_cond_cfg_tokens=4)
    with torch.no_grad():
        z_t = _noisy_latent(enc, torch.tensor([5, 7, 9, 5, 1]), obs, index=2)
        t = torch.full((1,), 0.3)
        # as upstream _generate_samples_single_batch does for unconditional generation
        step = dict(model=elf, z=z_t, t_batch=t, config=cfg, self_cond_cfg_scale=3.0,
                    cond_seq=torch.zeros_like(z_t), cond_seq_mask=torch.zeros(z_t.shape[:2]))
        _, x_first = su._forward_sample_self_cond(x_pred_prev=torch.zeros_like(z_t), **step)
        _, x_second = su._forward_sample_self_cond(x_pred_prev=x_first, **step)
        torch.testing.assert_close(elf.final_layer(out["hidden"][-1][None]), x_second)
        inp = torch.cat([z_t, x_first], -1)
        torch.testing.assert_close(out["hidden"][0], elf.text_proj(elf.self_cond_proj(inp))[0])


def test_elf_decoder_matches_upstream_decode(tiny_elf):
    """Decoder mode = the input of upstream _dlm_decode_batch's final step (t = 1, CFG token)."""
    _, elf = tiny_elf
    enc = tiny_t5()
    ids = torch.tensor([5, 7, 9, 5, 1])
    out = ELFExtractor(enc, elf, ELFObservation(mode="decoder", cfg_scale=3.0))(ids)
    with torch.no_grad():
        z = enc(input_ids=ids[None]).last_hidden_state / 0.2
        _, logits = elf(torch.cat([z, torch.zeros_like(z)], -1), torch.ones(1), deterministic=True,
                        self_cond_cfg_scale=torch.full((1,), 3.0), decoder_step_active=True)
        h = out["hidden"][-1][None]
        torch.testing.assert_close(F.gelu(h @ elf.proj_kernel + elf.proj_bias, approximate="tanh")
                                   @ elf.unembed_kernel + elf.unembed_bias, logits)


def test_elf_self_cond_pred_rejected_for_decoder(tiny_elf):
    _, elf = tiny_elf
    with pytest.raises(ValueError):
        ELFExtractor(tiny_t5(), elf, ELFObservation(mode="decoder", self_cond="pred"))


# ------------------------------------------------- conditional (translation) ELF

@pytest.fixture(scope="module")
def tiny_cond_elf():
    if not elf_available:
        pytest.skip("ELF source not prepared (python scripts/prepare_elf.py)")
    from tokid.models import import_elf

    module = import_elf(DEFAULT_ELF_SOURCE)
    torch.manual_seed(1)
    return module.ELF(text_encoder_dim=32, max_length=16, hidden_size=48, depth=3, num_heads=4,
                      bottleneck_dim=8, num_time_tokens=4, num_self_cond_cfg_tokens=4,
                      num_model_mode_tokens=4, vocab_size=100).eval()


PROMPT, TARGET = torch.tensor([5, 7, 9, 5, 1]), torch.tensor([11, 12, 13, 1])


def cond_extractor(elf, enc, **obs):
    from tokid.models import ELFCondExtractor

    return ELFCondExtractor(enc, elf, ELFObservation(**obs), max_length=16, max_input_length=8, pad_id=1)


def cond_inputs(ext, obs, index=0):
    """Upstream-style latent: x0 from the conditional encoder masks, prompt kept clean."""
    seq, enc_mask, m = ext.sequence(PROMPT, TARGET)
    x0 = encode_masked(ext.encoder, seq, enc_mask) / obs.latent_std
    gen = torch.Generator().manual_seed(obs.noise_seed * 1_000_003 + index)
    z = obs.t * x0 + (1 - obs.t) * torch.randn(x0.shape, generator=gen) * obs.noise_scale
    return x0, torch.where(m > 0, x0, z), m


def test_cond_sequence_matches_upstream_collate(tiny_cond_elf):
    from utils.encoder_utils import build_self_attn_cond_masks  # upstream
    import numpy as np

    ext = cond_extractor(tiny_cond_elf, tiny_t5())
    seq, enc_mask, m = ext.sequence(PROMPT, TARGET)
    assert seq[0].tolist() == [5, 7, 9, 5, 1, 11, 12, 13, 1] + [1] * 7
    pos = np.arange(16)[None]
    ref, _, cond = build_self_attn_cond_masks(pos < 5, pos < 9, xp=np)
    torch.testing.assert_close(enc_mask, torch.from_numpy(ref))
    torch.testing.assert_close(m[..., 0], torch.from_numpy(cond))
    with torch.no_grad():  # prompt rows of the latent = T5 on the prompt alone
        x = encode_masked(ext.encoder, seq, enc_mask)[0, :5]
        torch.testing.assert_close(x, ext.encoder(input_ids=PROMPT[None]).last_hidden_state[0])
    with pytest.raises(ValueError):
        ext.sequence(torch.arange(9), TARGET)  # prompt longer than max_input_length


def test_cond_first_step_t0_zero_self_cond(tiny_cond_elf):
    """t = 0, zero self-cond. = upstream's first sampling step: pure noise + clean prompt."""
    enc = tiny_t5()
    ext = cond_extractor(tiny_cond_elf, enc, t=0.0, noise_seed=2)
    out = ext(PROMPT, index=3, target_ids=TARGET)
    assert len(out["hidden"]) == 4 and all(h.shape == (5, 48) for h in out["hidden"])
    torch.testing.assert_close(out["tokens"], enc.get_input_embeddings()(PROMPT))
    with torch.no_grad():
        x0, z, m = cond_inputs(ext, ext.obs, index=3)
        sc = m * x0  # generation_utils: x_pred = restore_cond(zeros, cond_seq, cond_seq_mask)
        inp = torch.cat([z, sc], -1)
        torch.testing.assert_close(out["hidden"][0], tiny_cond_elf.text_proj(tiny_cond_elf.self_cond_proj(inp))[0, :5])
        ref, _ = tiny_cond_elf(inp, torch.zeros(1), self_cond_cfg_scale=torch.ones(1))
        torch.testing.assert_close(tiny_cond_elf.final_layer(out["hidden"][-1][None]), ref[:, :5])


def test_cond_self_cond_pred_matches_upstream_ode_step(tiny_cond_elf):
    """scpred + guidance 2: self-cond. = x_pred of upstream _ode_step (CFG 2, SC-CFG 1)."""
    from types import SimpleNamespace
    import importlib

    su = importlib.import_module("utils.sampling_utils")
    ext = cond_extractor(tiny_cond_elf, tiny_t5(), t=0.3, self_cond="pred", guidance=2.0, noise_seed=1)
    out = ext(PROMPT, index=0, target_ids=TARGET)
    cfg = SimpleNamespace(t_eps=0.05, self_cond_prob=0.5, num_self_cond_cfg_tokens=4)
    with torch.no_grad():
        x0, z, m = cond_inputs(ext, ext.obs)
        _, x_pred = su._ode_step(model=tiny_cond_elf, z=z, t=0.3, t_next=0.4, x_pred_prev=m * x0, config=cfg,
                                 cfg_scale=2.0, self_cond_cfg_scale=1.0, cond_seq=x0, cond_seq_mask=m[..., 0])
        torch.testing.assert_close(x_pred[:, :5], x0[:, :5])  # prompt restored
        inp = torch.cat([z, x_pred], -1)
        torch.testing.assert_close(out["hidden"][0], tiny_cond_elf.text_proj(tiny_cond_elf.self_cond_proj(inp))[0, :5])
        ref, _ = tiny_cond_elf(inp, torch.full((1,), 0.3), self_cond_cfg_scale=torch.ones(1))
        torch.testing.assert_close(tiny_cond_elf.final_layer(out["hidden"][-1][None]), ref[:, :5])


def test_cond_decoder_matches_upstream_decode(tiny_cond_elf):
    from types import SimpleNamespace
    import importlib

    gu = importlib.import_module("utils.generation_utils")
    ext = cond_extractor(tiny_cond_elf, tiny_t5(), mode="decoder")
    out = ext(PROMPT, target_ids=TARGET)
    with torch.no_grad():
        x0, _, _ = cond_inputs(ext, ext.obs)
        cfg = SimpleNamespace(num_self_cond_cfg_tokens=4, self_cond_prob=0.5, use_bf16=False)
        ids = gu._dlm_decode_batch(x0, tiny_cond_elf, 1.0, cfg, 1.0)
        h = out["hidden"][-1][None]
        logits = (F.gelu(h @ tiny_cond_elf.proj_kernel + tiny_cond_elf.proj_bias, approximate="tanh")
                  @ tiny_cond_elf.unembed_kernel + tiny_cond_elf.unembed_bias)
        assert torch.equal(logits.argmax(-1), ids[:, :5])


@pytest.mark.parametrize("obs", [dict(guidance=2.0), dict(guidance=2.0, self_cond="pred", conditional=False)])
def test_guidance_needs_conditional_self_cond_pred(obs):
    from tokid.models import check_observation

    conditional = obs.pop("conditional", True)
    with pytest.raises(ValueError):
        check_observation(ELFObservation(**obs), conditional)


def test_encode_masked_matches_padding_and_3d_masks():
    enc = tiny_t5()
    ids = torch.tensor([[5, 7, 9, 5, 1, 1, 1, 1]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    # every row sees the same columns -> identical to the 2D padding mask (any transformers version)
    rows = valid[:, None, :].expand(1, 8, 8).float()
    native_2d = enc(input_ids=ids, attention_mask=valid).last_hidden_state
    torch.testing.assert_close(encode_masked(enc, ids, rows), native_2d, rtol=0, atol=0)
    # a per-row mask (prompt rows see only the prompt): older transformers take it natively
    is_cond = torch.arange(8) < 3
    m = ((is_cond[:, None] & is_cond[None, :]) | (~is_cond[:, None] & valid[0].bool()[None, :]))[None].float()
    try:
        native_3d = enc(input_ids=ids, attention_mask=m).last_hidden_state
    except RuntimeError:  # transformers >= 4.48: 2D masks only
        native_3d = None
    out = encode_masked(enc, ids, m)
    if native_3d is not None:
        torch.testing.assert_close(out, native_3d, rtol=0, atol=0)
    # prompt rows depend only on the prompt
    torch.testing.assert_close(out[0, :3], enc(input_ids=ids[:, :3]).last_hidden_state[0], rtol=1e-5, atol=1e-5)
