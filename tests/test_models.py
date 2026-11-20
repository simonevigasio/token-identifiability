"""Extractor wiring on tiny randomly initialised models (no downloads)."""

import pytest
import torch
from transformers import T5Config, T5EncoderModel

from tokid.models import DEFAULT_ELF_SOURCE, ELFExtractor, ELFObservation, T5Extractor, VariableLengthRoPE


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
