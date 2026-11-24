"""ELF variant tags (no downloads)."""

import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from tokid.extract import ELF_DEFAULTS, main, model_tag, parse_elf_tag

ROOT = Path(__file__).resolve().parents[1]


def elf_args(**kw):
    return Namespace(model="elf", **{**ELF_DEFAULTS, **kw})


@pytest.mark.parametrize("kw,tag", [
    ({}, "elf-b"),
    ({"elf_t": 0.3}, "elf-b_t0.3"),
    ({"elf_mode": "decoder", "elf_cfg_scale": 3.0}, "elf-b_decoder_cfg3"),
    ({"elf_t": 0.1, "elf_cfg_scale": 3.0, "elf_self_cond": "pred"}, "elf-b_t0.1_cfg3_scpred"),
    ({"elf_t": 0.05, "elf_cfg_scale": 1.5}, "elf-b_t0.05_cfg1.5"),
    ({"elf_weights": "de-en", "elf_t": 0.0}, "elf-b-de-en_t0"),
    ({"elf_weights": "de-en", "elf_t": 0.3, "elf_guidance": 2.0, "elf_self_cond": "pred"},
     "elf-b-de-en_t0.3_g2_scpred"),
    ({"elf_weights": "de-en", "elf_mode": "decoder"}, "elf-b-de-en_decoder"),
])
def test_tag_round_trip(kw, tag):
    assert model_tag(elf_args(**kw)) == tag
    assert parse_elf_tag(tag) == {**ELF_DEFAULTS, **kw}


@pytest.mark.parametrize("bad", ["t5-small", "elf-b_x3", "elf-b_cfg3_t0.3", "elf-b-de", "elf-b-de-en_scpred_g2"])
def test_bad_variants_rejected(bad):
    with pytest.raises(SystemExit):
        main(["--model", "elf", "--variant", bad])


def test_variant_conflicts_with_explicit_flags():
    with pytest.raises(SystemExit):
        main(["--model", "elf", "--variant", "elf-b_decoder_cfg3", "--elf-t", "0.5"])


@pytest.mark.parametrize("var,count", [("ELF_INFERENCE", 6), ("ELF_DEEN_INFERENCE", 7)])
def test_inference_variants_are_canonical(var, count):
    names = subprocess.run(["bash", "-c", f"source scripts/models.sh; echo ${var}"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    assert len(names) == count
    for name in names:
        assert model_tag(elf_args(**parse_elf_tag(name))) == name


@pytest.mark.parametrize("argv", [
    ["--variant", "elf-b_decoder_scpred"],
    ["--elf-mode", "decoder", "--elf-self-cond", "pred"],
    ["--elf-t", "1.5"],
    ["--variant", "elf-b_t0.3_g2_scpred"],                       # guidance needs the conditional checkpoint
    ["--variant", "elf-b-de-en_t0.3_g2", "--dataset", "wmt14"],  # ... and self-cond. pred
    ["--variant", "elf-b-de-en_t0"],                             # de-en needs wmt14 (default mrpc)
    ["--variant", "elf-b_t0", "--dataset", "wmt14"],             # owt is not conditional
    ["--variant", "elf-b-de-en_t0", "--dataset", "wmt14", "--no-eos"],
])
def test_invalid_elf_settings_fail_before_loading(argv):
    with pytest.raises(SystemExit):
        main(["--model", "elf", *argv])
