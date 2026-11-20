"""End-to-end smoke test on synthetic features: experiments + plots (no downloads)."""

import json

import torch

from tokid import experiments, plots


def fake_features(path, model="ELF-B", tag="elf-b", n_sent=40, layers=4, d=24, d_tok=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(5, 12, (n_sent,), generator=g)
    n = int(lengths.sum())
    token_ids = torch.randint(0, 30, (n,), generator=g)
    table = torch.randn(30, d_tok, generator=g)
    tokens = table[token_ids]
    proj = torch.randn(d_tok, d, generator=g)
    hidden = [tokens @ proj if d != d_tok else tokens]
    for _ in range(layers):
        hidden.append(hidden[-1] + 0.3 * torch.randn(n, d, generator=g))
    sent = torch.arange(n_sent).repeat_interleave(lengths)
    offsets = torch.zeros(n_sent + 1, dtype=torch.long)
    offsets[1:] = lengths.cumsum(0)
    pos = torch.cat([torch.arange(int(l)) for l in lengths])
    torch.save({"token_ids": token_ids, "sentence_ids": sent, "positions": pos, "offsets": offsets,
                "tokens": tokens, "hidden": torch.stack(hidden), "texts": ["x"] * n_sent,
                "meta": {"model": model, "tag": tag, "dataset": "mrpc", "num_sentences": n_sent,
                         "num_tokens": n, "min_len": int(lengths.min()), "max_len": int(lengths.max()),
                         "mean_len": float(lengths.float().mean())}}, path)


FAST = ["--folds", "2", "--max-epochs", "4", "--patience", "2", "--device", "cpu", "--lr", "1e-3",
        "--mlp-hidden", "8"]


def test_full_pipeline(tmp_path):
    elf, t5 = tmp_path / "elf-b_mrpc.pt", tmp_path / "t5-small_mrpc.pt"
    fake_features(elf)
    fake_features(t5, model="google-t5/t5-small", tag="t5-small", d=16)
    res_elf, res_t5 = tmp_path / "results/elf-b_mrpc", tmp_path / "results/t5-small_mrpc"
    for feats, res in ((elf, res_elf), (t5, res_t5)):
        for target in ("token", "input"):
            experiments.main(["--features", str(feats), "--experiment", "identity", "--target", target,
                              "--out", str(res / f"identity_{target}.json"), *FAST])
        experiments.main(["--features", str(feats), "--experiment", "hidden2hidden",
                          "--out", str(res / "hidden2hidden.json"), *FAST])
        experiments.main(["--features", str(feats), "--experiment", "neighbours", "--offsets", "-1", "2",
                          "--probes", "linear", "--out", str(res / "neighbours_token.json"), *FAST])

    r = json.loads((res_elf / "identity_token.json").read_text())
    assert "naive_cos" not in r["configs"]          # 24-d hidden vs 16-d tokens: no naive baseline
    assert len(r["configs"]["mlp_cos"]["test_mean"]) == 5
    assert len(r["configs"]["linear_l2"]["cross_mean"]) == 5
    r = json.loads((res_t5 / "identity_token.json").read_text())
    assert r["configs"]["naive_cos"]["test_mean"][0] == 1.0  # layer 0 == token embeddings
    r = json.loads((res_elf / "hidden2hidden.json").read_text())
    assert r["layers"] == [1, 2, 3, 4] and "naive_l2" in r["configs"]
    # LR search path
    experiments.main(["--features", str(t5), "--experiment", "identity", "--probes", "linear", "--losses", "cos",
                      "--lr-search", "--lr-grid", "1e-2", "1e-3", "--out", str(tmp_path / "lr.json"), *FAST])
    assert json.loads((tmp_path / "lr.json").read_text())["lr"]["linear_cos"] in (1e-2, 1e-3)

    figs = tmp_path / "figures"
    plots.main(["--results", str(res_elf), "--out", str(figs / "elf")])
    plots.main(["--results", str(res_t5), "--out", str(figs / "t5")])
    plots.main(["--pipeline", str(res_t5), str(res_elf), "--out", str(figs / "pipe")])
    for f in ("identity_token/fig2a_identifiability.png", "identity_token/fig2b_crosslayer_linear_cos.png",
              "identity_token/fig7_generalization_l2.png", "hidden2hidden/fig11_hidden2hidden.png",
              "neighbours_token/fig14_neighbours_linear_cos.png", "summary.md"):
        assert (figs / "elf" / f).is_file(), f
    assert (figs / "pipe/pipeline_identifiability.png").is_file()


def test_resume_after_interruption(tmp_path, monkeypatch):
    feats = tmp_path / "t5-small_mrpc.pt"
    fake_features(feats, model="google-t5/t5-small", tag="t5-small", d=16)
    args = ["--features", str(feats), "--experiment", "neighbours", "--offsets", "-1", "1", *FAST]
    ref_out, out = tmp_path / "ref.json", tmp_path / "res.json"
    experiments.main([*args, "--out", str(ref_out)])

    real_fit, calls = experiments.fit, {"n": 0}

    def crashing_fit(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt("simulated time limit")
        return real_fit(*a, **kw)

    monkeypatch.setattr(experiments, "fit", crashing_fit)
    try:
        experiments.main([*args, "--out", str(out)])
    except KeyboardInterrupt:
        pass
    partial = tmp_path / "res.json.partial"
    assert partial.exists() and not out.exists()
    assert len(json.loads(partial.read_text())["units"]) >= 4

    calls["n"] = -10_000  # no more crashes; count trainings of the resumed run
    experiments.main([*args, "--out", str(out)])
    total = 2 * 2 * 4  # offsets x folds x (2 probes x 2 losses)
    assert calls["n"] + 10_000 == total - 4  # the 4 finished trainings were not redone
    assert not partial.exists()
    ref, res = json.loads(ref_out.read_text()), json.loads(out.read_text())
    for k in ("-1", "1"):
        for cfg in ref["offsets"][k]["configs"]:
            assert ref["offsets"][k]["configs"][cfg]["test"] == res["offsets"][k]["configs"][cfg]["test"]

    # A checkpoint written with other settings is refused, not mixed in.
    partial.write_text(json.dumps({"settings": {"other": 1}, "units": {}}))
    out.unlink()
    import pytest
    with pytest.raises(SystemExit):
        experiments.main([*args, "--out", str(out)])
