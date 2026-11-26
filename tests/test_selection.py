"""Model-selection protocol (tokid.selection) and query-position extraction, on
synthetic data (no downloads)."""

import json

import numpy as np
import pytest
import torch

from tokid import selection
from tokid.extract import extract_queries
from tokid.probes import BatchedLinear, BatchedMLP, BatchedProbe, info_nce_loss


def fake_prompts(n_docs=40, domains=("a", "b", "c", "d"), lengths=(8, 12, 16), vocab=40, q=5, seed=0):
    rng = np.random.default_rng(seed)
    prompts = []
    for i in range(n_docs):
        L = int(lengths[i % len(lengths)])
        ids = rng.integers(3, vocab, L - 1).tolist() + [1]
        queries = sorted(rng.choice(L - 1, min(q, L - 1), replace=False).tolist())
        prompts.append({"ids": ids, "domain": domains[i % len(domains)], "length": L, "queries": queries,
                        "text": "x"})
    return prompts


def fake_run(embedding, proj, layers=3, noise=0.3, seed=0):
    def run(ids, index):
        g = torch.Generator().manual_seed(seed * 1000 + index)
        h = [embedding[ids] @ proj]
        for _ in range(layers):
            h.append(h[-1] + noise * torch.randn(h[-1].shape, generator=g))
        return {"p": {"tokens": embedding[ids], "hidden": h}}
    return run


def write_features(tmp_path, d=24, d_tok=16, vocab=40, n_docs=40, layers=3):
    torch.manual_seed(0)
    embedding = torch.randn(vocab, d_tok)
    proj = torch.eye(d_tok) if d == d_tok else torch.randn(d_tok, d)
    out = tmp_path / "fake_multidoc.pt"

    def meta(label):
        return {"model": "fake", "tag": "fake", "dataset": "multidoc", "domain_names": ["a", "b", "c", "d"],
                "ood_domains": ["d"]}

    extract_queries(fake_run(embedding, proj, layers), ["p"], fake_prompts(n_docs, vocab=vocab), {"p": out},
                    embedding, meta, log_every=7)
    return out


def test_extract_queries_resumes_identically(tmp_path):
    embedding, proj = torch.randn(40, 16), torch.randn(16, 24)
    prompts = fake_prompts()
    ref = tmp_path / "ref" / "x_multidoc.pt"
    ref.parent.mkdir()
    extract_queries(fake_run(embedding, proj), ["p"], prompts, {"p": ref}, embedding, lambda l: {}, log_every=5)

    out = tmp_path / "x_multidoc.pt"
    run, calls = fake_run(embedding, proj), {"n": 0}

    def crashing(ids, index):
        calls["n"] += 1
        if index == 23:
            raise KeyboardInterrupt("time limit")
        return run(ids, index)

    with pytest.raises(KeyboardInterrupt):
        extract_queries(crashing, ["p"], prompts, {"p": out}, embedding, lambda l: {}, log_every=5)
    assert not out.exists() and (tmp_path / "x_multidoc.progress.json").exists()
    calls["n"] = 0
    extract_queries(lambda i, k: (calls.__setitem__("n", calls["n"] + 1), run(i, k))[1], ["p"], prompts, {"p": out},
                    embedding, lambda l: {}, log_every=5)
    assert calls["n"] == len(prompts) - 20  # resumed at the last checkpoint (prompt 20)
    a = np.load(ref.with_name("x_multidoc.hidden.npy"))
    b = np.load(out.with_name("x_multidoc.hidden.npy"))
    np.testing.assert_array_equal(a, b)
    f = torch.load(out, weights_only=False)
    assert f["format"] == "queries" and f["query_index"].numel() == a.shape[1]
    # the stored rows are the hidden states of the query positions
    i = 3
    h = run(torch.tensor(prompts[i]["ids"]), i)["p"]["hidden"]
    rows = slice(int(f["query_offsets"][i]), int(f["query_offsets"][i + 1]))
    np.testing.assert_allclose(b[2, rows], h[2][prompts[i]["queries"]].half().numpy())


def test_protocol_splits_are_disjoint_and_stratified():
    domains = [d for d in "abcde" for _ in range(40)]
    lengths = [64, 128, 256, 512] * 50
    s = selection.protocol_splits(domains, lengths, ood=["e"], seed=3)
    test_id, test_ood, dev = set(s["test_id"]), set(s["test_ood"]), set(s["dev"])
    assert not (test_id & dev) and not (test_ood & dev) and not (test_id & test_ood)
    assert len(test_id | test_ood | dev) == 200
    assert {domains[i] for i in test_ood} == {"e"}
    for d in "abcd":  # 20% of each source, stratified by length
        assert sum(domains[i] == d for i in test_id) == 8
        assert sorted(lengths[i] for i in test_id if domains[i] == d) == [64, 64, 128, 128, 256, 256, 512, 512]
    assert set(s["search"]["train"]) | set(s["search"]["val"]) == dev
    vals = [set(f["val"]) for f in s["cv"]]
    assert set().union(*vals) == dev and sum(map(len, vals)) == len(dev)
    for f in s["cv"]:
        assert not set(f["train"]) & set(f["val"])


def test_batched_probe_matches_paper_probes():
    x = torch.randn(3, 7, 10)
    lin = BatchedLinear(3, 10, 6, torch.Generator().manual_seed(1))
    p = BatchedProbe(3, 10, 6, hidden=(), bias=False, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(p(x), lin(x))
    mlp = BatchedMLP(3, 10, 6, hidden=12, generator=torch.Generator().manual_seed(2))
    p = BatchedProbe(3, 10, 6, hidden=(12,), bias=True, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(p(x), mlp(x))


def test_info_nce_prefers_true_token():
    cand = torch.nn.functional.normalize(torch.randn(4, 5, 8), dim=-1)
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[0, 4] = False
    slot = torch.tensor([0, 1, 2, 3])
    good = cand[torch.arange(4), slot][None]
    bad = cand[torch.arange(4), (slot + 1) % 4][None]
    assert info_nce_loss(good, cand, mask, slot, 0.1) < info_nce_loss(bad, cand, mask, slot, 0.1)


def test_identity_probe_identifies_everything(tmp_path):
    path = write_features(tmp_path, d=16, d_tok=16)
    data = selection.QueryData(selection.load_query_features(path, torch.device("cpu")), torch.device("cpu"))
    ok = data.correct(selection.Identity(), list(range(data.num_docs)), "cos")
    assert ok[0].all()  # layer 0 = the token embeddings themselves


TINY = ["--n-random", "3", "--rungs", "1", "2", "--finalists", "2", "--cv-folds", "2", "--max-epochs", "3",
        "--patience", "1", "--seeds", "2", "--paper-max-epochs", "2", "--bootstrap", "20", "--device", "cpu"]


def test_selection_end_to_end(tmp_path):
    path = write_features(tmp_path)
    out = tmp_path / "selection.json"
    selection.main(["--features", str(path), "--out", str(out), *TINY])
    r = json.loads(out.read_text())
    G = 4
    assert len(r["chosen"]) == G and set(r["chosen"]) <= set(r["finalists"])
    assert len(r["rungs"]) == 2 and len(r["rungs"][0]) == 3 + 4 and len(r["finalists"]) == 2
    dev = set(r["splits"]["dev"])
    assert not dev & set(r["splits"]["test_id"]) and not dev & set(r["splits"]["test_ood"])
    sel = r["test"]["selected"]
    assert set(sel) == {"id", "ood"} and set(sel["ood"]["by_source"]) == {"d"}
    assert len(sel["id"]["all"]["rate"]) == G
    assert all(lo <= rate + 1e-9 and rate <= hi + 1e-9
               for rate, lo, hi in zip(sel["id"]["all"]["rate"], sel["id"]["all"]["lo"], sel["id"]["all"]["hi"]))
    assert {"paper_linear_cos", "paper_mlp_l2"} <= set(r["test"])
    assert not (tmp_path / "selection.json.partial").exists()

    from tokid import plots
    plots.main(["--results", str(tmp_path), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "selection" / "selection_layers.png").is_file()
    assert "Model selection" in (tmp_path / "fig" / "summary.md").read_text()


def test_trajectory_points_match_single_runs():
    from tokid.models import DEFAULT_ELF_SOURCE, ELFObservation, ELFTrajectoryExtractor, import_elf
    from transformers import T5Config, T5EncoderModel

    if not (DEFAULT_ELF_SOURCE / "src/modules/model.py").is_file():
        pytest.skip("ELF source not prepared (python scripts/prepare_elf.py)")
    module = import_elf(DEFAULT_ELF_SOURCE)
    torch.manual_seed(1)
    elf = module.ELF(text_encoder_dim=32, max_length=16, hidden_size=48, depth=3, num_heads=4, bottleneck_dim=8,
                     num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4, vocab_size=100).eval()
    torch.manual_seed(0)
    enc = T5EncoderModel(T5Config(vocab_size=100, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4)).eval()
    ids = torch.tensor([5, 7, 9, 5, 1])

    def ext(**obs):
        return ELFTrajectoryExtractor(enc, elf, ELFObservation(**obs), max_length=16, max_input_length=8,
                                      num_steps=8)

    multi = ext(mode="decoder").run_points(ids, index=2)
    assert list(multi) == ["t0", "t0.1", "t0.3", "t0.5", "t0.7", "t0.9", "decoder"]
    for label, obs in (("t0", dict(t=0.0)), ("t0.3", dict(t=0.3)), ("decoder", dict(mode="decoder"))):
        single = ext(**obs)(ids, index=2)
        for a, b in zip(multi[label]["hidden"], single["hidden"]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_selection_resumes_after_interruption(tmp_path, monkeypatch):
    path = write_features(tmp_path)
    ref, out = tmp_path / "ref.json", tmp_path / "res.json"
    selection.main(["--features", str(path), "--out", str(ref), *TINY])

    real, calls = selection.train, {"n": 0}

    def crashing(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 12:
            raise KeyboardInterrupt("simulated time limit")
        return real(*a, **kw)

    monkeypatch.setattr(selection, "train", crashing)
    with pytest.raises(KeyboardInterrupt):
        selection.main(["--features", str(path), "--out", str(out), *TINY])
    assert len(json.loads((tmp_path / "res.json.partial").read_text())["units"]) == 11
    calls["n"] = -10_000
    selection.main(["--features", str(path), "--out", str(out), *TINY])
    a, b = json.loads(ref.read_text()), json.loads(out.read_text())
    for k in ("chosen", "rungs", "cv", "test"):
        assert a[k] == b[k], k


def test_extract_queries_resumes_when_killed_while_finalizing(tmp_path, monkeypatch):
    embedding, proj = torch.randn(40, 16), torch.randn(16, 24)
    prompts = fake_prompts(n_docs=10)
    base = fake_run(embedding, proj)

    def run(ids, index):  # two observation points, like the ELF trajectory
        out = base(ids, index)["p"]
        return {"a": out, "b": {"tokens": out["tokens"], "hidden": [2 * h for h in out["hidden"]]}}

    ref = {l: tmp_path / "ref" / f"x_{l}_multidoc.pt" for l in "ab"}
    ref["a"].parent.mkdir()
    extract_queries(run, ["a", "b"], prompts, ref, embedding, lambda l: {}, log_every=4)

    out = {l: tmp_path / f"x_{l}_multidoc.pt" for l in "ab"}
    real_save, calls = torch.save, {"n": 0}

    def dying_save(*a, **kw):  # killed after "a" is finalized, before "b"
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("time limit")
        return real_save(*a, **kw)

    monkeypatch.setattr(torch, "save", dying_save)
    with pytest.raises(KeyboardInterrupt):
        extract_queries(run, ["a", "b"], prompts, out, embedding, lambda l: {}, log_every=4)
    monkeypatch.setattr(torch, "save", real_save)
    assert out["a"].exists() and not out["b"].exists()
    runs = {"n": 0}
    extract_queries(lambda i, k: (runs.__setitem__("n", runs["n"] + 1), run(i, k))[1], ["a", "b"], prompts, out,
                    embedding, lambda l: {}, log_every=4)
    assert runs["n"] == 0  # nothing recomputed, only finalized
    for l in "ab":
        np.testing.assert_array_equal(np.load(ref[l].with_name(f"x_{l}_multidoc.hidden.npy")),
                                      np.load(out[l].with_name(f"x_{l}_multidoc.hidden.npy")))
    assert not (tmp_path / "x_a_multidoc.progress.json").exists()


# ------------------------------------------------------------------ shared probes (--share)

def write_variants(tmp_path, names=("t0", "t1"), scales=(1.0, 2.0), layers=3, n_docs=40, vocab=40):
    """Feature files of the same prompts for several 'trajectory points' (layers + input)."""
    torch.manual_seed(0)
    embedding, proj = torch.randn(vocab, 16), torch.randn(16, 24)
    base = fake_run(embedding, proj, layers)

    def run(ids, index):
        out = base(ids, index)["p"]
        return {n: {"tokens": out["tokens"], "hidden": [s * h for h in out["hidden"]]} for n, s in zip(names, scales)}

    paths = {n: tmp_path / f"fake_{n}_multidoc.pt" for n in names}
    meta = lambda label: {"model": "fake", "tag": f"fake_{label}", "dataset": "multidoc",  # noqa: E731
                          "domain_names": ["a", "b", "c", "d"], "ood_domains": ["d"]}
    extract_queries(run, list(names), fake_prompts(n_docs, vocab=vocab), paths, embedding, meta, log_every=7)
    return [paths[n] for n in names]


def test_probe_cells():
    assert selection.probe_cells("none", 2, 3).tolist() == [[0], [1], [2], [3], [4], [5]]
    assert selection.probe_cells("time", 2, 3).tolist() == [[0, 3], [1, 4], [2, 5]]
    assert selection.probe_cells("layer", 2, 3).tolist() == [[0, 1, 2], [3, 4, 5]]
    assert selection.probe_cells("both", 2, 3).tolist() == [[0, 1, 2, 3, 4, 5]]


def test_cell_store_shards_match_one_tensor():
    x = torch.randn(5, 7, 3).half()
    one = selection.CellStore([(x, list(range(5)))], "cpu")
    two = selection.CellStore([(x[[0, 2, 4]], [0, 2, 4]), (x[[1, 3]], [1, 3])], "cpu")
    c, q = torch.randint(0, 5, (4, 6)), torch.randint(0, 7, (6,))
    torch.testing.assert_close(one.gather(c, q), x[c, q], rtol=0, atol=0)
    torch.testing.assert_close(two.gather(c, q), x[c, q], rtol=0, atol=0)


def test_load_cells_rejects_different_prompts(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = write_variants(tmp_path / "a", names=("t0",), scales=(1.0,))
    b = write_variants(tmp_path / "b", names=("t0",), scales=(1.0,), n_docs=30)
    with pytest.raises(ValueError, match="different prompts"):
        selection.load_cells([a[0], b[0]], torch.device("cpu"))


def test_shared_standardization_pools_the_cells(tmp_path):
    paths = write_variants(tmp_path)
    feats = selection.load_cells(paths, torch.device("cpu"))
    data = selection.QueryData(feats, torch.device("cpu"), selection.probe_cells("both", 2, 4))
    tr = data.queries(range(12))
    cfg = {"arch": "linear", "depth": 0, "width": 0, "dropout": 0.0, "standardize": True}
    model = selection.build_probe(cfg, data, tr, seed=0)
    pooled = torch.cat([data.x.gather(torch.full((1, tr.numel()), c), tr)[0] for c in range(8)]).double()
    torch.testing.assert_close(model.mean[0, 0], pooled.mean(0).float())


def test_shared_probes_are_independent_of_each_other(tmp_path):
    """share=layer: the probe of point t0 must not depend on the states of point t1."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = write_variants(tmp_path / "a", scales=(1.0, 2.0))
    b = write_variants(tmp_path / "b", scales=(1.0, -3.0))
    cfg = {"arch": "mlp", "depth": 1, "width": 8, "dropout": 0.1, "standardize": True, "loss": "infonce",
           "tau": 0.1, "lr": 1e-3, "weight_decay": 0.01, "batch_size": 16}
    weights = []
    for paths in (a, b):
        data = selection.QueryData(selection.load_cells(paths, torch.device("cpu")), torch.device("cpu"),
                                   selection.probe_cells("layer", 2, 4))
        model, info = selection.train(cfg, data, list(range(20)), list(range(20, 30)), seed=3, max_epochs=4,
                                      patience=2)
        weights.append(([w[0].clone() for w in model.weights], info["best_epoch"][0]))
    for wa, wb in zip(weights[0][0], weights[1][0]):
        torch.testing.assert_close(wa, wb, rtol=0, atol=0)
    assert weights[0][1] == weights[1][1]


@pytest.mark.parametrize("share,probes", [("time", 4), ("layer", 2), ("both", 1)])
def test_shared_selection_end_to_end(tmp_path, share, probes):
    from tokid import plots

    paths = write_variants(tmp_path)
    root = tmp_path / "results"
    for p in paths:  # dedicated per-point runs: the comparison in the summary
        selection.main(["--features", str(p), "--out", str(root / p.stem / "selection.json"), *TINY])
    out = root / f"fake_share-{share}_multidoc" / "selection.json"
    selection.main(["--features", *map(str, paths), "--share", share, "--out", str(out), *TINY])
    r = json.loads(out.read_text())
    assert r["meta"]["tag"] == f"fake_share-{share}" and r["share"] == share
    assert len(r["chosen"]) == probes and len(r["probe_cells"]) == probes
    assert len(r["test"]["selected"]["id"]["all"]["rate"]) == 8 and r["cells"][5] == [1, 1]
    ded = json.loads((root / paths[0].stem / "selection.json").read_text())
    assert r["splits"] == ded["splits"]  # same documents as the per-point runs
    plots.main(["--results", str(out.parent), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "selection" / "shared_layers.png").is_file()
    assert (tmp_path / "fig" / "selection" / "shared_delta.png").is_file()
    text = (tmp_path / "fig" / "summary.md").read_text()
    assert f"Shared probe ({share})" in text and "Shared - dedicated" in text
