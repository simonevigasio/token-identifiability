"""Generality splits of the probe g_l (no downloads)."""

import json

import numpy as np
import torch

from tokid import experiments, plots
from tokid.splits import domain_splits, unseen_token_splits
from test_pipeline import FAST, fake_features


def test_domain_splits_leave_one_domain_out():
    doms = ["a"] * 5 + ["b"] * 7 + ["c"] * 4
    folds = domain_splits(doms, ["a", "b", "c"], seed=1)
    assert [f["held_out"] for f in folds] == ["a", "b", "c"]
    for f in folds:
        test_doms = {doms[i] for i in f["test"]}
        train_doms = {doms[i] for i in np.concatenate([f["train"], f["val"]])}
        assert test_doms == {f["held_out"]} and f["held_out"] not in train_doms
        assert len(f["train"]) + len(f["val"]) + len(f["test"]) == len(doms)


def test_unseen_token_splits_hold_out_types():
    rng = np.random.default_rng(0)
    token_ids = rng.integers(0, 50, 2000)
    folds = unseen_token_splits(100, token_ids, num_folds=3, seed=0)
    for f in folds:
        share = np.isin(token_ids, f["test_types"]).mean()
        assert 0.15 <= share < 0.3  # covers >= 15% of occurrences (whole types)
    assert not np.array_equal(folds[0]["test_types"], folds[1]["test_types"])


def test_split_indices_keep_test_types_out_of_training():
    task = experiments.Task([0], torch.zeros(1, 6, 2), torch.zeros(1, 6, 2), torch.tensor([0, 0, 0, 1, 1, 1]),
                            torch.zeros(1, 6), torch.zeros(1, 6, 2), torch.tensor([0, 0, 0, 1, 1, 1]),
                            torch.zeros(1, 6), query_type=torch.tensor([7, 8, 9, 7, 8, 9]))
    idx = experiments.split_indices(task, {"train": [0], "val": [0], "test": [1], "test_types": [8]})
    assert idx["train"].tolist() == [0, 2] and idx["val"].tolist() == [0, 2] and idx["test"].tolist() == [4]


def with_domains(path, names=("x", "y", "z")):
    feats = torch.load(path, weights_only=False)
    n = feats["meta"]["num_sentences"]
    feats["domains"] = [names[i % len(names)] for i in range(n)]
    feats["meta"]["domain_names"] = list(names)
    torch.save(feats, path)


def test_generality_experiments_and_plots(tmp_path):
    feats = tmp_path / "elf-b-xsum_t0.3_docmix.pt"
    fake_features(feats, tag="elf-b-xsum_t0.3", n_sent=30)
    with_domains(feats)
    res = tmp_path / "res"
    for split, name in (("sentence", "identity_token"), ("domain", "identity_token_domain"),
                        ("unseen-token", "identity_token_unseen-token")):
        experiments.main(["--features", str(feats), "--experiment", "identity", "--split", split,
                          "--probes", "linear", "--out", str(res / f"{name}.json"), *FAST])
    r = json.loads((res / "identity_token_domain.json").read_text())
    assert r["split"] == "domain" and len(r["folds"]) == 3
    assert set(r["configs"]["linear_cos"]["test_by_domain_mean"]) == {"x", "y", "z"}
    r = json.loads((res / "identity_token.json").read_text())
    assert set(r["configs"]["linear_l2"]["test_by_domain_mean"]) <= {"x", "y", "z"}  # domains in the test sets
    plots.main(["--results", str(res), "--out", str(tmp_path / "fig")])
    for f in ("generalization/generalization_linear_cos.png",
              "generalization/generalization_by_domain_linear_cos.png", "summary.md"):
        assert (tmp_path / "fig" / f).is_file(), f
    assert "Test rate per domain" in (tmp_path / "fig/summary.md").read_text()
