"""Figures in the style of Brunner et al. (2020), Fig. 2a/2b, 7, 8-11, 14-17, plus tables.

    python -m tokid.plots --results results/elf-b_mrpc
    python -m tokid.plots --pipeline results/t5-small_mrpc results/elf-b_mrpc --out figures/pipeline_mrpc

Writes PNG + PDF figures and a ``summary.md`` with all mean rates.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical palette (fixed order) and neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
PROBE_COLOR = {"mlp": SERIES[0], "linear": SERIES[1], "naive": SERIES[2]}
METRIC_STYLE = {"cos": "-", "l2": "--"}
PROBE_LABEL = {"mlp": "MLP", "linear": "lin", "naive": "naive"}
METRIC_LABEL = {"cos": "cos", "l2": "L2"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 4,
    "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
})


def label(name):
    probe, metric = name.split("_")
    return f"$\\hat g^{{\\mathrm{{{PROBE_LABEL[probe]}}}}}_{{{METRIC_LABEL[metric]}}}$"


def style(name):
    probe, metric = name.split("_")
    return dict(color=PROBE_COLOR[probe], linestyle=METRIC_STYLE[metric], marker="o")


def load(path):
    return json.loads(Path(path).read_text())


def title_of(res):
    m = res["meta"]
    tgt = f", target={res['target']}" if res.get("target") else ""
    return f"{m['tag']} on {m['dataset'].upper()}{tgt}"


def trivial_layer0(res):
    """hidden[0] equals the target for T5 (token embeddings) and for target='input'."""
    return res.get("target") == "input" or not res["meta"]["model"].startswith("ELF")


def layer_axis(res):
    layers = np.asarray(res["layers"])
    keep = layers > 0 if trivial_layer0(res) else np.ones_like(layers, dtype=bool)
    return layers, keep


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def order(configs):
    pref = ["mlp_cos", "linear_cos", "naive_cos", "mlp_l2", "linear_l2", "naive_l2"]
    return [c for c in pref if c in configs]


def xticks(ax, layers, is_elf_input):
    ax.set_xticks(layers)
    ax.set_xticklabels(["in" if (l == 0 and is_elf_input) else str(l) for l in layers])


def fig_identity(res, out):
    layers, keep = layer_axis(res)
    elf0 = not trivial_layer0(res)
    cfgs = res["configs"]
    # Fig. 2a: same-layer identifiability.
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for name in order(cfgs):
        ax.plot(layers[keep], np.asarray(cfgs[name]["test_mean"])[keep], label=label(name), **style(name))
    xticks(ax, layers[keep], elf0)
    ax.set(xlabel="Layer", ylabel="Identifiability rate", title=f"Fig. 2a style - {title_of(res)}")
    ax.title.set_fontsize(9)
    ax.legend(ncol=2)
    save(fig, out, "fig2a_identifiability")

    # Fig. 2b / 8 / 9 / 10: g trained on layer l, tested on all layers.
    L = int(layers.max())
    trained = sorted({1, max(1, L // 2), max(1, L - 1), L} | ({0} if elf0 else set()))
    names = {"linear_cos": "fig2b_crosslayer_linear_cos", "linear_l2": "fig8_crosslayer_linear_l2",
             "mlp_l2": "fig9_crosslayer_mlp_l2", "mlp_cos": "fig10_crosslayer_mlp_cos"}
    for cfg, fname in names.items():
        if cfg not in cfgs or "cross_mean" not in cfgs[cfg]:
            continue
        cross = np.asarray(cfgs[cfg]["cross_mean"])
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for i, l in enumerate(trained):
            a = int(np.where(layers == l)[0][0])
            ax.plot(layers[keep], cross[a][keep], color=SERIES[i], marker="o",
                    label=("l=in" if (l == 0 and elf0) else f"l={l}"))
        xticks(ax, layers[keep], elf0)
        ax.set(xlabel="Test layer", ylabel="Identifiability rate",
               title=f"{label(cfg)} trained on layer l - {title_of(res)}")
        ax.title.set_fontsize(9)
        ax.legend(ncol=2)
        save(fig, out, fname)

    # Fig. 7: train vs test (L2).
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    drawn = False
    for probe in ("linear", "mlp"):
        cfg = cfgs.get(f"{probe}_l2")
        if cfg and "train_mean" in cfg:
            c = PROBE_COLOR[probe]
            ax.plot(layers[keep], np.asarray(cfg["train_mean"])[keep], color=c, linestyle=":", marker="o",
                    label=f"{PROBE_LABEL[probe]} train")
            ax.plot(layers[keep], np.asarray(cfg["test_mean"])[keep], color=c, linestyle="-", marker="o",
                    label=f"{PROBE_LABEL[probe]} test")
            drawn = True
    if drawn:
        xticks(ax, layers[keep], elf0)
        ax.set(xlabel="Layer", ylabel="1-NN recovery accuracy (L2)", title=f"Fig. 7 style - {title_of(res)}")
        ax.title.set_fontsize(9)
        ax.legend()
        save(fig, out, "fig7_generalization_l2")
    else:
        plt.close(fig)


def fig_hidden2hidden(res, out):
    layers = np.asarray(res["layers"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for name in order(res["configs"]):
        ax.plot(layers, res["configs"][name]["test_mean"], label=label(name), **style(name))
    ax.set_xticks(layers)
    ax.set(xlabel="Layer l (identify $e^{l-1}$ from $e^l$)", ylabel="Identifiability rate",
           title=f"Fig. 11 style - {title_of(res)}")
    ax.title.set_fontsize(9)
    ax.legend(ncol=2)
    save(fig, out, "fig11_hidden2hidden")


def fig_neighbours(res, out):
    offsets = sorted(res["offsets"], key=lambda k: (int(k) > 0, abs(int(k))))
    first = res["offsets"][offsets[0]]
    layers, keep = layer_axis({**res, "layers": first["layers"]})
    elf0 = not trivial_layer0(res)
    fignum = {"linear_cos": 14, "mlp_cos": 15, "mlp_l2": 16, "linear_l2": 17}
    for cfg in order(first["configs"]):
        if cfg.startswith("naive"):
            continue
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for k in offsets:
            kk = int(k)
            ax.plot(layers[keep], np.asarray(res["offsets"][k]["configs"][cfg]["test_mean"])[keep],
                    color=SERIES[abs(kk) - 1 if abs(kk) <= 8 else 7], linestyle="-" if kk < 0 else "--",
                    marker="o", label=f"i{kk:+d}")
        xticks(ax, layers[keep], elf0)
        ax.set(xlabel="Layer", ylabel="Identifiability rate",
               title=f"Fig. {fignum.get(cfg, '')} style: recover $x_{{i+k}}$, {label(cfg)} - {title_of(res)}")
        ax.title.set_fontsize(8)
        ax.legend(ncol=2)
        save(fig, out, f"fig{fignum.get(cfg, 'x')}_neighbours_{cfg}")


def fig_pipeline(t5_res, elf_res, out):
    """Token identifiability along the whole ELF pipeline: tokens -> T5-small -> ELF-B."""
    t5_layers = np.asarray(t5_res["layers"])
    elf_layers = np.asarray(elf_res["layers"])
    t5_keep = t5_layers > 0
    n_t5 = int(t5_keep.sum())
    xs_t5 = np.arange(1, n_t5 + 1)
    xs_elf = np.arange(n_t5 + 1, n_t5 + 1 + len(elf_layers))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name in order(t5_res["configs"]):
        st = style(name)
        y_t5 = np.asarray(t5_res["configs"][name]["test_mean"])[t5_keep]
        ax.plot(xs_t5, y_t5, label=label(name), **st)
        if name in elf_res["configs"]:
            ax.plot(xs_elf, elf_res["configs"][name]["test_mean"], **st)
    ax.axvline(n_t5 + 0.5, color=AXIS, linewidth=1)
    ax.text(n_t5 / 2 + 0.5, 0.02, "T5-small encoder", transform=ax.get_xaxis_transform(), ha="center", color=INK2)
    ax.text(n_t5 + 1 + len(elf_layers) / 2, 0.02, "ELF-B", transform=ax.get_xaxis_transform(), ha="center",
            color=INK2)
    ax.set_xticks(np.concatenate([xs_t5, xs_elf]))
    ax.set_xticklabels([str(l) for l in t5_layers[t5_keep]] + ["in" if l == 0 else str(l) for l in elf_layers])
    ax.set(xlabel="Layer", ylabel="Identifiability rate",
           title=f"Token identifiability along tokens -> T5-small -> ELF-B ({elf_res['meta']['dataset'].upper()})")
    ax.title.set_fontsize(9)
    ax.legend(ncol=3)
    save(fig, out, "pipeline_identifiability")


def table(res, configs, layers, key="test_mean"):
    head = "| config | " + " | ".join(str(l) for l in layers) + " |"
    sep = "|---" * (len(layers) + 1) + "|"
    rows = [head, sep]
    for name in configs:
        vals = res["configs"][name][key]
        std = res["configs"][name].get(key.replace("mean", "std"))
        cells = [f"{v:.3f}" + (f" ± {s:.3f}" if std else "") for v, s in zip(vals, std or [None] * len(vals))]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def summary(results, out):
    lines = ["# Token identifiability results", "",
             "Mean ± std over folds of the test identifiability rate (fraction of tokens whose "
             "nearest same-sentence neighbour is the correct one). Column = layer "
             "(0 = input to the first block).", ""]
    for name, res in results.items():
        m = res["meta"]
        lines += [f"## {name}: {title_of(res)}", "",
                  f"{m['num_sentences']} sentences, {m['num_tokens']} tokens "
                  f"(len {m['min_len']}-{m['max_len']}, mean {m['mean_len']:.1f}); "
                  f"chance ≈ {1 / m['mean_len']:.3f}. Learning rates: {res.get('lr')}", ""]
        if "offsets" in res:
            for k, r in sorted(res["offsets"].items(), key=lambda kv: int(kv[0])):
                lines += [f"### offset k={int(k):+d} ({r['num_queries']} queries)", "",
                          table(r, order(r["configs"]), r["layers"]), ""]
        else:
            lines += [table(res, order(res["configs"]), res["layers"]), ""]
            if any("train_mean" in c for c in res["configs"].values()):
                tr = [c for c in order(res["configs"]) if "train_mean" in res["configs"][c]]
                lines += ["Train-set rates (generalization, Fig. 7):", "",
                          table(res, tr, res["layers"], "train_mean"), ""]
    (out / "summary.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, nargs="*", default=[], help="Result directories (or JSON files).")
    p.add_argument("--pipeline", type=Path, nargs=2, metavar=("T5_SMALL_DIR", "ELF_DIR"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.pipeline:
        t5_dir, elf_dir = args.pipeline
        out = args.out or Path("figures") / f"pipeline_{elf_dir.name.split('_')[-1]}"
        fig_pipeline(load(t5_dir / "identity_token.json"), load(elf_dir / "identity_token.json"), out)
        print(f"wrote {out}")

    for path in args.results:
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        out = args.out or Path("figures") / (path.name if path.is_dir() else path.parent.name)
        results = {}
        for f in files:
            res = load(f)
            results[f.stem] = res
            sub = out / f.stem
            if res["experiment"] == "identity":
                fig_identity(res, sub)
            elif res["experiment"] == "hidden2hidden":
                fig_hidden2hidden(res, sub)
            else:
                fig_neighbours(res, sub)
        if results:
            out.mkdir(parents=True, exist_ok=True)
            summary(results, out)
            print(f"wrote {out} ({len(results)} result files)")


if __name__ == "__main__":
    main()
