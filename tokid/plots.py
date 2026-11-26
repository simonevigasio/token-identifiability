"""Figures in the style of Brunner et al. (2020), Fig. 2a/2b, 7, 8-11, 14-17, plus tables.

    python -m tokid.plots --results results/elf-b_mrpc
    python -m tokid.plots --pipeline results/t5-small_mrpc results/elf-b_mrpc --out figures/pipeline_mrpc

Writes PNG + PDF figures and a ``summary.md`` with all mean rates.
"""

import argparse
import json
import os
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
    split = f", split={res['split']}" if res.get("split", "sentence") != "sentence" else ""
    return f"{m['tag']} on {m['dataset'].upper()}{tgt}{split}"


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
           title=f"Token identifiability along tokens -> T5-small -> ELF-B "
                 f"({elf_res['meta'].get('tag', 'elf-b')}, {elf_res['meta']['dataset'].upper()})")
    ax.title.set_fontsize(9)
    ax.legend(ncol=3)
    save(fig, out, "pipeline_identifiability")


SPLIT_LABEL = {"sentence": "in-domain (sentence split)", "domain": "leave-one-domain-out",
               "unseen-token": "unseen word types"}


def fig_generalization(results, out):
    """How general g_l is: the same probe config under the three splits, per layer, and
    per held-out domain (in-domain vs leave-one-domain-out)."""
    by_split = {r.get("split", "sentence"): r for r in results.values()
                if r["experiment"] == "identity" and r.get("target") == "token"}
    if len(by_split) < 2:
        return
    ref = next(iter(by_split.values()))
    layers, keep = layer_axis(ref)
    elf0 = not trivial_layer0(ref)
    configs = [c for c in order(ref["configs"]) if not c.startswith("naive")]
    for cfg in configs:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for i, split in enumerate(s for s in SPLIT_LABEL if s in by_split):
            y = np.asarray(by_split[split]["configs"][cfg]["test_mean"])[keep]
            ax.plot(layers[keep], y, color=SERIES[i], marker="o", label=SPLIT_LABEL[split])
        xticks(ax, layers[keep], elf0)
        ax.set(xlabel="Layer", ylabel="Identifiability rate",
               title=f"Generality of g_l, {label(cfg)} - {ref['meta']['tag']}")
        ax.title.set_fontsize(9)
        ax.legend()
        save(fig, out, f"generalization_{cfg}")
        if {"sentence", "domain"} <= by_split.keys():
            ind = by_split["sentence"]["configs"][cfg].get("test_by_domain_mean", {})
            ood = by_split["domain"]["configs"][cfg].get("test_by_domain_mean", {})
            names = [d for d in ref["meta"].get("domain_names") or [] if d in ind and d in ood]
            if not names:
                continue
            fig, axes = plt.subplots(1, len(names), figsize=(2.2 * len(names), 3.0), sharey=True)
            for ax, d in zip(np.atleast_1d(axes), names):
                ax.plot(layers[keep], np.asarray(ind[d])[keep], color=SERIES[0], marker="o", label="in-domain")
                ax.plot(layers[keep], np.asarray(ood[d])[keep], color=SERIES[1], marker="o", label="held out")
                ax.set_title(d, fontsize=9)
                ax.set_xlabel("Layer")
                ax.set_xticks(layers[keep][::2])
            np.atleast_1d(axes)[0].set_ylabel("Identifiability rate")
            np.atleast_1d(axes)[0].legend()
            fig.suptitle(f"Per domain, {label(cfg)} - {ref['meta']['tag']}", fontsize=9)
            save(fig, out, f"generalization_by_domain_{cfg}")


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", [SURFACE, SERIES[0], "#0d2f5c"])


def fig_selection(res, out):
    """Model-selection protocol (tokid.selection): test identifiability per layer of the
    selected g_l (in-distribution and OOD test documents, 95% bootstrap CI) against the
    paper's probes; per-source and per-length breakdowns of the selected g_l."""
    layers = np.asarray(res["layers"])
    elf0 = not trivial_layer0({"meta": res["meta"], "configs": {}})
    tag = res["meta"]["tag"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    sel = res["test"]["selected"]
    for name, ls in (("id", "-"), ("ood", "--")):
        if name not in sel:
            continue
        a = sel[name]["all"]
        ax.fill_between(layers, a["lo"], a["hi"], color=SERIES[0], alpha=0.15, linewidth=0)
        ax.plot(layers, a["rate"], color=SERIES[0], linestyle=ls, marker="o",
                label=f"selected g_l, {'in-distribution' if name == 'id' else 'OOD'} test")
    for i, cfg in enumerate(("paper_mlp_cos", "paper_linear_cos")):
        if cfg in res["test"] and "id" in res["test"][cfg]:
            ax.plot(layers, res["test"][cfg]["id"]["all"]["rate"], color=SERIES[1 + i], linewidth=1.2, marker="o",
                    markersize=3, label=f"paper {cfg.split('_')[1]} (cos), in-distribution")
    xticks(ax, layers, elf0)
    ax.set(xlabel="Layer", ylabel="Test identifiability rate", ylim=(0, 1.02),
           title=f"Selected probes vs paper probes - {tag}")
    ax.title.set_fontsize(9)
    ax.legend(loc="lower left")
    save(fig, out, "selection_layers")

    rows, names = [], []
    for name in ("id", "ood"):
        for src, v in (sel.get(name, {}).get("by_source") or {}).items():
            if v is not None:
                rows.append(v["rate"])
                names.append(src + (" (OOD)" if name == "ood" else ""))
    if rows:
        fig, ax = plt.subplots(figsize=(0.45 * len(layers) + 2.4, 0.28 * len(rows) + 1.2))
        im = ax.imshow(np.asarray(rows), aspect="auto", cmap=_seq_cmap(), vmin=0, vmax=1)
        ax.set_yticks(range(len(names)), names)
        ax.set_xticks(range(len(layers)), ["in" if (l == 0 and elf0) else str(l) for l in layers])
        ax.grid(False)
        ax.set(xlabel="Layer", title=f"Selected g_l, test rate per source - {tag}")
        ax.title.set_fontsize(9)
        fig.colorbar(im, ax=ax, label="Identifiability rate", fraction=0.04)
        save(fig, out, "selection_by_source")

    by_len = sel.get("id", {}).get("by_length") or {}
    if by_len:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for i, (L, v) in enumerate(sorted(by_len.items(), key=lambda kv: int(kv[0]))):
            ax.plot(layers, v["rate"], color=SERIES[i], marker="o", label=f"{L} tokens")
        xticks(ax, layers, elf0)
        ax.set(xlabel="Layer", ylabel="Test identifiability rate", ylim=(0, 1.02),
               title=f"Selected g_l by prompt length (in-distribution) - {tag}")
        ax.title.set_fontsize(9)
        ax.legend(title="prompt length")
        save(fig, out, "selection_by_length")


SHARE_TEXT = {
    "none": "one probe per (point, layer)",
    "time": "time-independent: one probe per layer, shared by every point of the trajectory",
    "layer": "layer-independent: one probe per point, shared by every layer",
    "both": "time- and layer-independent: one probe for every (point, layer)",
}


def is_shared(res):
    return "variants" in res


def variant_names(tags):
    """Tags without their common prefix: elf-b-xsum_t0.1 -> t0.1."""
    prefix = os.path.commonprefix(list(tags))
    prefix = prefix[:prefix.rfind("_") + 1] if "_" in prefix else ""
    return [t[len(prefix):] or t for t in tags]


def load_baseline(res, root):
    """The dedicated per-layer runs of each point (results/<tag>_<dataset>/selection.json), if present."""
    out = {}
    for tag in res["variants"]:
        f = Path(root) / f"{tag}_{res['meta']['dataset']}" / "selection.json"
        if f.is_file():
            b = load(f)
            if b.get("splits") == res["splits"]:  # same documents, else not comparable
                out[tag] = b
    return out


def _grid(res, key, test="id", field="all"):
    """Per-cell values of res['test'][key][test][field] -> (V, L) arrays rate, lo, hi."""
    V, L = len(res["variants"]), res["layers_per_variant"]
    a = res["test"][key][test][field]
    return tuple(np.asarray(a[k], float).reshape(V, L) for k in ("rate", "lo", "hi"))


def _base_grid(res, base, test="id"):
    """(V, L) dedicated-probe rates (NaN where a point has no per-layer run)."""
    V, L = len(res["variants"]), res["layers_per_variant"]
    g = np.full((V, L), np.nan)
    for v, tag in enumerate(res["variants"]):
        if tag in base and test in base[tag]["test"]["selected"]:
            g[v] = base[tag]["test"]["selected"][test]["all"]["rate"]
    return g


def fig_shared(res, out, base):
    """Shared probes (tokid.selection --share): test rate per point and layer, against the
    dedicated per-layer probes of each point, and the difference shared - dedicated."""
    names = variant_names(res["variants"])
    L = res["layers_per_variant"]
    layers = np.arange(L)
    elf0 = not trivial_layer0({"meta": res["meta"], "configs": {}})
    tag, share = res["meta"]["tag"], res["share"]
    tests = [t for t in ("id", "ood") if t in res["test"]["selected"]]
    fig, axes = plt.subplots(1, len(tests), figsize=(5.2 * len(tests), 3.8), squeeze=False)
    lo_all = 1.0
    for ax, test in zip(axes[0], tests):
        rate, _, _ = _grid(res, "selected", test)
        ded = _base_grid(res, base, test)
        for v, name in enumerate(names):
            color = SERIES[v % len(SERIES)]
            ax.plot(layers, rate[v], color=color, marker="o", label=name)
            if not np.isnan(ded[v]).all():
                ax.plot(layers, ded[v], color=color, linestyle="--", linewidth=1, alpha=0.7)
        lo_all = min(lo_all, np.nanmin(rate), np.nanmin(ded) if not np.isnan(ded).all() else 1.0)
        xticks(ax, layers, elf0)
        ax.set(xlabel="Layer", ylabel="Test identifiability rate",
               title=f"{'In-distribution' if test == 'id' else 'OOD'} test")
        ax.title.set_fontsize(9)
    for ax in axes[0]:
        ax.set_ylim(max(0.0, lo_all - 0.01), 1.002)
    axes[0][0].legend(title="point (solid: shared, dashed: dedicated)", loc="lower left", ncol=2)
    fig.suptitle(f"Shared probe ({share}) - {tag}", fontsize=9)
    save(fig, out, "shared_layers")

    ded = _base_grid(res, base, "id")
    if not np.isnan(ded).all():
        rate, _, _ = _grid(res, "selected", "id")
        delta = 100 * (rate - ded)                                   # percentage points
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("div", [SERIES[1], SURFACE, SERIES[0]])
        m = max(float(np.nanmax(np.abs(delta))), 0.1)
        fig, ax = plt.subplots(figsize=(0.55 * L + 2.4, 0.42 * len(names) + 1.4))
        im = ax.imshow(delta, aspect="auto", cmap=cmap, vmin=-m, vmax=m)
        for v in range(len(names)):
            for l in range(L):
                if not np.isnan(delta[v, l]):
                    ax.text(l, v, f"{delta[v, l]:+.1f}", ha="center", va="center", fontsize=6.5, color=INK)
        ax.set_yticks(range(len(names)), names)
        ax.set_xticks(range(L), ["in" if (l == 0 and elf0) else str(l) for l in layers])
        ax.grid(False)
        ax.set(xlabel="Layer", ylabel="Point",
               title=f"Shared ({share}) - dedicated probes\nin-distribution test rate, percentage points")
        ax.title.set_fontsize(9)
        fig.colorbar(im, ax=ax, label="Difference (pp)", fraction=0.04)
        save(fig, out, "shared_delta")


def shared_summary(name, res, base):
    names = variant_names(res["variants"])
    L = res["layers_per_variant"]
    elf0 = not trivial_layer0({"meta": res["meta"], "configs": {}})
    cols = ["in" if (l == 0 and elf0) else str(l) for l in range(L)]
    head = "| point | " + " | ".join(cols) + " |"
    sep = "|---" * (L + 1) + "|"

    def grid_rows(rate, lo=None, hi=None, fmt="{:.3f}"):
        rows = []
        for v, n in enumerate(names):
            cells = []
            for l in range(L):
                if np.isnan(rate[v, l]):
                    cells.append("–")
                elif lo is None:
                    cells.append(fmt.format(rate[v, l]))
                else:
                    cells.append(f"{rate[v, l]:.3f} [{lo[v, l]:.3f}, {hi[v, l]:.3f}]")
            rows.append(f"| {n} | " + " | ".join(cells) + " |")
        return rows

    sp, share = res["splits"], res["share"]
    lines = [f"## {name}: Shared probe ({share}) - {res['meta']['tag']} on {res['meta']['dataset']}", "",
             f"{SHARE_TEXT[share]}. {len(res['probe_cells'])} probe(s), each trained on all (query, cell) pairs "
             f"of its {len(res['probe_cells'][0])} cell(s); points: {', '.join(names)}. Same protocol and "
             f"documents as the per-point runs: {len(sp['dev'])} dev / {len(sp['test_id'])} in-distribution "
             f"test / {len(sp['test_ood'])} OOD test prompts; {len(res['pool'])} configurations, halving "
             f"rounds {res['args']['rungs']} epochs, {len(res['finalists'])} finalists, "
             f"{res['args']['cv_folds']}-fold CV; refit with {res['args']['seeds']} seeds. "
             "Test rate [95% bootstrap CI over prompts]; row = point, column = layer.", "",
             "Selected configuration per probe (CV mean validation rate over its cells, optimistic):", "",
             "| probe | cells | config | CV val rate |", "|---|---|---|---|"]
    for p, (cells, c, v) in enumerate(zip(res["probe_cells"], res["chosen"], res["chosen_cv_rate"])):
        if share == "time":
            what = f"layer {cols[res['cells'][cells[0]][1]]}, every point"
        elif share == "layer":
            what = f"point {names[res['cells'][cells[0]][0]]}, every layer"
        elif share == "both":
            what = "every point and layer"
        else:
            vv, ll = res["cells"][cells[0]]
            what = f"point {names[vv]}, layer {cols[ll]}"
        lines.append(f"| {p} | {what} | {c}: {describe_config(res['pool'][c])} | {v:.3f} |")
    for test in ("id", "ood"):
        if test not in res["test"]["selected"]:
            continue
        label = "in-distribution" if test == "id" else "OOD"
        rate, lo, hi = _grid(res, "selected", test)
        lines += ["", f"Shared probe, {label} test:", "", head, sep, *grid_rows(rate, lo, hi)]
        ur, ul, uh = _grid(res, "selected", test, "unseen_types")
        lines += ["", f"Shared probe, {label} test, unseen token types:", "", head, sep, *grid_rows(ur, ul, uh)]
        ded = _base_grid(res, base, test)
        if not np.isnan(ded).all():
            lines += ["", f"Dedicated per-layer probes of each point (per-point runs), {label} test:", "", head,
                      sep, *grid_rows(ded)]
            lines += ["", f"Shared - dedicated, {label} test (percentage points):", "", head, sep,
                      *grid_rows(100 * (rate - ded), fmt="{:+.2f}")]
    for key in res["test"]:
        if key != "selected" and "id" in res["test"][key]:
            rate, _, _ = _grid(res, key, "id")
            lines += ["", f"{key} (paper procedure, shared the same way), in-distribution test:", "", head, sep,
                      *grid_rows(rate)]
    if not base:
        lines += ["", "No per-point selection.json found next to this run: no comparison with dedicated probes."]
    return lines + [""]


def describe_config(cfg):
    arch = cfg["arch"] if cfg["arch"] != "mlp" else f"mlp {cfg['depth']}x{cfg['width']}"
    extra = [f"lr {cfg['lr']:.1e}", f"wd {cfg['weight_decay']:g}", f"bs {cfg['batch_size']}"]
    if cfg["dropout"]:
        extra.append(f"dropout {cfg['dropout']:g}")
    if cfg["standardize"]:
        extra.append("std. input")
    loss = cfg["loss"] + (f" (tau {cfg['tau']:.3f})" if cfg["loss"] == "infonce" else "")
    return f"{arch}, {loss}, " + ", ".join(extra)


def selection_summary(name, res):
    layers = res["layers"]
    head = "| | " + " | ".join(str(l) for l in layers) + " |"
    sep = "|---" * (len(layers) + 1) + "|"

    def row(label, a):
        return f"| {label} | " + " | ".join(f"{r:.3f} [{lo:.3f}, {hi:.3f}]"
                                            for r, lo, hi in zip(a["rate"], a["lo"], a["hi"])) + " |"

    sp = res["splits"]
    lines = [f"## {name}: Model selection - {res['meta']['tag']} on {res['meta']['dataset']}", "",
             f"{len(sp['dev'])} dev / {len(sp['test_id'])} in-distribution test / {len(sp['test_ood'])} OOD test "
             f"prompts (OOD sources: {', '.join(res['meta'].get('ood_domains', []))}). Search: "
             f"{len(res['pool'])} configurations, halving rounds {res['args']['rungs']} epochs, "
             f"{len(res['finalists'])} finalists, {res['args']['cv_folds']}-fold CV on dev; refit with "
             f"{res['args']['seeds']} seeds. Test rate [95% bootstrap CI over prompts]; column = layer.", "",
             "Selected configuration per layer (CV mean validation rate is optimistic, it was maximised):", "",
             "| layer | config | CV val rate |", "|---|---|---|"]
    for l, (c, v) in enumerate(zip(res["chosen"], res["chosen_cv_rate"])):
        lines.append(f"| {layers[l]} | {c}: {describe_config(res['pool'][c])} | {v:.3f} |")
    lines += ["", "Test rates:", "", head, sep]
    for test in ("id", "ood"):
        if test in res["test"]["selected"]:
            lines.append(row(f"selected, {test}", res["test"]["selected"][test]["all"]))
            lines.append(row(f"selected, {test}, unseen token types",
                             res["test"]["selected"][test]["unseen_types"]))
    for key, val in res["test"].items():
        if key == "selected":
            continue
        for test in ("id", "ood"):
            if test in val:
                lines.append(row(f"{key}, {test}", val[test]["all"]))
    for test in ("id", "ood"):
        sel = res["test"]["selected"].get(test)
        if not sel:
            continue
        lines += ["", f"Selected g_l per source ({test}):", "", head, sep]
        lines += [row(src, v) for src, v in sel["by_source"].items() if v is not None]
        if test == "id":
            lines += ["", "Selected g_l per prompt length (id):", "", head, sep]
            lines += [row(f"{L} tokens", v) for L, v in sorted(sel["by_length"].items(), key=lambda kv: int(kv[0]))]
    return lines + [""]


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


def summary(results, out, baselines=None):
    lines = ["# Token identifiability results", "",
             "Mean ± std over folds of the test identifiability rate (fraction of tokens whose "
             "nearest same-sentence neighbour is the correct one). Column = layer "
             "(0 = input to the first block).", ""]
    for name, res in results.items():
        if res["experiment"] == "selection" and is_shared(res):
            lines += shared_summary(name, res, (baselines or {}).get(name, {}))
            continue
        if res["experiment"] == "selection":
            lines += selection_summary(name, res)
            continue
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
            for cfg in order(res["configs"]):
                by_dom = res["configs"][cfg].get("test_by_domain_mean")
                if by_dom:
                    hold = " (each domain held out of training)" if res.get("split") == "domain" else ""
                    lines += [f"Test rate per domain, {cfg}{hold}:", "",
                              table({"configs": {d: {"test_mean": v} for d, v in by_dom.items()}},
                                    list(by_dom), res["layers"]), ""]
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
        results, baselines = {}, {}
        for f in files:
            res = load(f)
            results[f.stem] = res
            sub = out / f.stem
            if res["experiment"] == "identity":
                fig_identity(res, sub)
            elif res["experiment"] == "hidden2hidden":
                fig_hidden2hidden(res, sub)
            elif res["experiment"] == "selection" and is_shared(res):
                baselines[f.stem] = load_baseline(res, f.parent.parent)  # results/<tag>_<dataset>/ next to it
                fig_shared(res, sub, baselines[f.stem])
            elif res["experiment"] == "selection":
                fig_selection(res, sub)
            else:
                fig_neighbours(res, sub)
        if results:
            out.mkdir(parents=True, exist_ok=True)
            summary(results, out, baselines)
            fig_generalization(results, out / "generalization")
            print(f"wrote {out} ({len(results)} result files)")


if __name__ == "__main__":
    main()
