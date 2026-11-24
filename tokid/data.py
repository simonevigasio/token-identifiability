"""GLUE development sets used by the paper (Sec. 4.1 and Appendix D), and the WMT14
De-En validation set of the conditional ELF-B translation checkpoint."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GlueSpec:
    config: str
    split: str
    fields: tuple
    default_max_examples: int | None
    t5_template: str  # T5's own GLUE text format (Raffel et al., 2020), optional


DATASETS = {
    # Sec. 4.1: all 408 MRPC dev examples, sentence pairs.
    "mrpc": GlueSpec("mrpc", "validation", ("sentence1", "sentence2"), None,
                     "mrpc sentence1: {sentence1} sentence2: {sentence2}"),
    # App. D.1.1: all 1043 CoLA dev examples.
    "cola": GlueSpec("cola", "validation", ("sentence",), None, "cola sentence: {sentence}"),
    # App. D.1.2: token identifiability uses the first 500 MNLI-matched dev examples.
    "mnli": GlueSpec("mnli", "validation_matched", ("premise", "hypothesis"), 500,
                     "mnli hypothesis: {hypothesis} premise: {premise}"),
}


@dataclass(frozen=True)
class TranslationSpec:
    """Pre-tokenized prompt/target pairs, exactly as ELF's conditional training/eval data.

    ``condition_input_ids`` (the prompt) is the T5 tokenization of ``input`` with </s>, so
    tokenizing the text again gives the same ids (checked for all 3000 WMT14 examples).
    Upstream truncates prompts to ``max_input_length`` (dropping </s>); we keep only
    prompts that fit, so every prompt row is a whole, untruncated sentence."""
    repo: str
    revision: str
    default_max_examples: int | None
    max_input_length: int


TRANSLATION = {
    # ELF-B-de-en-torch eval set (train_de-en_ELF-B.yml): German prompt -> English target.
    "wmt14": TranslationSpec("embedded-language-flows/wmt14_de-en_validation_t5",
                             "bcc92bda4b07c329354843724c9d9ea740f30685", 500, 64),
}
DATASET_NAMES = sorted([*DATASETS, *TRANSLATION])


def format_example(example: dict, spec: GlueSpec, pair_format: str) -> str:
    if pair_format == "concat":
        # Plain text: the same string is given to T5 and ELF (ELF-B was trained on raw OWT text).
        return " ".join(example[f].strip() for f in spec.fields)
    if pair_format == "t5":
        return spec.t5_template.format(**{f: example[f].strip() for f in spec.fields})
    raise ValueError(f"unknown pair_format {pair_format!r}")


def load_translation(name: str, max_examples: int | None = None, cache_dir: str | None = None):
    """-> (prompt texts, target token ids) of the first examples whose prompt fits.

    Reads the Arrow file directly (works with any `datasets` version and offline)."""
    import pyarrow as pa
    from huggingface_hub import snapshot_download

    spec = TRANSLATION[name]
    root = Path(snapshot_download(spec.repo, repo_type="dataset", revision=spec.revision, cache_dir=cache_dir))
    tables = []
    for f in sorted(root.glob("data-*.arrow")):
        with pa.memory_map(str(f)) as src:
            tables.append(pa.ipc.open_stream(src).read_all())
    table = pa.concat_tables(tables)
    texts, targets = [], []
    limit = spec.default_max_examples if max_examples is None else max_examples
    for text, cond, target in zip(table.column("input").to_pylist(), table.column("condition_input_ids").to_pylist(),
                                  table.column("input_ids").to_pylist()):
        if limit is not None and 0 < limit <= len(texts):
            break
        if len(cond) <= spec.max_input_length:
            texts.append(text)
            targets.append([int(i) for i in target])
    return texts, targets


def load_texts(name: str, max_examples: int | None = None, pair_format: str = "concat",
               cache_dir: str | None = None) -> list[str]:
    if name in TRANSLATION:
        return load_translation(name, max_examples, cache_dir)[0]
    from datasets import load_dataset

    spec = DATASETS[name]
    data = load_dataset("nyu-mll/glue", spec.config, split=spec.split, cache_dir=cache_dir)
    limit = spec.default_max_examples if max_examples is None else max_examples
    if limit is not None and limit > 0:
        data = data.select(range(min(limit, len(data))))
    return [format_example(ex, spec, pair_format) for ex in data]
