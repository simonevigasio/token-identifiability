"""GLUE development sets used by the paper (Sec. 4.1 and Appendix D)."""

from dataclasses import dataclass


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


def format_example(example: dict, spec: GlueSpec, pair_format: str) -> str:
    if pair_format == "concat":
        # Plain text: the same string is given to T5 and ELF (ELF-B was trained on raw OWT text).
        return " ".join(example[f].strip() for f in spec.fields)
    if pair_format == "t5":
        return spec.t5_template.format(**{f: example[f].strip() for f in spec.fields})
    raise ValueError(f"unknown pair_format {pair_format!r}")


def load_texts(name: str, max_examples: int | None = None, pair_format: str = "concat",
               cache_dir: str | None = None) -> list[str]:
    from datasets import load_dataset

    spec = DATASETS[name]
    data = load_dataset("nyu-mll/glue", spec.config, split=spec.split, cache_dir=cache_dir)
    limit = spec.default_max_examples if max_examples is None else max_examples
    if limit is not None and limit > 0:
        data = data.select(range(min(limit, len(data))))
    return [format_example(ex, spec, pair_format) for ex in data]
