"""GLUE development sets used by the paper (Sec. 4.1 and Appendix D), the WMT14 De-En
validation set of the conditional ELF-B translation checkpoint, and two prompt sets for the
ELF-B XSum (summarization) checkpoint: ``docmix`` (six domains, all prompts 256 tokens) and
``multidoc`` (17 sources, prompts of 64-512 tokens, the model-selection protocol of
tokid.selection)."""

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


@dataclass(frozen=True)
class DocMixSpec:
    """English document prompts from several domains, all cut to the same length.

    Every prompt is the first ``prompt_len - 1`` T5 tokens of a document plus </s>, so
    all domains have the same number of identification candidates per prompt. Documents
    shorter than that are skipped; the first ``docs_per_domain`` long-enough documents
    (in file order) of every domain are used.
    * xsum: BBC articles, XSum validation split as tokenized for ELF (the checkpoint's
      own prompt distribution).
    * the others: the Pile test split (uncopyrighted copy), by ``pile_set_name``. Books
      start at the first paragraph break after 20% of the text, past the front matter.
    """
    domains: tuple = ("xsum", "web", "wiki", "pubmed", "legal", "books")
    prompt_len: int = 256
    docs_per_domain: int = 64


DOCMIX = {"docmix": DocMixSpec()}
XSUM_DATA = ("embedded-language-flows/xsum_validation_t5", "b0b4f5de92434d308eceeb9e38fd3f8d7d2b6a9e")
PILE_DATA = ("monology/pile-uncopyrighted", "3be90335b66f24456a5d6659d9c8d208c0357119", "test.jsonl.zst")
PILE_DOMAINS = {"web": "Pile-CC", "wiki": "Wikipedia (en)", "pubmed": "PubMed Abstracts", "legal": "FreeLaw",
                "books": "Gutenberg (PG-19)"}


@dataclass(frozen=True)
class MultiDocSpec:
    """Documents from many sources, for training g_l on as varied a context as possible.

    * sources: XSum validation articles (the checkpoint's own prompt distribution) and
      16 subsets of the Pile test split (uncopyrighted copy); Ubuntu IRC has too few
      long documents and is left out.
    * ood: sources never used to train or select g_l (out-of-distribution test only):
      two close to the training sources (HackerNews, NIH grant abstracts) and one far
      (code).
    * Each document gets a prompt length L from ``lengths`` (first L - 1 T5 tokens +
      </s>), balanced per source: the buckets that still need documents and fit the
      document are drawn with probability proportional to their remaining need. Up to
      ``docs_per_source`` documents per source, in file order.
    * ``queries_per_doc`` positions (never the final </s>) are drawn per document; only
      their hidden states are stored, while every token of the prompt stays a candidate
      of the nearest-neighbour lookup. More documents with fewer tokens each give more
      varied context for the same memory.
    """
    sources: tuple = ("xsum", "web", "wiki", "stackexchange", "github", "uspto", "pmc", "legal", "pubmed",
                      "arxiv", "dm_math", "nih", "hackernews", "enron", "europarl", "books", "philpapers")
    ood: tuple = ("hackernews", "nih", "github")
    lengths: tuple = (64, 128, 256, 512)
    docs_per_source: int = 200
    queries_per_doc: int = 64
    min_docs: int = 40
    seed: int = 0


MULTIDOC = {"multidoc": MultiDocSpec()}
PILE_SETS = {**PILE_DOMAINS, "stackexchange": "StackExchange", "github": "Github", "uspto": "USPTO Backgrounds",
             "pmc": "PubMed Central", "arxiv": "ArXiv", "dm_math": "DM Mathematics", "nih": "NIH ExPorter",
             "hackernews": "HackerNews", "enron": "Enron Emails", "europarl": "EuroParl",
             "philpapers": "PhilPapers"}
DATASET_NAMES = sorted([*DATASETS, *TRANSLATION, *DOCMIX, *MULTIDOC])


def read_arrow_dir(root) -> "pyarrow.Table":
    import pyarrow as pa

    tables = []
    for f in sorted(Path(root).glob("data-*.arrow")):
        with pa.memory_map(str(f)) as src:
            tables.append(pa.ipc.open_stream(src).read_all())
    return pa.concat_tables(tables)


def iter_pile(wanted: dict, cache_dir=None):
    """Yield (our source name, text) for the Pile test documents of the sets in ``wanted``
    ({pile_set_name: name}), in file order. Books start at the first paragraph break after
    20% of the text, past the front matter."""
    import io
    import json

    import zstandard
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(PILE_DATA[0], PILE_DATA[2], repo_type="dataset", revision=PILE_DATA[1],
                           cache_dir=cache_dir)
    with open(path, "rb") as fh:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"):
            doc = json.loads(line)
            name = wanted.get(doc["meta"]["pile_set_name"])
            if name is None:
                continue
            text = doc["text"]
            if name == "books":
                start = text.find("\n\n", len(text) // 5)
                text = text[start:] if start >= 0 else ""
            yield name, text


def load_docmix(name: str = "docmix", tokenizer=None, cache_dir: str | None = None, docs_per_domain=None):
    """-> list of {"ids", "text", "domain"} prompts, grouped by domain in DocMixSpec order."""
    from huggingface_hub import snapshot_download

    spec = DOCMIX[name]
    per_domain = docs_per_domain or spec.docs_per_domain
    if tokenizer is None:
        from transformers import AutoTokenizer
        from tokid.models import T5_REVISIONS

        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", revision=T5_REVISIONS["google-t5/t5-small"],
                                                  cache_dir=cache_dir)
    keep, eos = spec.prompt_len - 1, tokenizer.eos_token_id
    found = {d: [] for d in spec.domains}

    def add(domain, ids):
        if len(found[domain]) < per_domain and len(ids) >= keep:
            ids = [int(i) for i in ids[:keep]] + [eos]
            found[domain].append({"ids": ids, "text": tokenizer.decode(ids), "domain": domain})

    if "xsum" in found:
        root = snapshot_download(XSUM_DATA[0], repo_type="dataset", revision=XSUM_DATA[1], cache_dir=cache_dir)
        for cond in read_arrow_dir(root).column("condition_input_ids").to_pylist():
            add("xsum", cond[:-1] if cond and cond[-1] == eos else cond)
            if len(found["xsum"]) == per_domain:
                break
    wanted = {PILE_DOMAINS[d]: d for d in spec.domains if d in PILE_DOMAINS}
    if wanted:
        for domain, text in iter_pile(wanted, cache_dir):
            if len(found[domain]) == per_domain:
                continue
            # 8000 characters are far more than 255 tokens; avoids tokenizing whole books
            add(domain, tokenizer(text[:8000].strip(), add_special_tokens=False)["input_ids"])
            if all(len(v) == per_domain for v in found.values()):
                break
    missing = {d: len(v) for d, v in found.items() if len(v) < per_domain}
    if missing:
        raise ValueError(f"not enough long documents for {missing} (wanted {per_domain} each)")
    return [p for d in spec.domains for p in found[d]]


def load_multidoc(name: str = "multidoc", tokenizer=None, cache_dir: str | None = None, spec=None):
    """-> list of prompts {"ids", "text", "domain", "length", "queries"}, grouped by source
    in MultiDocSpec order. ``queries``: the sorted positions whose hidden states are kept."""
    import numpy as np
    from huggingface_hub import snapshot_download

    spec = spec or MULTIDOC[name]
    if tokenizer is None:
        from transformers import AutoTokenizer
        from tokid.models import T5_REVISIONS

        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", revision=T5_REVISIONS["google-t5/t5-small"],
                                                  cache_dir=cache_dir)
    eos = tokenizer.eos_token_id
    per_bucket = spec.docs_per_source // len(spec.lengths)
    need = {s: {L: per_bucket for L in spec.lengths} for s in spec.sources}
    found = {s: [] for s in spec.sources}
    rng = {s: np.random.default_rng([spec.seed, k]) for k, s in enumerate(spec.sources)}

    def add(source, ids):
        """ids: document tokens without </s>; True if the source is full."""
        fits = [L for L in spec.lengths if need[source][L] > 0 and len(ids) >= L - 1]
        if fits:
            w = np.array([need[source][L] for L in fits], dtype=float)
            L = int(rng[source].choice(fits, p=w / w.sum()))
            need[source][L] -= 1
            found[source].append({"ids": [int(i) for i in ids[:L - 1]] + [eos], "domain": source, "length": L})
        return sum(need[source].values()) == 0

    if "xsum" in found:
        root = snapshot_download(XSUM_DATA[0], repo_type="dataset", revision=XSUM_DATA[1], cache_dir=cache_dir)
        for cond in read_arrow_dir(root).column("condition_input_ids").to_pylist():
            if add("xsum", cond[:-1] if cond and cond[-1] == eos else cond):
                break
    wanted = {PILE_SETS[s]: s for s in spec.sources if s in PILE_SETS}
    if wanted:
        max_len = max(spec.lengths)
        for source, text in iter_pile(wanted, cache_dir):
            if sum(need[source].values()) == 0:
                continue
            text = text.strip()
            # 6 characters per token is ample, even for code and maths
            add(source, tokenizer(text[:6 * max_len], add_special_tokens=False, verbose=False)["input_ids"])
            if all(sum(n.values()) == 0 for n in need.values()):
                break
    short = {s: len(v) for s, v in found.items() if len(v) < spec.min_docs}
    if short:
        raise ValueError(f"too few documents for {short} (need >= {spec.min_docs})")
    prompts = [p for s in spec.sources for p in found[s]]
    for i, p in enumerate(prompts):
        n_cand = len(p["ids"]) - 1  # never the final </s>
        k = min(spec.queries_per_doc, n_cand)
        p["queries"] = sorted(int(q) for q in np.random.default_rng([spec.seed, 1_000_003, i]).choice(
            n_cand, k, replace=False))
        p["text"] = tokenizer.decode(p["ids"])
    return prompts


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
    from huggingface_hub import snapshot_download

    spec = TRANSLATION[name]
    table = read_arrow_dir(snapshot_download(spec.repo, repo_type="dataset", revision=spec.revision,
                                             cache_dir=cache_dir))
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
    if name in DOCMIX:
        return [p["text"] for p in load_docmix(name, cache_dir=cache_dir)]
    if name in MULTIDOC:
        return [p["text"] for p in load_multidoc(name, cache_dir=cache_dir)]
    from datasets import load_dataset

    spec = DATASETS[name]
    data = load_dataset("nyu-mll/glue", spec.config, split=spec.split, cache_dir=cache_dir)
    limit = spec.default_max_examples if max_examples is None else max_examples
    if limit is not None and limit > 0:
        data = data.select(range(min(limit, len(data))))
    return [format_example(ex, spec, pair_format) for ex in data]
