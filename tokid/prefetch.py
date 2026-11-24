"""Download every model and dataset into the Hugging Face cache (run once, with internet).

Afterwards the jobs can run with HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1, so compute
nodes never need network access.

    python -m tokid.prefetch --models t5-small t5-base elf-b elf-b-de-en elf-b-xsum --datasets mrpc cola mnli wmt14 docmix multidoc
"""

import argparse
from pathlib import Path

from tokid.data import DATASET_NAMES, load_texts
from tokid.models import DEFAULT_ELF_SOURCE, load_elf, load_t5


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    models = ["t5-small", "t5-base", "elf-b", "elf-b-de-en", "elf-b-xsum"]
    p.add_argument("--models", nargs="+", default=models, choices=models)
    p.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=DATASET_NAMES)
    p.add_argument("--elf-source", type=Path, default=DEFAULT_ELF_SOURCE)
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args(argv)

    for name in args.datasets:
        texts = load_texts(name, cache_dir=args.cache_dir)
        print(f"[prefetch] dataset {name}: {len(texts)} examples")
    for model in args.models:
        if model.startswith("elf-b"):
            weights = "owt" if model == "elf-b" else model.removeprefix("elf-b-")
            _, _, meta = load_elf(args.elf_source, cache_dir=args.cache_dir, weights=weights)
            print(f"[prefetch] {meta['model']} checkpoint step {meta['checkpoint_step']} + {meta['encoder']}")
        else:
            _, _, meta = load_t5(f"google-t5/{model}", cache_dir=args.cache_dir)
            print(f"[prefetch] {meta['model']} @ {meta['revision']}")
    print("[prefetch] done")


if __name__ == "__main__":
    main()
