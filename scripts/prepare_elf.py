"""Check out the official PyTorch ELF implementation at a pinned commit.

The ELF model code is imported from this checkout; it is never modified.
By default it is cloned from GitHub, so it works on a fresh machine. Pass
``--source /path/to/local/ELF`` to clone from an existing local clone instead.
"""

import argparse
from pathlib import Path
import subprocess


REPO_URL = "https://github.com/lillian039/ELF.git"
BRANCH = "pytorch_elf"
REVISION = "b29d8833609e9ab7f67cd9da39435ac5cea04837"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=REPO_URL, help="Git URL or local path of an ELF clone.")
    parser.add_argument("--destination", type=Path, default=ROOT / ".deps/elf-pytorch")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        head = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if head != REVISION:
            parser.error(f"{destination} is at {head}; expected {REVISION}. Remove it or choose another destination.")
        print(f"Already prepared: {destination} ({head})")
        return
    source = args.source
    if Path(source).exists():
        source = str(Path(source).resolve())
    subprocess.run(["git", "clone", "--no-checkout", source, str(destination)], check=True)
    # The pinned commit lives on the pytorch_elf branch; fetch it if the clone lacks it.
    have = subprocess.run(["git", "-C", str(destination), "cat-file", "-e", f"{REVISION}^{{commit}}"],
                          capture_output=True).returncode == 0
    if not have:
        subprocess.run(["git", "-C", str(destination), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", REVISION], check=True)
    if not (destination / "src/modules/model.py").is_file():
        raise SystemExit("Checkout does not contain src/modules/model.py; wrong repository?")
    print(f"Prepared {destination} at {REVISION}")


if __name__ == "__main__":
    main()
