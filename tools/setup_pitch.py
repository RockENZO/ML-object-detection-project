"""Fetch pinned PnLCalib source and verified official weights into ignored artifacts."""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from match_analysis.hashing import digest  # noqa: E402


def setup(destination):
    assets = json.loads((ROOT / "match_analysis/pnl_assets.json").read_text())
    if not destination.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", assets["repository"], str(destination)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", assets["commit"]], check=True
        )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        subprocess.call(["git", "-C", str(destination), "diff", "--quiet", "HEAD"])
        or actual != assets["commit"]
    ):
        raise ValueError("PnLCalib checkout differs from pinned commit")
    for name, record in assets["weights"].items():
        target = destination / name
        if not target.exists():
            temporary = target.with_suffix(".download")
            urllib.request.urlretrieve(record["url"], temporary)
            if digest(temporary) != record["sha256"]:
                temporary.unlink()
                raise ValueError("Weight checksum mismatch")
            temporary.replace(target)
        if digest(target) != record["sha256"]:
            raise ValueError("Weight checksum mismatch: " + name)
    print("Verified pinned calibration assets at " + str(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/pnlcalib")
    setup(parser.parse_args().output)
