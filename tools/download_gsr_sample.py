"""Fetch one permitted GSR clip from a pinned public ZIP using HTTP ranges."""

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from match_analysis.hashing import digest

REVISION = "3cc710eb6d53a23350a3d2863311c1c0a2e645d7"


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        response = requests.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        self.size = int(response.headers["content-length"])
        self.pos = 0
        self.cache = None

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, n, whence=0):
        self.pos = n if whence == 0 else self.pos + n if whence == 1 else self.size + n
        return self.pos

    def fetch(self, start, end):
        response = requests.get(
            self.url + f"?range={start}-{end}",
            headers={"Range": f"bytes={start}-{end - 1}"},
            timeout=300,
        )
        if response.status_code != 206 or len(response.content) != end - start:
            raise RuntimeError("HTTP range unavailable; refusing full archive download")
        return response.content

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos + n)
        if self.cache and self.cache[0] <= self.pos and end <= self.cache[1]:
            data = self.cache[2][self.pos - self.cache[0] : end - self.cache[0]]
        else:
            data = self.fetch(self.pos, end) if end > self.pos else b""
        self.pos = end
        return data


def download(clip, output, split="valid"):
    if not re.fullmatch(r"SNGS-\d{3}", clip):
        raise ValueError("Clip ID must be SNGS-000 format")
    output = Path(output)
    root = output / clip
    if root.exists():
        raise ValueError("Clip directory exists; preserve downloaded provenance")
    url = f"https://huggingface.co/datasets/SoccerNet/SN-GSR-2024/resolve/{REVISION}/{split}.zip"
    remote = RemoteZip(url)
    with zipfile.ZipFile(remote) as archive:
        entries = [i for i in archive.infolist() if i.filename.startswith(clip + "/")]
        if not entries:
            raise ValueError("Clip is not in this source split")
        lo = min(i.header_offset for i in entries)
        hi = max(
            i.header_offset
            + 30
            + len(i.filename.encode())
            + len(i.extra)
            + i.compress_size
            for i in entries
        )
        if hi - lo > 1024**3:
            raise ValueError("Unexpected clip span over 1 GiB")
        print(f"Downloading {clip}: {(hi - lo) / 1024**2:.1f} MiB", flush=True)
        remote.cache = (lo, hi, remote.fetch(lo, hi))
        for item in entries:
            if item.is_dir():
                continue
            path = (output / item.filename).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Unsafe ZIP path")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(item))
    files = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    (root / "download.json").write_text(
        json.dumps(
            {
                "source": url,
                "revision": REVISION,
                "source_split": split,
                "file_sha256": files,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Downloaded labels and JPEGs. Follow documented ffmpeg conversion and freeze commands.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="SNGS-033")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    download(args.clip, args.output, args.split)
