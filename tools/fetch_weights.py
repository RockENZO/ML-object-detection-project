"""Download the published demo detector without loading it or overwriting files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

# Immutable source and independently recorded digest in the checked-in
# reports/detector_ball1280robust_val_cpu_20261019.json validation report.
SOURCE_COMMIT = 'f02f90d42acf560a955d58cb821e5dd8c10986d7'
CHECKPOINT_SHA256 = '5c5674f7775fd67b9ef6cd49a62ab8bc9aba5187bece9cc420897af5060ed3d3'
SOURCE_URL = ('https://raw.githubusercontent.com/RockENZO/ML-object-detection-project/'
              + SOURCE_COMMIT + '/best.pt')
MAX_BYTES = 32 * 1024 * 1024


def fetch_weights(output):
    """Verify a bounded download, then atomically claim an unused output path.

    No torch/pickle loading, extraction, target dependency installation or
    training. Hard-link support is required; never fall back to overwrite.
    """
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Checkpoint destination already exists; choose a new path')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.weights-', delete=False) as staged:
            temporary = Path(staged.name)
            digest, total = hashlib.sha256(), 0
            with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
                while True:
                    block = response.read(min(1024 * 1024, MAX_BYTES + 1 - total))
                    if not block:
                        break
                    total += len(block)
                    if total > MAX_BYTES:
                        raise ValueError('Checkpoint exceeds the 32 MiB download limit')
                    digest.update(block)
                    staged.write(block)
            if digest.hexdigest() != CHECKPOINT_SHA256:
                raise ValueError('Checkpoint SHA-256 mismatch; no output published')
        # Same-filesystem, atomic no-clobber publication. A concurrent file or
        # symlink is preserved even if it appeared after the initial check.
        os.link(temporary, output)
        return {'path': str(output), 'bytes': total, 'sha256': digest.hexdigest(),
                'source_commit': SOURCE_COMMIT, 'source_url': SOURCE_URL,
                'license': 'AGPL-3.0', 'model_loaded': False}
    finally:
        if temporary is not None:
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/football-ball-1280-robust.pt'))
    args = parser.parse_args(argv)
    try:
        report = fetch_weights(args.output)
    except (OSError, ValueError, urllib.error.URLError) as error:
        parser.exit(1, f'Checkpoint download failed: {error}\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
