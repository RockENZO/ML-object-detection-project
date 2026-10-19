"""Export a stripped checkpoint with a report identifying its actual bytes."""
import argparse
import json
from pathlib import Path
import shutil

from prepare_grouped_split import digest


def export_training(run, output, stripper=None):
    run, output = Path(run), Path(output)
    source = run / 'weights/best.pt'
    if (output / 'best.pt').resolve() == source.resolve():
        raise ValueError('Export must not overwrite the training checkpoint')
    report = json.loads((run.parent / (run.name + '-selection.json')).read_text())
    original_hash = digest(source)
    if original_hash != report['checkpoint_sha256']:
        raise ValueError('Training checkpoint differs from the selection report')
    if stripper is None:
        from ultralytics.utils.torch_utils import strip_optimizer
        stripper = strip_optimizer
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'best.pt'
    stripper(str(source), str(target))
    report['training_checkpoint_sha256'] = original_hash
    report['checkpoint_sha256'] = digest(target)
    report['export'] = {'optimizer_stripped': True, 'checkpoint': 'best.pt'}
    shutil.copyfile(run / 'results.csv', output / 'results.csv')
    (output / 'selection.json').write_text(json.dumps(report, indent=2) + '\n')
    state = run.parent / (run.name + '-training.json')
    if state.is_file():
        shutil.copyfile(state, output / 'training.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    export_training(args.run, args.output)
