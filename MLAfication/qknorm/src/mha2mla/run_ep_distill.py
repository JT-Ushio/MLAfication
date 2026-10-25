"""Probe the largest feasible micro-batch in fresh processes, then train."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from ep_distill_utils import accumulation_steps, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume-tag')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    run_dir = Path(config['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_path = run_dir / 'selected-micro-batch.json'
    trainer = Path(__file__).with_name('train_ep.py')
    base = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node', str(config['ep_size']), str(trainer), '--config', args.config]
    if args.resume_tag:
        selected = json.loads(selected_path.read_text())['micro_batch']
    else:
        if selected_path.exists() or (Path(config['checkpoint_dir']) / 'native' / 'latest').exists():
            raise RuntimeError('Run already started; use --resume-tag or a new run directory')
        selected = None
        for micro_batch in config.get('micro_batch_candidates', [8, 4, 2, 1]):
            accumulation_steps(config['global_batch_size'], micro_batch, config['ep_size'])
            result_path = run_dir / f'probe-mb{micro_batch}-result.json'
            # A stale result must never make a crashed retry look successful.
            result_path.unlink(missing_ok=True)
            log_path = run_dir / f'probe-mb{micro_batch}.log'
            with log_path.open('w') as log:
                result = subprocess.run(base + ['--micro-batch', str(micro_batch), '--probe-steps',
                                               str(config.get('probe_steps', 3))], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                tail = log_path.read_text(errors='replace')
                if re.search(r'CUDA out of memory|torch\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED', tail):
                    atomic_json(run_dir / f'probe-mb{micro_batch}-failure.json',
                                {'reason': 'CUDA OOM', 'returncode': result.returncode})
                    continue
                raise RuntimeError(f'Probe failed for a reason other than CUDA OOM; inspect {log_path}')
            report = json.loads(result_path.read_text())
            headroom = report['device_total_gib'] - report['peak_reserved_gib']
            if headroom < config.get('minimum_headroom_gib', 8):
                atomic_json(run_dir / f'probe-mb{micro_batch}-failure.json',
                            {'reason': 'insufficient memory margin', 'headroom_gib': headroom})
                continue
            selected = micro_batch
            atomic_json(selected_path, dict(report, headroom_gib=headroom))
            break
        if selected is None:
            raise RuntimeError('No candidate fit with the configured memory margin')
    command = base + ['--micro-batch', str(selected)]
    if args.resume_tag:
        command += ['--resume-tag', args.resume_tag]
    with (run_dir / 'training.log').open('a') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
