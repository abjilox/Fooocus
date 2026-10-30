"""Colab entry point. Keep notebook cells small and test setup independently."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.colab_runtime import prepare, start, DEFAULT_DEPENDENCIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'start'])
    parser.add_argument('--state-file', default='/content/fooocus-colab-state.json')
    parser.add_argument('--drive-root', default=None)
    parser.add_argument('--dependency-dir', default=DEFAULT_DEPENDENCIES)
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the package cache during prepare only')
    args, extra = parser.parse_known_args()
    if args.action == 'prepare':
        if extra:
            parser.error(f'Unexpected setup arguments: {extra}')
        prepare(ROOT, args.state_file, args.drive_root, args.dependency_dir, args.rebuild)
    else:
        if args.rebuild:
            parser.error('--rebuild is only available with prepare')
        start(args.state_file, extra, repo=ROOT, drive_root=args.drive_root, dependency_dir=args.dependency_dir)


if __name__ == '__main__':
    main()
