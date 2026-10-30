"""Colab entry point. Keep notebook cells small and test setup independently."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.colab_runtime import prepare, start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'start'])
    parser.add_argument('--state-file', default='/content/fooocus-colab-state.json')
    parser.add_argument('--drive-root', default=None)
    args, extra = parser.parse_known_args()
    if args.action == 'prepare':
        if extra:
            parser.error(f'Unexpected setup arguments: {extra}')
        prepare(ROOT, args.state_file, args.drive_root)
    else:
        start(args.state_file, extra)


if __name__ == '__main__':
    main()
