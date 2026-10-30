"""Prepare Fooocus without replacing Colab's preinstalled GPU stack."""

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from modules.colab_support import atomic_copy, record_timing
from modules.colab_cache import (CacheError, SETUP_HINT, installed_packages, protected_constraints,
                                 build_cache, restore_cache, package_environment, MARKER)
from modules.colab_process import run_live

DEFAULT_DEPENDENCIES = '/content/fooocus-deps'


def ensure_environment(requirements):
    """Validation only. The normal application start must never run pip."""
    from modules.launch_util import requirements_met
    if not requirements_met(str(requirements)):
        raise CacheError(f'Prepared packages do not satisfy the application profile. {SETUP_HINT}')
    print('[Colab] Prepared packages verified; no installation.', flush=True)


def check_runtime():
    print(f'[Colab] Python: {sys.version}', flush=True)
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError('Use the current Python 3.13 Colab GPU runtime (Runtime > Change runtime type).')
    import torch
    import torchvision
    print(f'[Colab] torch={torch.__version__}, torchvision={torchvision.__version__}, CUDA={torch.version.cuda}')
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU available. In Colab select Runtime > Change runtime type > GPU.')
    print(f'[Colab] GPU: {torch.cuda.get_device_name(0)}', flush=True)
    for name in ('numpy', 'transformers', 'gradio', 'tokenizers'):
        try:
            print(f'[Colab] {name}={importlib.metadata.version(name)}')
        except importlib.metadata.PackageNotFoundError:
            print(f'[Colab] {name}: not installed')


MODEL_PATHS = {
    'path_checkpoints': 'checkpoints', 'path_loras': 'loras', 'path_vae': 'vae',
    'path_embeddings': 'embeddings', 'path_vae_approx': 'vae_approx',
    'path_upscale_models': 'upscale_models', 'path_inpaint': 'inpaint',
    'path_controlnet': 'controlnet', 'path_clip_vision': 'clip_vision',
    'path_fooocus_expansion': 'prompt_expansion/fooocus_expansion',
    'path_safety_checker': 'safety_checker',
    'path_sam': 'sam',
}


def prepare_paths(repo, drive_root=None):
    repo = Path(repo).resolve()
    models, outputs = repo / 'models', repo / 'outputs'
    env = {'FOOOCUS_COLAB': '1', 'FOOOCUS_MODEL_ROOT': str(models),
           'FOOOCUS_REPO_ROOT': str(repo),
           'FOOOCUS_OUTPUT_ROOT': str(outputs), 'path_outputs': str(outputs),
           'HF_HOME': str(repo / 'cache/huggingface'),
           'REQS_FILE': str(repo / 'requirements_colab.txt'), 'GRADIO_ANALYTICS_ENABLED': 'False'}
    for key, folder in MODEL_PATHS.items():
        path = models / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    outputs.mkdir(exist_ok=True)
    env['config_path'] = str(repo / 'config.txt')
    if drive_root:
        drive_root = Path(drive_root)
        if not drive_root.parent.is_dir():
            raise RuntimeError('Google Drive is not mounted. Run the Drive mount cell first.')
        drive_root.mkdir(exist_ok=True)
        env.update(FOOOCUS_DRIVE_MODEL_ROOT=str(drive_root / 'models'),
                   FOOOCUS_DRIVE_ROOT=str(drive_root),
                   FOOOCUS_DRIVE_OUTPUT_ROOT=str(drive_root / 'outputs'),
                   FOOOCUS_DRIVE_HF_HOME=str(drive_root / 'cache/huggingface'))
        config = drive_root / 'config.txt'
        if config.is_file() and not (repo / 'config.txt').exists():
            atomic_copy(config, repo / 'config.txt')
        cache = drive_root / 'hash_cache.txt'
        if cache.is_file() and not (repo / 'hash_cache.txt').exists():
            atomic_copy(cache, repo / 'hash_cache.txt')
    return env


def _state(repo, state_file, drive_root, dependency_dir):
    env = prepare_paths(repo, drive_root)
    env.update(FOOOCUS_DEPENDENCY_DIR=str(Path(dependency_dir).resolve()),
               FOOOCUS_COLAB_STARTED=str(time.time()),
               FOOOCUS_COLAB_TIMINGS=str(Path(state_file).with_suffix('.timings.jsonl')))
    python_env = package_environment(dependency_dir, repo)
    env.update(PYTHONPATH=python_env['PYTHONPATH'], PYTHONUNBUFFERED='1')
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    state = {'repo': str(Path(repo).resolve()), 'revision': revision, 'env': env}
    Path(state_file).write_text(json.dumps(state, indent=2))
    return state


def _cache_root(repo, drive_root):
    return Path(drive_root if drive_root else repo) / 'cache/python'


def prepare(repo, state_file, drive_root=None, dependency_dir=DEFAULT_DEPENDENCIES, rebuild=False):
    started = time.time()
    check_runtime()
    # Check the mount before doing any one-time package work.
    prepare_paths(repo, drive_root)
    requirements = Path(repo) / 'requirements_colab.txt'
    build_cache(requirements, dependency_dir, _cache_root(repo, drive_root), repo, rebuild=rebuild)
    state = _state(repo, state_file, drive_root, dependency_dir)
    os.environ.update(state['env'])
    os.environ['FOOOCUS_COLAB_STARTED'] = str(started)
    record_timing('package_cache_ready')
    manifest = json.loads((Path(dependency_dir) / MARKER).read_text())
    packages = dict(installed_packages(), **manifest['additions'])
    Path(state_file).with_suffix('.packages.json').write_text(json.dumps(packages, indent=2, sort_keys=True))
    print('[Colab] One-time setup complete. Next sessions: Konfiguration, then Start.', flush=True)


def start(state_file, extra_args=(), repo=None, drive_root=None, dependency_dir=DEFAULT_DEPENDENCIES):
    # CLI callers supply the checkout and mount directly: a fresh VM needs no state file.
    if repo is None:
        try:
            previous = json.loads(Path(state_file).read_text())
        except (OSError, ValueError) as error:
            raise CacheError(f'No checkout was supplied. {SETUP_HINT}') from error
        repo = previous['repo']
        drive_root = previous['env'].get('FOOOCUS_DRIVE_ROOT')
        dependency_dir = previous['env'].get('FOOOCUS_DEPENDENCY_DIR', dependency_dir)
    requirements = Path(repo) / 'requirements_colab.txt'
    restore_cache(requirements, dependency_dir, _cache_root(repo, drive_root))
    state = _state(repo, state_file, drive_root, dependency_dir)
    env = os.environ.copy()
    for key in ('FOOOCUS_DRIVE_ROOT', 'FOOOCUS_DRIVE_MODEL_ROOT', 'FOOOCUS_DRIVE_OUTPUT_ROOT', 'FOOOCUS_DRIVE_HF_HOME'):
        env.pop(key, None)
    env.update(state['env'])
    print('[Colab] Starting Fooocus with local cached packages; no pip/resolver.', flush=True)
    run_live([sys.executable, '-u', 'launch.py', '--share', '--disable-xformers',
              '--attention-pytorch', *extra_args], cwd=state['repo'], env=env)
