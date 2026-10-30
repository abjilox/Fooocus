"""Prepare Fooocus without replacing Colab's preinstalled GPU stack."""

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from modules.colab_support import atomic_copy, record_timing

PROTECTED = {'torch', 'torchvision', 'torchaudio', 'triton', 'numpy'}


def installed_packages():
    return {dist.metadata['Name'].lower().replace('_', '-'): dist.version
            for dist in importlib.metadata.distributions() if dist.metadata['Name']}


def protected_constraints(installed):
    return [f'{name}=={version}' for name, version in sorted(installed.items())
            if name in PROTECTED or name.startswith(('nvidia-', 'cuda-'))]


def install_requirements(requirements, installed):
    with tempfile.TemporaryDirectory(prefix='fooocus-pip-') as temporary:
        constraints = Path(temporary) / 'constraints.txt'
        constraints.write_text('\n'.join(protected_constraints(installed)) + '\n')
        # Wheels only: never spend a Colab session compiling native extensions.
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
                        '--constraint', str(constraints), '-r', str(requirements)], check=True)
    after = installed_packages()
    for name, version in installed.items():
        if name in PROTECTED or name.startswith(('nvidia-', 'cuda-')):
            if after.get(name) != version:
                raise RuntimeError(f'Colab package unexpectedly changed: {name} {version} -> {after.get(name)}')


def ensure_environment(requirements):
    from modules.launch_util import requirements_met
    if requirements_met(str(requirements)):
        print('[Colab] Requirements satisfied; no installation needed.')
        return
    install_requirements(requirements, installed_packages())
    if not requirements_met(str(requirements)):
        raise RuntimeError('Colab requirements are still not satisfied after installation.')


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


def prepare(repo, state_file, drive_root=None):
    started = time.time()
    check_runtime()
    env = prepare_paths(repo, drive_root)
    env.update(FOOOCUS_COLAB_STARTED=str(started), FOOOCUS_COLAB_TIMINGS=str(Path(state_file).with_suffix('.timings.jsonl')))
    os.environ.update(env)
    ensure_environment(env['REQS_FILE'])
    record_timing('environment_ready')
    packages = installed_packages()
    Path(state_file).with_suffix('.packages.json').write_text(json.dumps(packages, indent=2, sort_keys=True))
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    Path(state_file).write_text(json.dumps({'repo': str(Path(repo).resolve()), 'revision': revision, 'env': env}, indent=2))
    print('[Colab] Setup complete. Run the Start cell.')


def start(state_file, extra_args=()):
    state = json.loads(Path(state_file).read_text())
    env = os.environ.copy()
    # Do not inherit Drive paths from a previous prepare without Drive.
    for key in ('FOOOCUS_DRIVE_ROOT', 'FOOOCUS_DRIVE_MODEL_ROOT', 'FOOOCUS_DRIVE_OUTPUT_ROOT', 'FOOOCUS_DRIVE_HF_HOME'):
        env.pop(key, None)
    env.update(state['env'])
    env['FOOOCUS_COLAB_STARTED'] = str(time.time())
    subprocess.run([sys.executable, 'launch.py', '--share', '--disable-xformers',
                    '--attention-pytorch', *extra_args], cwd=state['repo'], env=env, check=True)
