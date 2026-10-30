"""Build once, restore locally: additional packages only, never a CUDA environment."""
import compileall
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

MARKER = '.fooocus-cache.json'
SCHEMA = 1
GPU_PACKAGES = {'torch', 'torchvision', 'torchaudio', 'triton'}
PROTECTED = GPU_PACKAGES | {'numpy', 'gradio', 'transformers'}
SETUP_HINT = 'Run the one-time Einrichtung / prepare cell to create or renew the package cache.'


class CacheError(RuntimeError):
    pass


def canonical(name):
    return name.lower().replace('_', '-').replace('.', '-')


def gpu_package(name):
    return name in GPU_PACKAGES or name.startswith(('nvidia-', 'cuda-'))


def installed_packages():
    # distributions() yields the first search path first; an overlay must win.
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get('Name')
        if name:
            packages.setdefault(canonical(name), dist.version)
    return packages


def protected_constraints(installed):
    return [f'{name}=={version}' for name, version in sorted(installed.items())
            if name in PROTECTED or gpu_package(name)]


def cache_key(requirements, installed):
    identity = {
        'schema': SCHEMA, 'requirements': hashlib.sha256(Path(requirements).read_bytes()).hexdigest(),
        'python': list(sys.version_info[:2]), 'abi': sys.implementation.cache_tag,
        'bytecode': importlib.util.MAGIC_NUMBER.hex(), 'system': platform.system(),
        'machine': platform.machine(), 'libc': platform.libc_ver(),
        'protected': {n: v for n, v in sorted(installed.items()) if n in PROTECTED or gpu_package(n)},
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def new_manifest(requirements, installed, native_packages, additions):
    return {'schema': SCHEMA, 'key': cache_key(requirements, installed),
            'native_packages': native_packages, 'additions': additions,
            'created_at': time.time()}


def _valid(manifest, requirements, installed):
    return (manifest.get('schema') == SCHEMA and manifest.get('key') == cache_key(requirements, installed)
            and all(installed.get(n) == v for n, v in manifest.get('native_packages', {}).items()))


def _read_manifest(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise CacheError(f'Package cache manifest unavailable: {error}. {SETUP_HINT}') from error


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def publish_cache(package_dir, cache_root, manifest):
    """Publish an immutable archive, then atomically switch its manifest pointer."""
    package_dir, cache_root = Path(package_dir), Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    files = [p for p in package_dir.rglob('*') if p.name != MARKER]
    manifest['unpacked_bytes'] = sum(p.stat().st_size for p in files if p.is_file())
    with tempfile.TemporaryDirectory(prefix='.publish-', dir=cache_root) as temporary:
        archive_path = Path(temporary) / 'packages.tar.gz'
        with tarfile.open(archive_path, 'w:gz', compresslevel=1, dereference=True) as archive:
            for path in sorted(files):
                archive.add(path, arcname=str(path.relative_to(package_dir)), recursive=False)
        manifest['sha256'] = _digest(archive_path)
        manifest['archive'] = f"{manifest['key']}-{manifest['sha256']}.tar.gz"
        os.replace(archive_path, cache_root / manifest['archive'])
        _write_json(cache_root / (manifest['key'] + '.json'), manifest)
    return manifest


def _replace_packages(staged, target):
    target = Path(target)
    if target.exists() and not (target / MARKER).is_file():
        raise CacheError(f'Refusing to replace an existing non-cache directory: {target}')
    with tempfile.TemporaryDirectory(prefix='.previous-', dir=target.parent) as temporary:
        previous = Path(temporary) / 'packages'
        if target.exists():
            target.rename(previous)
        try:
            os.replace(staged, target)
        except BaseException:
            if previous.exists():
                previous.rename(target)
            raise


def restore_cache(requirements, target, cache_root, installed=None):
    installed = installed if installed is not None else installed_packages()
    target, cache_root = Path(target), Path(cache_root)
    if (target / MARKER).is_file():
        local = _read_manifest(target / MARKER)
        if _valid(local, requirements, installed):
            print('[Colab cache] Local packages ready; no transfer or installation.', flush=True)
            return target
    manifest = _read_manifest(cache_root / (cache_key(requirements, installed) + '.json'))
    if not _valid(manifest, requirements, installed):
        raise CacheError(f'Package cache does not match this runtime. {SETUP_HINT}')
    digest = manifest.get('sha256', '')
    expected_name = f"{manifest['key']}-{digest}.tar.gz"
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or manifest.get('archive') != expected_name:
        raise CacheError(f'Invalid package archive manifest. {SETUP_HINT}')
    if target.exists() and not (target / MARKER).is_file():
        raise CacheError(f'Refusing to replace an existing non-cache directory: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        archive_source = cache_root / expected_name
        needed = manifest['unpacked_bytes'] + archive_source.stat().st_size
        if shutil.disk_usage(target.parent).free < needed:
            raise CacheError(f'Not enough local space for the package cache ({needed} bytes).')
        with tempfile.TemporaryDirectory(prefix='.restore-', dir=target.parent) as temporary:
            archive_path = Path(temporary) / 'packages.tar.gz'
            print('[Colab cache] Copying one package archive from Drive...', flush=True)
            shutil.copyfile(archive_source, archive_path)
            if _digest(archive_path) != digest:
                raise CacheError(f'Package archive checksum mismatch. {SETUP_HINT}')
            print(f'[Colab cache] Transfer: {time.monotonic() - started:.1f}s', flush=True)
            unpack_started = time.monotonic()
            staged = Path(temporary) / 'packages'
            staged.mkdir()
            with tarfile.open(archive_path, 'r:gz') as archive:
                members = archive.getmembers()
                for member in members:
                    path = Path(member.name)
                    if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                        raise CacheError('Unsafe member in package archive.')
                archive.extractall(staged, members=members, filter='data')
            _write_json(staged / MARKER, manifest)
            _replace_packages(staged, target)
            print(f'[Colab cache] Local extraction: {time.monotonic() - unpack_started:.1f}s', flush=True)
    except (OSError, tarfile.TarError, KeyError) as error:
        raise CacheError(f'Could not restore package cache: {error}. {SETUP_HINT}') from error
    return target


def package_environment(target, repo=None):
    env = os.environ.copy()
    paths = [str(target)] + ([str(repo)] if repo else [])
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env['PYTHONUNBUFFERED'] = '1'
    return env


_PROBE = r'''
import importlib.metadata as m, json, sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pathlib import Path
requirements, target, output = map(Path, sys.argv[1:])
additions = {canonicalize_name(d.metadata['Name']): d.version for d in m.distributions(path=[str(target)])}
queue = [Requirement(line) for line in requirements.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
native, seen = {}, set()
while queue:
    requirement = queue.pop()
    if requirement.marker and not requirement.marker.evaluate({'extra': ''}): continue
    name = canonicalize_name(requirement.name)
    dist = m.distribution(name)
    if dist.version not in requirement.specifier:
        raise RuntimeError(f'{name} {dist.version} does not satisfy {requirement}')
    if name in seen: continue
    seen.add(name)
    if name not in additions: native[name] = dist.version
    queue.extend(Requirement(r) for r in (dist.requires or []))
import numpy, torch, torchvision, gradio, transformers, cv2, scipy, torchsde, timm
from rembg import remove
from segment_anything import sam_model_registry
from modules.transformers_compat import FooocusCLIPTextModel, no_init_weights
from modules.gradio_compat import gr
json.dump({'native': native, 'additions': additions}, output.open('w'))
print('[Colab cache] Dependency and import checks passed.', flush=True)
'''


def build_cache(requirements, target, cache_root, repo, rebuild=False):
    requirements, target, repo = Path(requirements), Path(target), Path(repo)
    installed = installed_packages()
    if target.exists() and not (target / MARKER).is_file():
        raise CacheError(f'Refusing to replace an existing non-cache directory: {target}')
    if not rebuild:
        try:
            return restore_cache(requirements, target, cache_root, installed)
        except CacheError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    print('[Colab cache] One-time setup: resolving only missing/incompatible packages.', flush=True)
    with tempfile.TemporaryDirectory(prefix='.build-', dir=target.parent) as temporary:
        work = Path(temporary)
        constraints = work / 'native.txt'
        constraints.write_text('\n'.join(protected_constraints(installed)) + '\n')
        report = work / 'report.json'
        subprocess.run([sys.executable, '-u', '-m', 'pip', 'install', '--dry-run',
                        '--only-binary=:all:', '--report', str(report), '-c', str(constraints),
                        '-r', str(requirements)], check=True)
        selected = json.loads(report.read_text())['install']
        urls = []
        for package in selected:
            name = canonical(package['metadata']['name'])
            if gpu_package(name) or (name in PROTECTED and name in installed):
                raise CacheError(f'Refusing to replace native package {name}.')
            download = package['download_info']
            url = download['url']
            if not url.split('?', 1)[0].endswith('.whl'):
                raise CacheError(f'No binary wheel for {name}; source compilation is disabled.')
            digest = download['archive_info']['hashes']['sha256']
            urls.append(url.split('#')[0] + '#sha256=' + digest)
        staged = work / 'packages'
        staged.mkdir()
        if urls:
            print(f'[Colab cache] Installing {len(urls)} additional packages locally.', flush=True)
            subprocess.run([sys.executable, '-u', '-m', 'pip', 'install', '--no-deps', '--ignore-installed',
                            '--no-compile', '--only-binary=:all:', '--target', str(staged), *urls], check=True)
        else:
            print('[Colab cache] All required packages already provided by Colab.', flush=True)
        metadata = work / 'validated.json'
        subprocess.run([sys.executable, '-u', '-c', _PROBE, str(requirements), str(staged), str(metadata)],
                       cwd=repo, env=package_environment(staged, repo), check=True)
        # Validate the actual adapters/model paths once; never during normal starts.
        subprocess.run([sys.executable, '-u', str(repo / 'scripts/run_tests.py')],
                       cwd=repo, env=package_environment(staged, repo), check=True)
        result = json.loads(metadata.read_text())
        if any(installed.get(n) != v for n, v in result['native'].items()):
            raise CacheError('Native packages changed while preparing the cache; retry setup.')
        compileall.compile_dir(str(staged), ddir=str(target), quiet=1)
        manifest = new_manifest(requirements, installed, result['native'], result['additions'])
        manifest = publish_cache(staged, cache_root, manifest)
        _write_json(staged / MARKER, manifest)
        _replace_packages(staged, target)
        print('[Colab cache] Verified package cache saved. Future starts do not install packages.', flush=True)
    return target
