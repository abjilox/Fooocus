"""Local working files with an optional persistent Drive mirror (no ML imports)."""

import json
import os
from pathlib import Path
import shutil
import time


def atomic_copy(source, destination):
    """Copy only changed complete files; a failed copy cannot replace a good file."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        return False
    stat = source.stat()
    if destination.is_file():
        previous = destination.stat()
        # Drive's filesystem may round modification times to whole seconds.
        if previous.st_size == stat.st_size and int(previous.st_mtime) == int(stat.st_mtime):
            return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < stat.st_size:
        raise OSError(f'Not enough free space to copy {source} ({stat.st_size} bytes)')
    temporary = destination.with_name(destination.name + '.partial')
    try:
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != stat.st_size:
            raise OSError(f'Incomplete copy of {source}')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def stage_tree(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        return
    for path in source.rglob('*'):
        if path.is_file() and not path.name.endswith(('.partial', '.incomplete', '.lock')):
            atomic_copy(path, destination / path.relative_to(source))


def drive_model_path(local_path):
    local_root = os.environ.get('FOOOCUS_MODEL_ROOT')
    drive_root = os.environ.get('FOOOCUS_DRIVE_MODEL_ROOT')
    if not local_root or not drive_root:
        return None
    try:
        relative = Path(local_path).resolve().relative_to(Path(local_root).resolve())
    except ValueError:
        return None
    return Path(drive_root) / relative


def stage_model(local_path):
    local_path = Path(local_path)
    if local_path.is_file():
        return False
    drive_path = drive_model_path(local_path)
    if drive_path is not None and drive_path.is_file():
        started = time.monotonic()
        print(f'[Colab] Copying model to local disk: {drive_path.name}')
        atomic_copy(drive_path, local_path)
        print(f'[Colab] Model ready in {time.monotonic() - started:.1f}s')
        return True
    return False


def persist_model(local_path):
    drive_path = drive_model_path(local_path)
    if drive_path is not None:
        try:
            atomic_copy(local_path, drive_path)
        except OSError as error:
            print(f'[Colab] Could not save model to Drive; local file retained: {error}')


def persist_output(filename):
    local_root = os.environ.get('FOOOCUS_OUTPUT_ROOT')
    drive_root = os.environ.get('FOOOCUS_DRIVE_OUTPUT_ROOT')
    if not local_root or not drive_root:
        return False
    try:
        relative = Path(filename).resolve().relative_to(Path(local_root).resolve())
        atomic_copy(filename, Path(drive_root) / relative)
        return True
    except (OSError, ValueError) as error:
        print(f'[Colab] Could not save output to Drive; local file retained at {filename}: {error}')
        return False


def stage_hf_cache():
    local = os.environ.get('HF_HOME')
    drive = os.environ.get('FOOOCUS_DRIVE_HF_HOME')
    if local and drive:
        stage_tree(drive, local)


def persist_hf_cache():
    local = os.environ.get('HF_HOME')
    drive = os.environ.get('FOOOCUS_DRIVE_HF_HOME')
    if local and drive:
        try:
            stage_tree(local, drive)
        except OSError as error:
            print(f'[Colab] Could not save Hugging Face cache to Drive; local files retained: {error}')


def persist_runtime_files():
    local = os.environ.get('FOOOCUS_REPO_ROOT')
    drive = os.environ.get('FOOOCUS_DRIVE_ROOT')
    if local and drive:
        for name in ('config.txt', 'hash_cache.txt'):
            source = Path(local) / name
            if source.is_file():
                try:
                    atomic_copy(source, Path(drive) / name)
                except OSError as error:
                    print(f'[Colab] Could not save {name} to Drive: {error}')


def record_timing(phase):
    started = os.environ.get('FOOOCUS_COLAB_STARTED')
    if started:
        elapsed = time.time() - float(started)
        print(f'[Colab timing] {phase}: {elapsed:.1f}s since start', flush=True)
        path = os.environ.get('FOOOCUS_COLAB_TIMINGS')
        if path:
            with open(path, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps({'phase': phase, 'started_at': float(started), 'elapsed_seconds': elapsed}) + '\n')
