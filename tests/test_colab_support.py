import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class ColabStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def support(self):
        from modules import colab_support
        return colab_support

    def test_complete_copy_is_reused_and_changed_source_is_updated(self):
        src, dst = self.root / 'drive/model', self.root / 'local/model'
        src.parent.mkdir()
        src.write_bytes(b'weights')
        self.assertTrue(self.support().atomic_copy(src, dst))
        with patch('shutil.copy2', side_effect=AssertionError('copied twice')):
            self.assertFalse(self.support().atomic_copy(src, dst))
        src.write_bytes(b'changed weights')
        self.assertTrue(self.support().atomic_copy(src, dst))
        self.assertEqual(dst.read_bytes(), src.read_bytes())

    def test_interrupted_copy_never_replaces_complete_file(self):
        src, dst = self.root / 'src', self.root / 'dst'
        src.write_bytes(b'new complete weights')
        dst.write_bytes(b'old')
        def fail(source, target):
            Path(target).write_bytes(b'partial')
            raise OSError('Drive disconnected')
        with patch('shutil.copy2', side_effect=fail):
            with self.assertRaises(OSError):
                self.support().atomic_copy(src, dst)
        self.assertEqual(dst.read_bytes(), b'old')

    def test_insufficient_space_does_not_start_copy(self):
        src, dst = self.root / 'src', self.root / 'target/dst'
        src.write_bytes(b'weights')
        with patch('shutil.disk_usage', return_value=types.SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, 'space'):
                self.support().atomic_copy(src, dst)
        self.assertFalse(dst.exists())

    def test_partial_files_are_not_staged(self):
        drive, local = self.root / 'drive', self.root / 'local'
        drive.mkdir()
        (drive / 'model.partial').write_bytes(b'incomplete')
        (drive / 'model').write_bytes(b'complete')
        self.support().stage_tree(drive, local)
        self.assertTrue((local / 'model').exists())
        self.assertFalse((local / 'model.partial').exists())

    def test_only_requested_drive_model_is_staged(self):
        local, drive = self.root / 'local/models', self.root / 'drive/models'
        (drive / 'checkpoints').mkdir(parents=True)
        (drive / 'checkpoints/chosen.safetensors').write_bytes(b'chosen')
        (drive / 'checkpoints/other.safetensors').write_bytes(b'other')
        env = {'FOOOCUS_MODEL_ROOT': str(local), 'FOOOCUS_DRIVE_MODEL_ROOT': str(drive)}
        with patch.dict(os.environ, env):
            self.assertTrue(self.support().stage_model(local / 'checkpoints/chosen.safetensors'))
        self.assertEqual((local / 'checkpoints/chosen.safetensors').read_bytes(), b'chosen')
        self.assertFalse((local / 'checkpoints/other.safetensors').exists())

    def test_model_mirror_cannot_escape_configured_root(self):
        env = {'FOOOCUS_MODEL_ROOT': str(self.root / 'models'),
               'FOOOCUS_DRIVE_MODEL_ROOT': str(self.root / 'drive')}
        with patch.dict(os.environ, env):
            self.assertIsNone(self.support().drive_model_path(self.root / 'outside'))

    def test_drive_failure_preserves_output_and_reports_error(self):
        output = self.root / 'outputs/date/image.png'
        output.parent.mkdir(parents=True)
        output.write_bytes(b'image')
        env = {'FOOOCUS_OUTPUT_ROOT': str(self.root / 'outputs'),
               'FOOOCUS_DRIVE_OUTPUT_ROOT': str(self.root / 'drive')}
        with patch.dict(os.environ, env), patch('shutil.copy2', side_effect=OSError('offline')):
            with patch('builtins.print') as report:
                self.assertFalse(self.support().persist_output(output))
        self.assertEqual(output.read_bytes(), b'image')
        self.assertTrue(any('offline' in str(call) for call in report.call_args_list))


class ColabInstallTests(unittest.TestCase):
    def runtime(self):
        from modules import colab_runtime
        return colab_runtime

    def test_constraints_protect_all_installed_gpu_packages(self):
        installed = {'torch': '2.11.0+cu130', 'torchvision': '0.26.0+cu130',
                     'numpy': '2.4.6', 'triton': '3.6.0', 'nvidia-cudnn-cu13': '9.1',
                     'gradio': '6.31.0', 'unrelated': '1.0'}
        constraints = self.runtime().protected_constraints(installed)
        for name in ('torch', 'torchvision', 'numpy', 'triton', 'nvidia-cudnn-cu13'):
            self.assertIn(f'{name}=={installed[name]}', constraints)
        self.assertNotIn('unrelated==1.0', constraints)

    def test_install_failure_propagates(self):
        from modules.colab_cache import build_cache
        error = subprocess.CalledProcessError(1, ['pip'])
        with tempfile.TemporaryDirectory() as tmp, patch('subprocess.run', side_effect=error):
            requirements = Path(tmp) / 'requirements.txt'
            requirements.write_text('numpy>=1\n')
            with self.assertRaises(subprocess.CalledProcessError):
                build_cache(requirements, Path(tmp) / 'local', Path(tmp) / 'drive', Path.cwd())

    def test_launcher_does_not_swallow_pip_failure(self):
        from modules.launch_util import run_pip
        with patch('modules.launch_util.run', side_effect=RuntimeError('pip failed')):
            with self.assertRaisesRegex(RuntimeError, 'pip failed'):
                run_pip('install missing-package', 'test')

    def test_no_install_when_requirements_are_met(self):
        with patch('modules.launch_util.requirements_met', return_value=True), patch('subprocess.run') as run:
            self.runtime().ensure_environment(Path('requirements_colab.txt'))
        run.assert_not_called()

    def test_normal_start_validation_never_installs_on_mismatch(self):
        from modules.colab_cache import CacheError
        with patch('modules.launch_util.requirements_met', return_value=False), patch('subprocess.run') as run:
            with self.assertRaisesRegex(CacheError, 'prepare'):
                self.runtime().ensure_environment(Path('requirements_colab.txt'))
        run.assert_not_called()

    def test_drive_config_is_copied_without_replacing_existing_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, drive = Path(tmp) / 'repo', Path(tmp) / 'drive/Fooocus'
            repo.mkdir()
            drive.mkdir(parents=True)
            (drive / 'config.txt').write_text('{"default_model": "mine.safetensors"}')
            env = self.runtime().prepare_paths(repo, drive)
            self.assertEqual((repo / 'config.txt').read_text(), (drive / 'config.txt').read_text())
            self.assertEqual(env['path_checkpoints'], str(drive.resolve() / 'models/checkpoints'))
            self.assertEqual(env['path_sam'], str(drive.resolve() / 'models/sam'))
            (repo / 'config.txt').write_text('local config')
            self.runtime().prepare_paths(repo, drive)
            self.assertEqual((repo / 'config.txt').read_text(), 'local config')

    def test_drive_models_and_huggingface_cache_are_used_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, drive = Path(tmp) / 'repo', Path(tmp) / 'drive/Fooocus'
            repo.mkdir()
            drive.mkdir(parents=True)
            env = self.runtime().prepare_paths(repo, drive)
            for key, folder in self.runtime().MODEL_PATHS.items():
                self.assertEqual(Path(env[key]), drive.resolve() / 'models' / folder)
            self.assertEqual(Path(env['HF_HOME']), drive.resolve() / 'cache/huggingface')
            self.assertFalse((repo / 'models/checkpoints').exists())
            from modules import colab_support
            with patch.dict(os.environ, env), patch('modules.colab_support.stage_tree', side_effect=AssertionError('cache copied')):
                colab_support.stage_hf_cache()
                colab_support.persist_hf_cache()
                self.assertIsNone(colab_support.drive_model_path(env['path_checkpoints']))

    def test_no_drive_keeps_local_model_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            env = self.runtime().prepare_paths(repo)
            self.assertEqual(Path(env['path_checkpoints']), repo.resolve() / 'models/checkpoints')
            self.assertEqual(Path(env['HF_HOME']), repo.resolve() / 'cache/huggingface')

    def test_only_missing_packaged_resources_are_seeded_not_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, drive = Path(tmp) / 'repo', Path(tmp) / 'drive/Fooocus'
            resource = repo / 'models/prompt_expansion/fooocus_expansion'
            resource.mkdir(parents=True)
            (resource / 'config.json').write_text('{"model_type":"gpt2"}')
            (resource / 'pytorch_model.bin').write_bytes(b'must not copy')
            drive.mkdir(parents=True)
            self.runtime().prepare_paths(repo, drive)
            destination = drive / 'models/prompt_expansion/fooocus_expansion'
            self.assertEqual((destination / 'config.json').read_text(), '{"model_type":"gpt2"}')
            self.assertFalse((destination / 'pytorch_model.bin').exists())
            (destination / 'config.json').write_text('custom configuration')
            self.runtime().prepare_paths(repo, drive)
            self.assertEqual((destination / 'config.json').read_text(), 'custom configuration')

    def test_cpu_runtime_stops_before_install(self):
        from types import ModuleType, SimpleNamespace
        torch = ModuleType('torch')
        torch.__version__ = '2.11.0'
        torch.version = SimpleNamespace(cuda='13.0')
        torch.cuda = SimpleNamespace(is_available=lambda: False)
        torchvision = ModuleType('torchvision')
        torchvision.__version__ = '0.26.0'
        with patch.dict(sys.modules, {'torch': torch, 'torchvision': torchvision}), patch('sys.version_info', (3, 13)):
            with self.assertRaisesRegex(RuntimeError, 'No CUDA GPU'):
                self.runtime().check_runtime()


class ModelDownloadTests(unittest.TestCase):
    def test_direct_drive_model_never_copies_or_downloads(self):
        from modules.model_loader import load_file_from_url
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / 'drive/models'
            models.mkdir(parents=True)
            (models / 'model').write_bytes(b'existing weights')
            env = {'FOOOCUS_MODEL_ROOT': str(models), 'FOOOCUS_DRIVE_MODEL_ROOT': str(models)}
            with patch.dict(os.environ, env), patch('shutil.copy2', side_effect=AssertionError('model copied')):
                result = load_file_from_url('https://example.org/model', model_dir=str(models))
            self.assertEqual(Path(result), models / 'model')

    def test_textual_embedding_is_staged_on_demand_from_nested_drive_folder(self):
        import ast
        import traceback
        import torch
        import safetensors.torch
        source = Path(__file__).resolve().parents[1] / 'ldm_patched/modules/sd1_clip.py'
        tree = ast.parse(source.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('expand_directory_list', 'load_embed')]
        namespace = {'os': os, 'torch': torch, 'traceback': traceback}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            local, drive = Path(tmp) / 'models', Path(tmp) / 'drive'
            nested = drive / 'embeddings/nested'
            nested.mkdir(parents=True)
            expected = torch.ones(1, 4)
            safetensors.torch.save_file({'clip_l': expected}, str(nested / 'custom.safetensors'))
            (nested / 'other.safetensors').write_bytes(b'unrequested')
            env = {'FOOOCUS_MODEL_ROOT': str(local), 'FOOOCUS_DRIVE_MODEL_ROOT': str(drive)}
            with patch.dict(os.environ, env):
                result = namespace['load_embed']('custom', str(local / 'embeddings'), 4, embed_key='clip_l')
            self.assertIsNotNone(result)
            torch.testing.assert_close(result, expected)
            self.assertFalse((local / 'embeddings/nested/other.safetensors').exists())

    def test_faceswap_downloader_reuses_drive_model(self):
        from extras.facexlib.utils import misc
        with tempfile.TemporaryDirectory() as tmp:
            local, drive = Path(tmp) / 'models', Path(tmp) / 'drive'
            (drive / 'controlnet').mkdir(parents=True)
            (drive / 'controlnet/face.pth').write_bytes(b'complete')
            env = {'FOOOCUS_MODEL_ROOT': str(local), 'FOOOCUS_DRIVE_MODEL_ROOT': str(drive)}
            with patch.dict(os.environ, env), patch.object(misc, 'download_url_to_file', side_effect=AssertionError('network download attempted'), create=True):
                result = misc.load_file_from_url('https://example.org/face.pth', save_dir=str(local / 'controlnet'))
            self.assertEqual(Path(result).read_bytes(), b'complete')

    def test_drive_model_is_used_without_network_download(self):
        from modules.model_loader import load_file_from_url
        with tempfile.TemporaryDirectory() as tmp:
            local, drive = Path(tmp) / 'models', Path(tmp) / 'drive'
            drive.mkdir()
            (drive / 'model').write_bytes(b'complete')
            env = {'FOOOCUS_MODEL_ROOT': str(local), 'FOOOCUS_DRIVE_MODEL_ROOT': str(drive)}
            hub = types.ModuleType('torch.hub')
            hub.download_url_to_file = lambda *a, **kw: self.fail('network download attempted')
            with patch.dict(os.environ, env), patch.dict(sys.modules, {'torch.hub': hub}):
                result = load_file_from_url('https://example.org/model', model_dir=str(local))
            self.assertEqual(Path(result).read_bytes(), b'complete')

    def test_download_failure_cannot_leave_a_complete_model(self):
        from modules.model_loader import load_file_from_url
        def fail(url, target, **kwargs):
            Path(target).write_bytes(b'partial')
            raise OSError('interrupted')
        hub = types.ModuleType('torch.hub')
        hub.download_url_to_file = fail
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'torch.hub': hub}):
            with self.assertRaises(OSError):
                load_file_from_url('https://example.org/model', model_dir=tmp, file_name='model')
            self.assertFalse((Path(tmp) / 'model').exists())


if __name__ == '__main__':
    unittest.main()
