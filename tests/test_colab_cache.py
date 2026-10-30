import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.requirements = self.root / 'requirements.txt'
        self.requirements.write_text('numpy>=2.1,<2.5\n')
        self.native = {'numpy': '2.1.3', 'torch': '2.11.0+cu130', 'gradio': '6.30.0'}

    def cache(self):
        from modules import colab_cache
        return colab_cache

    def publish(self):
        package_dir = self.root / 'built'
        package_dir.mkdir()
        (package_dir / 'addon.py').write_text('value = 42\n')
        manifest = self.cache().new_manifest(self.requirements, self.native, {'numpy': '2.1.3'}, {'addon': '1.0'})
        return self.cache().publish_cache(package_dir, self.root / 'drive', manifest)

    def test_archive_restores_then_local_start_uses_no_archive_or_installer(self):
        self.publish()
        target = self.root / 'local'
        with patch('subprocess.run', side_effect=AssertionError('installer started')):
            self.cache().restore_cache(self.requirements, target, self.root / 'drive', self.native)
            self.assertEqual((target / 'addon.py').read_text(), 'value = 42\n')
            with patch('tarfile.open', side_effect=AssertionError('archive opened twice')):
                self.cache().restore_cache(self.requirements, target, self.root / 'drive', self.native)

    def test_missing_cache_never_installs_and_explains_setup(self):
        with patch('subprocess.run', side_effect=AssertionError('installer started')):
            with self.assertRaisesRegex(self.cache().CacheError, 'prepare'):
                self.cache().restore_cache(self.requirements, self.root / 'local', self.root / 'missing', self.native)

    def test_changed_native_dependency_invalidates_cache(self):
        self.publish()
        changed = dict(self.native, numpy='2.2.0')
        with self.assertRaises(self.cache().CacheError):
            self.cache().restore_cache(self.requirements, self.root / 'local', self.root / 'drive', changed)

    def test_unrelated_base_package_change_does_not_invalidate(self):
        self.publish()
        changed = dict(self.native, unrelated='9.0')
        self.cache().restore_cache(self.requirements, self.root / 'local', self.root / 'drive', changed)
        self.assertTrue((self.root / 'local/addon.py').exists())

    def test_requirements_change_requires_new_cache(self):
        self.publish()
        self.requirements.write_text('numpy>=2.1,<2.5\naddon==2.0\n')
        with self.assertRaises(self.cache().CacheError):
            self.cache().restore_cache(self.requirements, self.root / 'local', self.root / 'drive', self.native)

    def test_damaged_archive_never_publishes_local_environment(self):
        manifest = self.publish()
        (self.root / 'drive' / manifest['archive']).write_bytes(b'broken')
        with self.assertRaisesRegex(self.cache().CacheError, 'checksum'):
            self.cache().restore_cache(self.requirements, self.root / 'local', self.root / 'drive', self.native)
        self.assertFalse((self.root / 'local').exists())

    def test_interrupted_rebuild_retains_previous_archive_and_manifest(self):
        manifest = self.publish()
        old = (self.root / 'drive' / (manifest['key'] + '.json')).read_bytes()
        (self.root / 'built/addon.py').write_text('value = 100\n')
        with patch('tarfile.open', side_effect=OSError('Drive disconnected')):
            with self.assertRaises(OSError):
                self.cache().publish_cache(self.root / 'built', self.root / 'drive', manifest)
        self.assertEqual((self.root / 'drive' / (manifest['key'] + '.json')).read_bytes(), old)
        self.assertTrue((self.root / 'drive' / manifest['archive']).exists())

    def test_existing_user_directory_is_not_replaced(self):
        self.publish()
        target = self.root / 'local'
        target.mkdir()
        (target / 'personal.txt').write_text('keep')
        with self.assertRaises(self.cache().CacheError):
            self.cache().restore_cache(self.requirements, target, self.root / 'drive', self.native)
        self.assertEqual((target / 'personal.txt').read_text(), 'keep')

    def test_fresh_runtime_start_restores_cache_without_setup_state_or_pip(self):
        from modules import colab_runtime
        repo, drive = self.root / 'repo', self.root / 'drive/Fooocus'
        repo.mkdir()
        req = repo / 'requirements_colab.txt'
        req.write_text(self.requirements.read_text())
        packages = self.root / 'built'
        packages.mkdir()
        (packages / 'addon.py').write_text('value = 42\n')
        manifest = self.cache().new_manifest(req, self.native, {'numpy': '2.1.3'}, {'addon': '1.0'})
        self.cache().publish_cache(packages, drive / 'cache/python', manifest)
        target, state = self.root / 'deps', self.root / 'new-state.json'
        with patch('modules.colab_cache.installed_packages', return_value=self.native), \
                patch('subprocess.run', side_effect=AssertionError('installer called')), \
                patch('subprocess.check_output', return_value='abc123'), \
                patch('modules.colab_runtime.run_live') as launch:
            colab_runtime.start(state, repo=repo, drive_root=drive, dependency_dir=target)
        self.assertTrue(state.is_file())
        self.assertTrue((target / 'addon.py').is_file())
        self.assertEqual(launch.call_args.kwargs['env']['PYTHONPATH'].split(__import__('os').pathsep)[0], str(target))
        self.assertIn('launch.py', launch.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
