"""Dependency constraints must accept the Colab image we support."""
from pathlib import Path
import unittest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


class NativeColabProfileTests(unittest.TestCase):
    def requirements(self):
        profile = Path(__file__).resolve().parents[1] / 'requirements_colab.txt'
        return {canonicalize_name(req.name): req for line in profile.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')
                for req in [Requirement(line)]}

    def test_existing_native_runtime_packages_satisfy_profile(self):
        native = {'torch': '2.11.0+cu130', 'torchvision': '0.26.0+cu130',
                  'numpy': '2.1.3', 'transformers': '5.21.0', 'gradio': '6.30.0'}
        requirements = self.requirements()
        for name, version in native.items():
            with self.subTest(package=name, version=version):
                self.assertIn(version, requirements[name].specifier)

    def test_original_image_library_minima_remain_acceptable(self):
        requirements = self.requirements()
        for name, version in {'pillow': '10.4.0', 'scipy': '1.14.0', 'timm': '1.0.7',
                              'einops': '0.8.0', 'tqdm': '4.66.4', 'psutil': '6.0.0'}.items():
            with self.subTest(package=name, version=version):
                self.assertIn(version, requirements[name].specifier)

    def test_background_removal_uses_agreed_numpy_compatible_release(self):
        requirements = self.requirements()
        self.assertEqual(str(requirements['rembg'].specifier), '==2.0.67')
        self.assertNotIn('onnxruntime-gpu', requirements)
        self.assertNotIn('groundingdino-py', requirements)
        self.assertIn('opencv-python-headless', requirements)
        self.assertNotIn('opencv-contrib-python', requirements)
