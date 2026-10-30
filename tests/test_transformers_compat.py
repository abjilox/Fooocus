"""Small, offline model contracts; no pretrained weights are downloaded."""
import unittest
import torch


class TransformersCompatibilityTests(unittest.TestCase):
    def test_pruning_preserves_selected_head_weights(self):
        from modules.transformers_compat import find_pruneable_heads_and_indices, prune_linear_layer
        layer = torch.nn.Linear(6, 6)
        heads, index = find_pruneable_heads_and_indices([1], 3, 2, set())
        pruned = prune_linear_layer(layer, index)
        self.assertEqual(heads, {1})
        torch.testing.assert_close(pruned.weight, layer.weight[[0, 1, 4, 5]])

    def test_grounding_results_keep_normalized_cpu_cxcywh(self):
        from modules.transformers_compat import grounding_outputs
        from types import SimpleNamespace
        outputs = SimpleNamespace(logits=torch.tensor([[[-8., 3., -8.], [-8., -8., -8.]]]),
                                  pred_boxes=torch.tensor([[[.5, .4, .2, .3], [.1, .1, .1, .1]]]))
        class Tokenizer:
            def decode(self, ids):
                return 'cat.' if ids == [12] else ''
        boxes, scores, phrases = grounding_outputs(outputs, torch.tensor([[11, 12, 13]]), Tokenizer(), .3, .25)
        torch.testing.assert_close(boxes, outputs.pred_boxes[0, :1])
        self.assertEqual(boxes.device.type, 'cpu')
        self.assertEqual(phrases, ['cat'])
        self.assertAlmostEqual(scores.item(), torch.sigmoid(torch.tensor(3.)).item())

    def test_tiny_clip_retains_legacy_checkpoint_keys_and_hidden_states(self):
        from modules.transformers_compat import FooocusCLIPTextModel, no_init_weights
        from transformers import CLIPTextConfig
        config = CLIPTextConfig(vocab_size=20, hidden_size=8, intermediate_size=16,
                                num_attention_heads=2, num_hidden_layers=2, max_position_embeddings=8,
                                bos_token_id=1, eos_token_id=2, pad_token_id=0)
        with no_init_weights():
            model = FooocusCLIPTextModel(config)
        self.assertIn('text_model.embeddings.token_embedding.weight', model.state_dict())
        model = FooocusCLIPTextModel(config).eval()
        output = model(input_ids=torch.tensor([[1, 3, 2]]), output_hidden_states=True, return_dict=True)
        self.assertEqual(output.last_hidden_state.shape, (1, 3, 8))
        self.assertEqual(len(output.hidden_states), 3)
        self.assertEqual(output.pooler_output.shape, (1, 8))

    def test_tiny_blip_decoder_forward_and_generation(self):
        from extras.BLIP.models.med import BertConfig, BertLMHeadModel
        config = BertConfig(vocab_size=20, hidden_size=8, intermediate_size=16,
                            num_attention_heads=2, num_hidden_layers=1, encoder_width=8, add_cross_attention=True)
        model = BertLMHeadModel(config).eval()
        inputs = torch.tensor([[1, 3]])
        encoder = torch.zeros(1, 2, 8)
        output = model(inputs, encoder_hidden_states=encoder, is_decoder=True)
        self.assertEqual(output.logits.shape, (1, 2, 20))
        result = model.generate(inputs, encoder_hidden_states=encoder, max_new_tokens=2, num_beams=3,
                                eos_token_id=2, pad_token_id=0)
        self.assertGreater(result.shape[1], 2)

    def test_tiny_safety_checker_keeps_original_vision_checkpoint_layout(self):
        from extras.safety_checker.models.safety_checker import StableDiffusionSafetyChecker
        from transformers import CLIPConfig, CLIPVisionConfig
        import numpy as np
        vision = CLIPVisionConfig(hidden_size=8, intermediate_size=16, num_attention_heads=2,
                                  num_hidden_layers=1, image_size=8, patch_size=4)
        config = CLIPConfig(vision_config=vision.to_dict(), projection_dim=8)
        model = StableDiffusionSafetyChecker(config).eval()
        self.assertIn('vision_model.vision_model.embeddings.patch_embedding.weight', model.state_dict())
        images, flags = model(torch.zeros(1, 3, 8, 8), np.zeros((1, 8, 8, 3)))
        self.assertEqual(images.shape, (1, 8, 8, 3))
        self.assertEqual(len(flags), 1)

    def test_tiny_gpt2_bin_weights_load_and_generate(self):
        from modules.transformers_compat import load_expansion_model
        from transformers import GPT2Config, GPT2LMHeadModel
        import tempfile
        config = GPT2Config(vocab_size=20, n_embd=8, n_layer=1, n_head=2,
                            bos_token_id=1, eos_token_id=2, pad_token_id=0)
        original = GPT2LMHeadModel(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory, safe_serialization=False)
            model = load_expansion_model(directory)
            torch.testing.assert_close(model.get_input_embeddings().weight, original.get_input_embeddings().weight)
            result = model.generate(torch.tensor([[1, 3]]), max_new_tokens=2, do_sample=False)
            self.assertGreater(result.shape[1], 2)

    def test_grounding_empty_detections_keep_tensor_shapes(self):
        from modules.transformers_compat import grounding_outputs
        from types import SimpleNamespace
        outputs = SimpleNamespace(logits=torch.full((1, 2, 3), -20.), pred_boxes=torch.zeros(1, 2, 4))
        boxes, scores, phrases = grounding_outputs(outputs, torch.tensor([[1, 2, 3]]), None, .9, .5)
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(phrases, [])

    def test_modern_grounding_dino_tiny_forward_without_custom_kernels(self):
        from modules.transformers_compat import MODERN_TRANSFORMERS
        if not MODERN_TRANSFORMERS:
            self.skipTest('Transformers 5 GroundingDINO backend')
        from transformers import BertConfig, SwinConfig, GroundingDinoConfig, AutoModelForZeroShotObjectDetection
        backbone = SwinConfig(image_size=64, embed_dim=16, depths=[1, 1, 1, 1],
                              num_heads=[1, 2, 4, 8], window_size=2, out_indices=[2, 3, 4])
        text = BertConfig(vocab_size=110, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=1, num_attention_heads=4)
        config = GroundingDinoConfig(backbone_config=backbone, text_config=text, d_model=32,
                                     encoder_layers=1, decoder_layers=2, encoder_attention_heads=4,
                                     decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
                                     num_queries=2, max_text_len=8, disable_custom_kernels=True)
        model = AutoModelForZeroShotObjectDetection.from_config(config).eval()
        with torch.no_grad():
            output = model(pixel_values=torch.zeros(1, 3, 128, 128),
                           input_ids=torch.tensor([[101, 10, 102]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
        self.assertEqual(output.pred_boxes.shape, (1, 2, 4))
        self.assertTrue(torch.isfinite(output.pred_boxes).all())

    def test_mask_module_import_does_not_require_optional_backends(self):
        import builtins
        import importlib.util
        import sys
        from pathlib import Path
        from types import ModuleType
        from unittest.mock import patch
        original_import = builtins.__import__
        forbidden = ('rembg', 'onnxruntime', 'segment_anything', 'extras.sam', 'extras.GroundingDINO', 'groundingdino')
        def without_optional(name, *args, **kwargs):
            if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden):
                raise ImportError('Optional mask backend unavailable: ' + name)
            return original_import(name, *args, **kwargs)
        path = Path(__file__).resolve().parents[1] / 'extras/inpaint_mask.py'
        spec = importlib.util.spec_from_file_location('mask_import_contract', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'modules.config': ModuleType('modules.config')}), patch('builtins.__import__', without_optional):
            spec.loader.exec_module(module)
            self.assertEqual(module.generate_mask_from_image(None), (None, 0, 0, 0))
            import numpy as np
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('error', FutureWarning)
                with self.assertRaisesRegex(ImportError, 'Optional mask backend unavailable: rembg'):
                    module.generate_mask_from_image(np.zeros((2, 2, 3), dtype=np.uint8),
                                                    mask_model='u2net', sam_options=None)

    def test_actual_patched_clip_loads_checkpoint_and_encodes_on_cpu(self):
        """Exercise Fooocus manual-cast ops and the real patched SD CLIP path."""
        import subprocess
        import sys
        import textwrap
        from pathlib import Path
        script = textwrap.dedent('''
            import sys
            sys.argv = ['test', '--always-cpu', '--disable-xformers', '--attention-pytorch']
            import args_manager
            import tempfile
            from pathlib import Path
            import torch
            from transformers import CLIPTextConfig
            from modules.transformers_compat import FooocusCLIPTextModel
            import modules.patch_clip
            import ldm_patched.modules.sd1_clip
            modules.patch_clip.patch_all_clip()
            config = CLIPTextConfig(vocab_size=20, hidden_size=8, intermediate_size=16,
                                    num_attention_heads=2, num_hidden_layers=2,
                                    max_position_embeddings=8, bos_token_id=1,
                                    eos_token_id=2, pad_token_id=0)
            reference = FooocusCLIPTextModel(config).eval()
            with tempfile.TemporaryDirectory() as directory:
                filename = Path(directory) / 'clip.json'
                config.to_json_file(str(filename))
                model = ldm_patched.modules.sd1_clip.SDClipModel(
                    max_length=8, textmodel_json_config=str(filename), dtype=torch.float32,
                    special_tokens={'start': 1, 'end': 2, 'pad': 0})
                model.transformer.load_state_dict(reference.state_dict(), strict=True)
                with torch.no_grad():
                    expected = reference(input_ids=torch.tensor([[1, 3, 2]]), return_dict=True)
                    hidden, pooled = model([[1, 3, 2]])
                assert hidden.shape == (1, 3, 8), hidden.shape
                assert pooled.shape == (1, 8), pooled.shape
                torch.testing.assert_close(hidden, expected.last_hidden_state)
                torch.testing.assert_close(pooled, expected.pooler_output)
        ''')
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
