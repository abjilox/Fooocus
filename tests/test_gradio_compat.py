"""Contracts at Fooocus' public Gradio boundary (also run on legacy profiles)."""
import unittest
import numpy as np
from modules import gradio_compat as compat


class EditorContractTests(unittest.TestCase):
    def test_empty_editor_has_no_image(self):
        self.assertIsNone(compat.editor_to_legacy(None))
        self.assertIsNone(compat.editor_to_legacy({'background': None, 'layers': [], 'composite': None}))

    def test_alpha_layers_are_unioned_without_using_the_composite(self):
        background = np.full((2, 2, 4), [12, 23, 34, 255], dtype=np.uint8)
        first = np.zeros((2, 2, 4), dtype=np.uint8)
        second = first.copy()
        first[0, 0] = [0, 0, 0, 128]  # Black paint still selects the mask.
        second[1, 1] = [255, 255, 255, 255]
        value = compat.editor_to_legacy({'background': background, 'layers': [first, second], 'composite': first})
        np.testing.assert_array_equal(value['image'], [[[12, 23, 34]] * 2] * 2)
        np.testing.assert_array_equal(value['mask'][..., 0], [[128, 0], [0, 255]])
        self.assertEqual(value['mask'].dtype, np.uint8)

    def test_rgba_background_is_composited_on_white(self):
        background = np.array([[[99, 12, 44, 0], [0, 0, 0, 128]]], dtype=np.uint8)
        result = compat.editor_to_legacy({'background': background, 'layers': []})
        np.testing.assert_array_equal(result['image'], [[[255, 255, 255], [127, 127, 127]]])
        np.testing.assert_array_equal(result['mask'], np.zeros((1, 2, 3), dtype=np.uint8))

    def test_empty_layers_produce_zero_mask_and_plain_images_roundtrip(self):
        image = np.full((2, 3, 3), 120, dtype=np.uint8)
        result = compat.editor_to_legacy(compat.legacy_to_editor(image))
        np.testing.assert_array_equal(result['image'], image)
        np.testing.assert_array_equal(result['mask'], np.zeros((2, 3, 3), dtype=np.uint8))

    def test_legacy_rgb_mask_roundtrips_to_alpha_layer(self):
        image = np.full((2, 2, 3), 20, dtype=np.uint8)
        mask = np.array([[0, 255], [0, 128]], dtype=np.uint8)
        result = compat.editor_to_legacy(compat.legacy_to_editor({'image': image, 'mask': np.repeat(mask[..., None], 3, axis=2)}))
        np.testing.assert_array_equal(result['mask'][..., 0], mask)

    def test_mask_retains_worker_rgb_channel_contract(self):
        result = compat.editor_to_legacy({'background': np.zeros((2, 3, 3), dtype=np.uint8), 'layers': []})
        self.assertEqual(result['mask'].shape, (2, 3, 3))
        # These are the existing async_worker inpaint and advanced-mask consumers.
        self.assertEqual(result['mask'][:, :, 0].shape, (2, 3))
        self.assertEqual(np.maximum(result['image'], result['mask']).shape, (2, 3, 3))

    def test_shape_mismatch_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            compat.editor_to_legacy({'background': np.zeros((2, 2, 3), dtype=np.uint8), 'layers': [np.zeros((3, 3, 4), dtype=np.uint8)]})


@unittest.skipUnless(compat.MODERN, 'Gradio 6 integration contract')
class ModernUIContracts(unittest.TestCase):
    def test_worker_payload_survives_gradio_component_updates(self):
        import asyncio
        from gradio.state_holder import SessionState
        gr = compat.gr
        with gr.Blocks() as app:
            image = compat.Image(tool='sketch', source='upload', brush_color='#FFFFFF')
            button = gr.Button('Change brush')
            button.click(lambda: gr.update(brush_color='#000000'), outputs=image)
        pixels = np.full((2, 2, 3), 27, dtype=np.uint8)
        state = SessionState(app)
        asyncio.run(app.process_api(0, [], state=state))
        current = state[image._id]
        result = current.preprocess(current.postprocess(pixels))
        self.assertIn('image', result)
        np.testing.assert_array_equal(result['image'], pixels)
        np.testing.assert_array_equal(result['mask'], np.zeros((2, 2, 3), dtype=np.uint8))
        app.close()

    def test_complete_ui_builds_without_model_downloads(self):
        import subprocess
        import sys
        from textwrap import dedent
        code = dedent("""
            import sys, types, os, tempfile, asyncio
            from unittest.mock import patch
            sys.path.insert(0, os.getcwd())
            sys.argv = ['fooocus-ui-smoke', '--disable-in-browser']
            import modules

            def stub(name, **attrs):
                module = types.ModuleType(name)
                module.__dict__.update(attrs)
                sys.modules[name] = module
                if name.startswith('modules.'):
                    setattr(modules, name.split('.')[-1], module)
                return module

            class AsyncTask:
                def __init__(self, args): self.args = args
            stub('modules.model_loader', load_file_from_url=lambda *a, **k: None)
            stub('modules.async_worker', AsyncTask=AsyncTask)
            stub('modules.util', is_json=lambda x: False)
            stub('modules.meta_parser', load_parameter_button_click=lambda *a: None)
            stub('modules.private_logger', get_current_html_path=lambda *a: '/tmp/log.html')
            stub('extras.inpaint_mask', SAMOptions=type('SAMOptions', (), {}))
            stub('launch', download_models=lambda *a, **k: None)
            with tempfile.TemporaryDirectory() as folder:
                os.environ['config_path'] = folder + '/config.json'
                os.environ['config_example_path'] = folder + '/example.json'
                os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
                import gradio
                original_wait_for = asyncio.wait_for
                original_template = gradio.routes.templates.TemplateResponse
                with patch.object(gradio.Blocks, 'launch') as launch:
                    import webui, shared
                assert asyncio.wait_for is original_wait_for
                assert gradio.routes.templates.TemplateResponse == original_template
                config = shared.gradio_root.get_config_file()
                assert len(config['components']) > 300
                assert len(config['dependencies']) > 70
                assert {'head', 'css'} <= launch.call_args.kwargs.keys()
                assert all(not p.endswith('Fooocus') for p in launch.call_args.kwargs['allowed_paths'])
                assert sum(c['type'] == 'imageeditor' for c in config['components']) == 2
                from fastapi import FastAPI
                from fastapi.testclient import TestClient
                from modules.ui_gradio_extensions import webpath, static_paths
                mounted = gradio.mount_gradio_app(
                    FastAPI(), shared.gradio_root, '/', allowed_paths=static_paths(),
                    head=launch.call_args.kwargs['head'], css=launch.call_args.kwargs['css'])
                with TestClient(mounted) as client:
                    assert client.get(webpath('javascript/script.js')).status_code == 200
                    assert client.get(webpath('sdxl_styles/samples/fooocus_v2.jpg')).status_code == 200
                    assert client.get('/gradio_api/file=' + os.path.abspath('config.txt')).status_code == 403
                    assert webpath('javascript/script.js') in client.get('/config').json()['head']
        """)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_public_components_events_and_updates_build(self):
        gr = compat.gr
        with gr.Blocks() as app:
            image = compat.Image(tool='sketch', source='upload', brush_color='#FFFFFF')
            button = gr.Button(label='Generate', value='Generate')
            dataset = gr.Dataset(components=[gr.Textbox()], samples=[['one']])
            button.click(lambda x: x, image, image, _js='(x) => x', show_progress=False).then(lambda: gr.Dataset.update(samples=[['two']]), outputs=dataset, show_progress=True)
        config = app.get_config_file()
        self.assertEqual(config['dependencies'][0]['show_progress'], 'hidden')
        self.assertEqual(config['dependencies'][1]['show_progress'], 'full')
        self.assertEqual(config['dependencies'][0]['js'], '(x) => x')
        self.assertEqual(image.get_config()['brush']['colors'], ['#FFFFFF'])
        self.assertEqual(gr.Dataset.update(samples=[['two']])['samples'], [['two']])
        self.assertTrue(any(item['type'] == 'imageeditor' for item in config['components']))


if __name__ == '__main__':
    unittest.main()
