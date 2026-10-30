"""Fooocus' boundary between its legacy UI contract and public Gradio 6 APIs.

The compatibility facade adapts only instances created through it. In particular,
modern Gradio's routing, templates and asyncio are never monkey patched.
"""
import functools
import inspect
from pathlib import Path

import gradio as _gr
import numpy as np
from PIL import Image as PILImage

MODERN = int(_gr.__version__.split('.')[0]) >= 6
all_components = []


def _array(value):
    if isinstance(value, (str, Path)):
        with PILImage.open(value) as image:
            return np.array(image)
    return np.asarray(value, dtype=np.uint8)


def _rgb(value):
    image = PILImage.fromarray(_array(value))
    if 'A' in image.getbands():
        image = PILImage.alpha_composite(PILImage.new('RGBA', image.size, 'white'), image.convert('RGBA'))
    return np.array(image.convert('RGB'))


def _mask(value):
    image = _array(value)
    if image.ndim == 3:
        # Editor layers use alpha, including black paint on transparent pixels.
        return image[..., 3] if image.shape[-1] == 4 else image[..., :3].max(axis=-1)
    return image


def editor_to_legacy(value):
    """Keep the unpainted RGB background and union each layer's alpha mask."""
    if value is None:
        return None
    if not isinstance(value, dict):
        value = {'background': value, 'layers': []}
    background = value.get('background')
    if background is None:
        background = value.get('composite')
    if background is None:
        return None
    image = _rgb(background)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for layer in value.get('layers') or []:
        if layer is None:
            continue
        layer_mask = _mask(layer)
        if layer_mask.shape != mask.shape:
            raise ValueError('Editor layer dimensions must match the background')
        mask = np.maximum(mask, layer_mask)
    return {'image': image, 'mask': np.repeat(mask[..., None], 3, axis=2)}


def legacy_to_editor(value):
    """Adapt worker image/mask outputs back to the public EditorValue contract."""
    if value is None:
        return None
    if isinstance(value, dict) and 'background' in value:
        return value
    if isinstance(value, dict):
        image = _rgb(value['image'])
        mask = _mask(value['mask'])
        if mask.shape != image.shape[:2]:
            raise ValueError('Mask dimensions must match the image')
        layer = np.full((*mask.shape, 4), 255, dtype=np.uint8)
        layer[..., 3] = mask
        return {'background': image, 'layers': [layer], 'composite': image}
    image = _rgb(value)
    return {'background': image, 'layers': [], 'composite': image}


def _event_kwargs(kwargs):
    kwargs = dict(kwargs)
    if '_js' in kwargs:
        kwargs['js'] = kwargs.pop('_js')
    if isinstance(kwargs.get('show_progress'), bool):
        kwargs['show_progress'] = 'full' if kwargs['show_progress'] else 'hidden'
    return kwargs


def _wrap_event(method):
    @functools.wraps(method)
    def event(*args, **kwargs):
        dependency = method(*args, **_event_kwargs(kwargs))
        for name in ('then', 'success', 'failure'):
            continuation = getattr(dependency, name, None)
            if continuation is not None:
                setattr(dependency, name, _wrap_event(continuation))
        return dependency
    return event


def _adapt_events(component):
    for name in ('change', 'input', 'click', 'select', 'upload', 'clear', 'edit', 'submit', 'load', 'release', 'blur', 'focus', 'stream', 'tick'):
        event = getattr(component, name, None)
        if callable(event):
            setattr(component, name, _wrap_event(event))
    all_components.append(component)
    return component


def update(**kwargs):
    if 'brush_color' in kwargs:
        color = kwargs.pop('brush_color')
        kwargs['brush'] = _gr.Brush(colors=[color], default_color=color, color_mode='fixed')
    return _gr.update(**kwargs)


class _ComponentFactory:
    def __init__(self, component):
        self.component = component

    def __call__(self, *args, **kwargs):
        # Button labels disappeared; its visible caption is already `value`.
        if self.component is _gr.Button:
            label = kwargs.pop('label', None)
            if label is not None:
                kwargs.setdefault('value', label)
        return _adapt_events(self.component(*args, **kwargs))

    def update(self, **kwargs):
        return update(**kwargs)

    def __getattr__(self, name):
        return getattr(self.component, name)


class _Facade:
    update = staticmethod(update)

    def __getattr__(self, name):
        value = getattr(_gr, name)
        if inspect.isclass(value) and hasattr(value, 'get_config'):
            factory = _ComponentFactory(value)
            setattr(self, name, factory)
            return factory
        return value


if MODERN:
    class LegacyImageEditor(_gr.ImageEditor):
        """ImageEditor template retaining Fooocus payloads after session updates."""
        is_template = True

        def preprocess(self, payload):
            return editor_to_legacy(super().preprocess(payload))

        def postprocess(self, value):
            return super().postprocess(legacy_to_editor(value))


def modern_image(*args, **kwargs):
    """Create public Image or ImageEditor with the legacy worker payload contract."""
    tool = kwargs.pop('tool', None)
    source = kwargs.pop('source', 'upload')
    brush_color = kwargs.pop('brush_color', '#FFFFFF')
    brush_radius = kwargs.pop('brush_radius', None)
    # Modern ImageEditor controls its own layer appearance and canvas sizing.
    for key in ('mask_opacity', 'shape', 'invert_colors'):
        kwargs.pop(key, None)
    kwargs['sources'] = [] if source == 'canvas' else [source]
    if tool == 'sketch':
        kwargs['image_mode'] = 'RGBA'
        kwargs['type'] = 'numpy'
        kwargs['brush'] = _gr.Brush(colors=[brush_color], default_color=brush_color,
                                     color_mode='fixed', default_size=brush_radius or 20)
        if 'value' in kwargs and not callable(kwargs['value']):
            kwargs['value'] = legacy_to_editor(kwargs['value'])
        component = LegacyImageEditor(*args, **kwargs)
    else:
        component = _gr.Image(*args, **kwargs)
    return _adapt_events(component)


if MODERN:
    gr = _Facade()
    Image = modern_image
else:
    gr = _gr
    # Import lazily: payload helpers can be tested without legacy private APIs.
    def Image(*args, **kwargs):
        from modules.gradio_hijack_legacy import Image as LegacyImage
        return LegacyImage(*args, **kwargs)
