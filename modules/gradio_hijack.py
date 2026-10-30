"""Select the public modern image adapter or the original legacy component."""
from modules.gradio_compat import MODERN

if MODERN:
    from modules.gradio_compat import Image, all_components
else:
    from modules.gradio_hijack_legacy import Image, all_components
