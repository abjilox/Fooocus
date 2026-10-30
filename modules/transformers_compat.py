"""Compatibility boundaries for Fooocus checkpoints and Transformers 4/5.

Keep the legacy state-dict layout: Transformers 5 flattened CLIP's text
encoder, while Fooocus checkpoints and textual inversion use text_model.*.
"""
import torch
import transformers
from transformers import CLIPTextModel, PreTrainedModel as HFPreTrainedModel
from transformers.generation import GenerationMixin
from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer

MODERN_TRANSFORMERS = int(transformers.__version__.split('.')[0]) >= 5
try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices
except ImportError:
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        # Transformers 4's utility, retained for the vendored BLIP architecture.
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head -= sum(h < head for h in already_pruned_heads)
            mask[head] = 0
        index = torch.arange(mask.numel())[mask.view(-1).eq(1)].long()
        return heads, index


if MODERN_TRANSFORMERS:
    class FooocusCLIPTextModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            config._attn_implementation = 'eager'
            self.text_model = CLIPTextModel(config)
            self.config = config

        def get_input_embeddings(self):
            return self.text_model.embeddings.token_embedding

        def set_input_embeddings(self, embeddings):
            self.text_model.embeddings.token_embedding = embeddings

        def forward(self, *args, **kwargs):
            return self.text_model(*args, **kwargs)
else:
    FooocusCLIPTextModel = CLIPTextModel


class PreTrainedModel(HFPreTrainedModel):
    """Mask helpers removed in v5, required by the vendored BLIP encoders."""
    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError('Expected a 2D or 3D encoder attention mask')
        return (1.0 - mask.to(self.dtype)) * torch.finfo(self.dtype).min

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask[None, None, :, None, None].expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask[:, None, :, None, None]
        if head_mask.dim() != 5:
            raise ValueError('Head mask must have 1, 2 or 5 dimensions')
        head_mask = head_mask.to(self.dtype)
        return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask


# PreTrainedModel already inherits GenerationMixin on older Transformers.
class LegacyGenerationMixin(GenerationMixin if MODERN_TRANSFORMERS else object):
    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False


def grounding_outputs(outputs, input_ids, tokenizer, box_threshold, text_threshold):
    """HF DINO logits and normalized cxcywh -> original inference contract."""
    probabilities = outputs.logits.detach().float().cpu().sigmoid()[0]
    keep = probabilities.max(dim=-1).values > box_threshold
    probabilities = probabilities[keep]
    boxes = outputs.pred_boxes.detach().float().cpu()[0][keep]
    token_ids = input_ids.detach().cpu()[0]
    phrases = []
    for probability in probabilities:
        selected = probability[:len(token_ids)] > text_threshold
        # Match GroundingDINO's exclusion of CLS and terminal tokens.
        selected[0] = False
        selected[-1] = False
        phrases.append(tokenizer.decode(token_ids[selected].tolist()).replace('.', '').strip())
    return boxes, probabilities.max(dim=-1).values, phrases


if MODERN_TRANSFORMERS:
    from transformers import CLIPVisionModel

    class FooocusCLIPVisionModel(torch.nn.Module):
        """Preserve safety checker vision_model.vision_model.* checkpoint keys."""
        def __init__(self, config):
            super().__init__()
            config._attn_implementation = 'eager'
            self.vision_model = CLIPVisionModel(config)

        def forward(self, *args, **kwargs):
            return self.vision_model(*args, **kwargs)
else:
    from transformers import CLIPVisionModel as FooocusCLIPVisionModel


def load_expansion_model(directory):
    from transformers import AutoModelForCausalLM
    # The shipped Fooocus expansion checkpoint is a tensor-only pytorch_model.bin.
    # Transformers 5 safely reads it on Colab's torch >= 2.6; do not bypass the
    # loader's security check or opt into arbitrary pickle deserialization.
    kwargs = {'weights_only': True} if MODERN_TRANSFORMERS else {}
    return AutoModelForCausalLM.from_pretrained(directory, **kwargs).eval()
