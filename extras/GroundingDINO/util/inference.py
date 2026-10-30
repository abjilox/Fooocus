from typing import Tuple, List

import ldm_patched.modules.model_management as model_management
from ldm_patched.modules.model_patcher import ModelPatcher
from modules.config import path_inpaint
from modules.model_loader import load_file_from_url

import numpy as np
import torch
from modules.transformers_compat import MODERN_TRANSFORMERS, grounding_outputs
from types import SimpleNamespace


class GroundingDinoModel:
    def __init__(self):
        self.config_file = 'extras/GroundingDINO/config/GroundingDINO_SwinT_OGC.py'
        self.model = None
        self.load_device = torch.device('cpu')
        self.offload_device = torch.device('cpu')

    @torch.no_grad()
    @torch.inference_mode()
    def predict_with_caption(
            self,
            image: np.ndarray,
            caption: str,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25
    ) -> Tuple[object, torch.Tensor, torch.Tensor, List[str]]:
        if MODERN_TRANSFORMERS:
            return self._predict_transformers(image, caption, box_threshold, text_threshold)
        from groundingdino.util.inference import Model, load_model
        if self.model is None:
            filename = load_file_from_url(
                url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                file_name='groundingdino_swint_ogc.pth',
                model_dir=path_inpaint)
            model = load_model(model_config_path=self.config_file, model_checkpoint_path=filename)

            self.load_device = model_management.text_encoder_device()
            self.offload_device = model_management.text_encoder_offload_device()

            model.to(self.offload_device)

            self.model = ModelPatcher(model, load_device=self.load_device, offload_device=self.offload_device)

        model_management.load_model_gpu(self.model)

        processed_image = Model.preprocess_image(image_bgr=image).to(self.load_device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.load_device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, boxes, logits, phrases

    def _predict_transformers(self, image, caption, box_threshold, text_threshold):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from modules.colab_support import stage_hf_cache, persist_hf_cache
        if self.model is None:
            stage_hf_cache()
            model_id = 'IDEA-Research/grounding-dino-tiny'
            self.processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id, disable_custom_kernels=True).eval()
            persist_hf_cache()
            self.load_device = model_management.text_encoder_device()
            self.offload_device = model_management.text_encoder_offload_device()
            model.to(self.offload_device)
            self.model = ModelPatcher(model, load_device=self.load_device, offload_device=self.offload_device)
        model_management.load_model_gpu(self.model)
        caption = caption.lower().strip()
        if not caption.endswith('.'):
            caption += '.'
        # The legacy preprocessing accepts BGR arrays. HF processors accept RGB.
        inputs = self.processor(images=image[:, :, ::-1].copy(), text=caption,
                                return_tensors='pt').to(self.load_device)
        outputs = self.model.model(**inputs)
        boxes, scores, phrases = grounding_outputs(
            outputs, inputs['input_ids'], self.processor.tokenizer, box_threshold, text_threshold)
        height, width = image.shape[:2]
        xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2), dim=-1)
        xyxy *= torch.tensor([width, height, width, height])
        detections = SimpleNamespace(xyxy=xyxy.numpy(), confidence=scores.numpy(), class_id=None)
        return detections, boxes, scores, phrases


def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    from groundingdino.util.inference import preprocess_caption, get_phrases_from_posmap
    caption = preprocess_caption(caption=caption)

    # override to use model wrapped by patcher
    model = model.model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]

    return boxes, logits.max(dim=1)[0], phrases


default_groundingdino = GroundingDinoModel().predict_with_caption
