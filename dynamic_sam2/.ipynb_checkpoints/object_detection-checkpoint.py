# object_detection.py
import cv2
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.ops import box_convert
from typing import Tuple, Optional

class BaseObjectDetectionModel:
    def detect(self, image_path: Path, text_prompt: str):
        raise NotImplementedError()

class DinoDetectionModel(BaseObjectDetectionModel):
    def __init__(
        self,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        dino_cfg_path: str = "../grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        dino_ckpt_path: str = "../gdino_checkpoints/groundingdino_swint_ogc.pth"
    ):
        super().__init__()
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        from grounding_dino.groundingdino.util.inference import load_model
        self.detector = load_model(
            model_config_path=dino_cfg_path,
            model_checkpoint_path=dino_ckpt_path,
            device=self.device
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect(self, image_path: Path, text_prompt: str):
        from grounding_dino.groundingdino.util.inference import predict
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return None, None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        img_tensor = self.transform(img_rgb).to(self.device)

        boxes, confidences, labels = predict(
            model=self.detector,
            image=img_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        if boxes is None or len(boxes) == 0:
            return None, None

        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

        return boxes, labels