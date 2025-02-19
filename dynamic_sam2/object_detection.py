# object_detection.py
import cv2
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.ops import box_convert
from typing import Tuple, Optional, List

class BaseObjectDetectionModel:
    def detect(self, image_path: Path) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[float]]]:
        """
        Detect objects in an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (bounding boxes in xyxy format, class labels, confidence scores)
        """
        raise NotImplementedError()

class DinoDetectionModel(BaseObjectDetectionModel):
    def __init__(
        self,
        text_prompt: str,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        dino_cfg_path: str = "../grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        dino_ckpt_path: str = "../gdino_checkpoints/groundingdino_swint_ogc.pth"
    ):
        super().__init__()
        self.text_prompt = text_prompt.lower()
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

    def detect(self, image_path: Path) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[float]]]:
        """
        Detect objects in an image using Grounding DINO
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (bounding boxes in xyxy format, class labels, confidence scores)
        """
        from grounding_dino.groundingdino.util.inference import predict
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return None, None, None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        img_tensor = self.transform(img_rgb).to(self.device)

        boxes, confidences, labels = predict(
            model=self.detector,
            image=img_tensor,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        if boxes is None or len(boxes) == 0:
            return None, None, None

        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        confidences = confidences.cpu().numpy().tolist()

        return boxes, labels, confidences

class YOLODetectionModel(BaseObjectDetectionModel):
    def __init__(
        self,
        model_path: str,
        target_categories: Optional[List[str]] = None,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        super().__init__()
        self.target_categories = target_categories if target_categories is not None else []
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Import the YOLO model (assuming ultralytics is installed)
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except ImportError:
            raise ImportError(
                "Could not import ultralytics. "
                "Please install it with: pip install ultralytics"
            )

    def _get_class_names(self) -> List[str]:
        """Return the class names from the loaded model"""
        # The model.names attribute contains a dictionary of class indices to names
        return self.model.names if hasattr(self.model, 'names') else []
        
    def _filter_results_by_categories(
        self, 
        results, 
    ) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Filter YOLOv8 results by the target categories
        
        Returns:
            Tuple of (bounding boxes, class labels, confidence scores)
        """
        filtered_boxes = []
        filtered_labels = []
        filtered_confidences = []
        class_names = self._get_class_names()
        
        # Get the detection results
        boxes = results[0].boxes
        
        # If no target categories specified, return all detections
        if not self.target_categories:
            all_boxes = []
            all_labels = []
            all_confidences = []
            for i, (box, cls_id, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
                if conf < self.conf_threshold:
                    continue
                    
                conf_value = conf.item() if hasattr(conf, 'item') else float(conf)
                class_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"class_{int(cls_id)}"
                all_boxes.append(box.cpu().numpy())
                all_labels.append(class_name)
                all_confidences.append(conf_value)
                
            if not all_boxes:
                return None, None, None
                
            return np.array(all_boxes), all_labels, all_confidences
        
        # Otherwise filter by target categories
        categories = [cat.lower() for cat in self.target_categories]
        
        # Process each detected object
        for i, (box, cls_id, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
            if conf < self.conf_threshold:
                continue
                
            # Get the class name for this detection
            class_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"class_{int(cls_id)}"
            class_name = class_name.lower()
            conf_value = conf.item() if hasattr(conf, 'item') else float(conf)
            
            # Check if this class matches any in our target categories
            for category in categories:
                if category in class_name or class_name in category:
                    filtered_boxes.append(box.cpu().numpy())
                    filtered_labels.append(class_name)
                    filtered_confidences.append(conf_value)
                    break
                    
        if not filtered_boxes:
            return None, None, None
            
        return np.array(filtered_boxes), filtered_labels, filtered_confidences

    def detect(
        self, 
        image_path: Path, 
    ) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[float]]]:
        """
        Detect objects in an image that match the target categories using YOLOv8
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (bounding boxes in xyxy format, class labels, confidence scores)
        """
        # Convert string categories to list if needed
        if isinstance(self.target_categories, str):
            categories = [cat.strip() for cat in self.target_categories.split(',')]
        else:
            categories = self.target_categories
            
        # Check if file exists
        if not Path(image_path).exists():
            return None, None, None
            
        # Run inference
        results = self.model(str(image_path), conf=self.conf_threshold, iou=self.iou_threshold)
        
        # No detections case
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, None, None
            
        # Filter results based on target categories
        return self._filter_results_by_categories(results)