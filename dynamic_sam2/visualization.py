#visualization.py
import cv2
import numpy as np
import random
import supervision as sv
from pathlib import Path
import logging
from typing import List, Optional

class VisualizationManager:
    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
        self.palette = self._create_color_palette()
        self.stable_id_to_color = {}

    def _create_color_palette(self, n=200, seed=42):
        random.seed(seed)
        unique_colors = set()
        colors = []
        while len(colors) < n:
            color = "#{:06X}".format(random.randint(0, 0xFFFFFF))
            if color not in unique_colors:
                unique_colors.add(color)
                colors.append(color)
        return sv.ColorPalette.from_hex(colors)

    def _get_color_for_id(self, stable_id):
        if stable_id not in self.stable_id_to_color:
            self.stable_id_to_color[stable_id] = len(self.stable_id_to_color) % len(self.palette)
        return self.stable_id_to_color[stable_id]

    def visualize_frame(self, frame_path, boxes, masks, labels, stable_ids):
        frame_name = Path(frame_path).stem
        #self.logger.debug(f"\nVisualizing frame {frame_name}")
        
        if boxes is None or len(boxes) == 0:
            self.logger.debug("No objects to visualize")
            return
            
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            self.logger.warning(f"Could not read frame {frame_path}")
            return

        class_ids = [self._get_color_for_id(sid) for sid in stable_ids]
        
        box_annotator = sv.BoxAnnotator(color=self.palette)
        mask_annotator = sv.MaskAnnotator(color=self.palette)
        label_annotator = sv.LabelAnnotator(color=self.palette)

        detections = sv.Detections(
            xyxy=boxes,
            mask=(masks.astype(bool) if masks is not None else None),
            class_id=np.array(class_ids)
        )

        annotated = frame_bgr.copy()
        if masks is not None:
            annotated = mask_annotator.annotate(scene=annotated, detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=[f"{l} (ID:{sid})" for l, sid in zip(labels, stable_ids)]
        )

        out_path = self.output_dir / f"{frame_name}_tracked.jpg"
        cv2.imwrite(str(out_path), annotated)