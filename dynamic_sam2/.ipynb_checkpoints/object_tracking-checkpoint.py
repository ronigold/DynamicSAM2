# object_tracking.py
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional
from .tracking_utils import BoxUtils

class ObjectTracker:
    def __init__(self, iou_threshold: float = 0.5, logger=None):
        self.iou_threshold = iou_threshold
        self.global_object_registry = {}
        self.stable_id_counter = 0
        self.previous_positions = {}
        self.movement_history = {}
        self.box_utils = BoxUtils()
        self.logger = logger or logging.getLogger(__name__)


    def get_stable_id(self, box, threshold=0.3):
        box_key = tuple(box.astype(float).tolist())
        best_match = None
        best_match_score = threshold
    
        for ref_box, stable_id in self.global_object_registry.items():
            base_iou = self.box_utils.box_iou_xyxy(np.array(ref_box), box)
            
            adjusted_threshold = threshold
            if stable_id in self.movement_history:
                history_len = len(self.movement_history[stable_id])
                if history_len > 5:
                    adjusted_threshold *= 0.8
                    
            if base_iou > adjusted_threshold and base_iou > best_match_score:
                best_match_score = base_iou
                best_match = ref_box
    
        if best_match is not None:
            matched_id = self.global_object_registry[best_match]
            self.global_object_registry[box_key] = matched_id
            if best_match != box_key:
                del self.global_object_registry[best_match]
            return matched_id
    
        self.stable_id_counter += 1
        self.global_object_registry[box_key] = self.stable_id_counter
        return self.stable_id_counter

    def merge_detections(self, old_boxes, old_masks, old_labels, old_stable_ids,
                         new_boxes, new_labels, frame_path=None, create_masks_fn=None):
        """
        Merges object detections by:
        1. Keeping active SAM2 tracked objects (filtering out lost tracks)
        2. Adding new DINO detections only if they don't overlap with existing tracks
        3. Filtering out small objects based on bounding box size and mask area
        """
    
        self.logger.info("\n=== Starting Detection Merge ===")
    
        # Minimum thresholds
        MIN_BOX_SIZE = 15      # Minimum width or height for bounding boxes
        MIN_MASK_AREA = 500    # Minimum area for masks
    
        # Filter out lost tracks ([0,0,1,1] boxes)
        active_indices = [i for i, box in enumerate(old_boxes) if not (box[0] == 0 and box[1] == 0 and box[2] == 1 and box[3] == 1)]
    
        if active_indices:
            final_boxes = old_boxes[active_indices]
            final_masks = old_masks[active_indices]
            final_labels = [old_labels[i] for i in active_indices]
            final_stable_ids = [old_stable_ids[i] for i in active_indices]
        else:
            # Create empty arrays with proper dimensions
            final_boxes = np.empty((0, 4), dtype=np.float32)
            if old_masks.ndim == 3:  # Check if masks are 2D or 3D
                mask_shape = old_masks.shape[1:]
                final_masks = np.empty((0, *mask_shape), dtype=old_masks.dtype)
            else:
                # Handle case where masks might be 4D (batch, height, width, channels)
                mask_shape = old_masks.shape[1:]
                final_masks = np.empty((0, *mask_shape), dtype=old_masks.dtype)
            final_labels = []
            final_stable_ids = []
    
        self.logger.info(f"Active SAM2 tracks after filtering: {len(final_stable_ids)}")
        
        # Apply size filtering for SAM2 tracked objects
        valid_indices = []
        for i, box in enumerate(final_boxes):
            width, height = box[2] - box[0], box[3] - box[1]
            if width >= MIN_BOX_SIZE and height >= MIN_BOX_SIZE:
                valid_indices.append(i)
            else:
                self.logger.debug(f"Removing tracked object {final_stable_ids[i]} - box too small (Width: {width}, Height: {height})")
    
        # Keep only valid tracked objects
        if valid_indices:
            final_boxes = final_boxes[valid_indices]
            final_masks = final_masks[valid_indices]
            final_labels = [final_labels[i] for i in valid_indices]
            final_stable_ids = [final_stable_ids[i] for i in valid_indices]
    
        self.logger.info(f"Filtered SAM2 tracks remaining: {len(final_stable_ids)}")
    
        # Log active tracks
        for idx, (box, obj_id) in enumerate(zip(final_boxes, final_stable_ids)):
            self.logger.info(f"Active Track {idx+1}: ID {obj_id}, Box {box}")
    
        self.logger.info(f"DINO new detections: {len(new_boxes) if new_boxes is not None else 0}")
    
        # Log new detections
        if new_boxes is not None:
            for idx, box in enumerate(new_boxes):
                self.logger.info(f"New Detection {idx+1}: Label {new_labels[idx]}, Box {box}")
    
        # If no active tracks and no new detections, return empty
        if len(final_boxes) == 0 and (new_boxes is None or len(new_boxes) == 0):
            self.logger.info("No active tracks or new detections")
            # Return empty arrays with proper dimensions
            if old_masks.ndim == 3:
                mask_shape = old_masks.shape[1:]
                return np.empty((0, 4), dtype=np.float32), np.empty((0, *mask_shape), dtype=old_masks.dtype), [], []
            else:
                mask_shape = old_masks.shape[1:]
                return np.empty((0, 4), dtype=np.float32), np.empty((0, *mask_shape), dtype=old_masks.dtype), [], []
    
        # If no new detections, keep active tracks
        if new_boxes is None or len(new_boxes) == 0:
            self.logger.info("No new detections to process")
            return final_boxes, final_masks, final_labels, final_stable_ids
    
        # Process new DINO detections
        added_count = 0
        for i, new_box in enumerate(new_boxes):
            # Check if it overlaps with any active track
            is_overlapping = any(self.box_utils.box_iou_xyxy(old_box, new_box) > 0.3 for old_box in final_boxes) if len(final_boxes) > 0 else False
    
            # Check bounding box size
            width, height = new_box[2] - new_box[0], new_box[3] - new_box[1]
            if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
                self.logger.debug(f"Skipping detection {i+1} - box too small (Width: {width}, Height: {height})")
                continue
    
            # If no overlap, add as a new object
            if not is_overlapping:
                self.logger.debug(f"Detection {i+1} has no overlap - adding as new object")
                frame_bgr = cv2.imread(str(frame_path))
                if frame_bgr is None:
                    continue
    
                new_mask = create_masks_fn(frame_bgr, new_box[np.newaxis, :])
                if new_mask is None or len(new_mask) == 0:
                    continue
    
                # Calculate mask area and filter out small masks
                mask_area = np.sum(new_mask > 0)
                if mask_area < MIN_MASK_AREA:
                    self.logger.debug(f"Skipping detection {i+1} - mask too small (Area: {mask_area})")
                    continue
    
                # Add to final lists - handle both empty and non-empty cases
                if len(final_boxes) == 0:
                    final_boxes = new_box[np.newaxis, :]
                    if new_mask.ndim == 3:
                        final_masks = new_mask
                    else:  # if 2D
                        final_masks = new_mask[np.newaxis, ...]
                else:
                    final_boxes = np.vstack([final_boxes, new_box])
                    if new_mask.ndim == 3:
                        final_masks = np.concatenate([final_masks, new_mask], axis=0)
                    else:  # if 2D
                        final_masks = np.concatenate([final_masks, new_mask[np.newaxis, ...]], axis=0)
                
                final_labels.append(new_labels[i])
    
                # Assign a new stable ID
                new_id = self.stable_id_counter + 1
                self.stable_id_counter = new_id
                final_stable_ids.append(new_id)
                added_count += 1
                self.logger.info(f"Added new object with ID {new_id}, Box {new_box}")
    
        self.logger.info("=== Merge Summary ===")
        self.logger.info(f"Initial active tracked objects: {len(active_indices)}")
        self.logger.info(f"New objects added: {added_count}")
        self.logger.info(f"Final object count: {len(final_boxes)}")
    
        # Print final bounding boxes
        for idx, (box, obj_id) in enumerate(zip(final_boxes, final_stable_ids)):
            self.logger.info(f"Final Object {idx+1}: ID {obj_id}, Box {box}")
    
        return final_boxes, final_masks, final_labels, final_stable_ids