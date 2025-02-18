import logging
import torch
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from .object_detection import BaseObjectDetectionModel
from .tracking_utils import BoxUtils
from .visualization import VisualizationManager
from .video_processing import VideoProcessor
from .object_tracking import ObjectTracker

# Fix for missing _C module in MultiScaleDeformableAttention
from grounding_dino.groundingdino.models.GroundingDINO.ms_deform_attn import multi_scale_deformable_attn_pytorch, MultiScaleDeformableAttnFunction

# Create replacement for MultiScaleDeformableAttnFunction.apply
class DummyFunction:
    @staticmethod
    def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        return multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

# Replace the original function with our implementation
MultiScaleDeformableAttnFunction.apply = DummyFunction.apply

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Sam2VideoTracker:
    def __init__(
        self,
        video_path: str,
        text_prompt: str,
        detection_model,
        output_dir: str = "tracking_results",
        frames_dir: str = "temp_frames",
        device: str = "cuda",
        target_fps: int = None,
        target_resolution: tuple = None,
        check_interval: int = 10,
        iou_threshold: float = 0.5,
        sam2_cfg_path: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path: str = os.path.join(BASE_DIR, "checkpoints", "sam2.1_hiera_large.pt"),
        generate_video: bool = True,
        min_tracked_frames: int = 5,  # Minimum frames for valid object detection
        save_masks: bool = False  # Whether to save masks in tracked_objects dictionary
    ):
        self.video_path = Path(video_path)
        self.text_prompt = text_prompt.lower()
        self.detection_model = detection_model
        self.output_dir = Path(output_dir)
        self.frames_dir = Path(frames_dir)
        self.device = device
        self.check_interval = check_interval
        self.generate_video = generate_video
        self.min_tracked_frames = min_tracked_frames
        self.save_masks = save_masks
        
        # Dictionary to track object positions, classes and masks across frames
        # {object_id: {
        #    "frames": {frame_idx: box_coordinates}, 
        #    "class": class_name,
        #    "masks": {frame_idx: mask} (optional)
        # }}
        self.tracked_objects = {}
        self.object_classes = {}  # Temporary storage for object classes during tracking
        
        # Initialize logging
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up frames directory at start
        if self.frames_dir.exists():
            shutil.rmtree(str(self.frames_dir))
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.output_dir / "tracker.log"
        self._init_logger()
        
        # Initialize components
        self.video_processor = VideoProcessor(
            self.video_path, target_fps, target_resolution,
            self.output_dir, self.frames_dir, self.logger
        )
        self.visualizer = VisualizationManager(self.output_dir, self.logger)
        self.object_tracker = ObjectTracker(iou_threshold, self.logger)

        # Load SAM2
        self.logger.info("Loading SAM2 models...")
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("Checkpoint path:", sam2_ckpt_path)
        sam2_model = build_sam2(sam2_cfg_path, sam2_ckpt_path)
        self.image_predictor = SAM2ImagePredictor(sam2_model)
        self.video_predictor = build_sam2_video_predictor(
            sam2_cfg_path, sam2_ckpt_path
        )
        
        # Storage for frame data
        self.frame_data = {}

    def _init_logger(self):
        self.logger = logging.getLogger('Sam2VideoTracker')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def _create_masks_from_boxes(self, image_bgr, boxes_xyxy):
        self.logger.debug(f"Creating masks for {len(boxes_xyxy)} boxes")
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.image_predictor.set_image(img_rgb)
        masks, _, _ = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=False
        )
        
        if masks is None:
            self.logger.warning("Failed to create masks")
            return None
            
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        elif masks.ndim == 2:
            masks = masks[np.newaxis, ...]
            
        return masks

    def _process_logits_to_masks(self, mask_logits):
        masks = (mask_logits > 0.0).cpu().numpy()
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        if (
            self.video_processor.process_width != self.video_processor.original_width
            or self.video_processor.process_height != self.video_processor.original_height
        ):
            resized = []
            for m in masks:
                rm = cv2.resize(
                    m.astype(np.uint8),
                    (self.video_processor.process_width, self.video_processor.process_height),
                    interpolation=cv2.INTER_NEAREST
                )
                resized.append(rm.astype(bool))
            masks = np.array(resized)
            
        return masks

    def _masks_to_boxes(self, masks):
        boxes = []
        for m in masks:
            coords = np.where(m)
            if coords[0].size > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                boxes.append([x_min, y_min, x_max, y_max])
            else:
                boxes.append([0, 0, 1, 1])
        return np.array(boxes)
    
    def _update_tracked_objects(self, frame_idx, boxes, stable_ids, labels=None, masks=None):
        """Update tracked objects dictionary with position data, class labels, and masks"""
        for i, obj_id in enumerate(stable_ids):
            # Skip invalid boxes [0,0,1,1]
            if (i < len(boxes) and 
                not (boxes[i][0] == 0 and boxes[i][1] == 0 and 
                     boxes[i][2] == 1 and boxes[i][3] == 1)):
                
                # Initialize object entry if it doesn't exist
                if obj_id not in self.tracked_objects:
                    obj_data = {
                        "frames": {},
                        "class": self.object_classes.get(obj_id, "unknown"),
                    }
                    if self.save_masks:
                        obj_data["masks"] = {}
                    self.tracked_objects[obj_id] = obj_data
                
                # Store box coordinates
                self.tracked_objects[obj_id]["frames"][frame_idx] = boxes[i].tolist()
                
                # Store mask if requested and available
                if self.save_masks and masks is not None and i < len(masks):
                    # Convert mask to binary and to efficient storage format
                    binary_mask = masks[i].astype(np.uint8)
                    self.tracked_objects[obj_id]["masks"][frame_idx] = binary_mask
                
                # Update class label if available
                if labels is not None and i < len(labels):
                    self.tracked_objects[obj_id]["class"] = labels[i]
                    self.object_classes[obj_id] = labels[i]
    
    def _cleanup_resources(self):
        """Clean up temporary resources"""
        # Remove tracked frame images
        tracked_frames = list(self.output_dir.glob("*_tracked.jpg"))
        for frame_path in tracked_frames:
            try:
                os.remove(frame_path)
            except Exception as e:
                self.logger.warning(f"Could not remove frame {frame_path}: {str(e)}")
                
        # Clean up temp frames directory
        if self.frames_dir.exists():
            try:
                shutil.rmtree(str(self.frames_dir))
                self.logger.debug("Cleaned up temporary frames directory")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp directory: {str(e)}")
    
    def _filter_objects_by_min_frames(self):
        """Filter out objects that appear in fewer than min_tracked_frames frames"""
        before_count = len(self.tracked_objects)
        filtered_objects = {}
        
        for obj_id, obj_data in self.tracked_objects.items():
            # Count valid frames (excluding [0,0,1,1] boxes which might be in the frames dict)
            valid_frames = {}
            valid_masks = {} if self.save_masks and "masks" in obj_data else None
            
            for frame_idx, box in obj_data["frames"].items():
                if not (box[0] == 0 and box[1] == 0 and box[2] == 1 and box[3] == 1):
                    valid_frames[frame_idx] = box
                    if valid_masks is not None and frame_idx in obj_data["masks"]:
                        valid_masks[frame_idx] = obj_data["masks"][frame_idx]
                    
            if len(valid_frames) >= self.min_tracked_frames:
                new_obj_data = {
                    "frames": valid_frames,
                    "class": obj_data["class"]
                }
                if valid_masks is not None:
                    new_obj_data["masks"] = valid_masks
                filtered_objects[obj_id] = new_obj_data
        
        removed_count = before_count - len(filtered_objects)
        self.logger.info(f"Filtered out {removed_count} objects with fewer than {self.min_tracked_frames} valid frames")
        self.tracked_objects = filtered_objects
        
    def _prepare_chunk_frames(self, start_frame, end_frame, frame_paths):
        """
        Prepare frames for the current chunk in a separate subdirectory
        Returns the path to the chunk directory and the relative frame paths
        """
        # Create chunk directory
        chunk_dir = self.frames_dir / f"chunk_{start_frame}_{end_frame}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy relevant frames to chunk directory
        chunk_frames = []
        for idx in range(start_frame, min(end_frame + 1, len(frame_paths))):
            source_path = frame_paths[idx]
            # Use same filename format as original extraction
            dest_path = chunk_dir / source_path.name
            
            # Copy the frame if it doesn't exist yet
            if not dest_path.exists():
                shutil.copy2(source_path, dest_path)
            
            chunk_frames.append(dest_path)
            
        self.logger.debug(f"Prepared {len(chunk_frames)} frames for chunk {start_frame}-{end_frame}")
        return chunk_dir, chunk_frames
    
    def process_video(self):
        """Process video with overlapping chunks, using merged results as starting point for next chunk."""
        self.logger.info("\n=== Starting Video Processing ===")
        try:
            # First extract all frames to a temporary directory
            all_frame_paths = self.video_processor.extract_frames()
            if not all_frame_paths:
                self.logger.error("No frames extracted")
                return self.tracked_objects
                
            total_frames = len(all_frame_paths)
            current_frame = 0
    
            # Process video in overlapping chunks
            while current_frame < total_frames:
                # Calculate chunk end
                chunk_end = min(current_frame + self.check_interval - 1, total_frames - 1)
    
                # Prepare frames for this chunk
                chunk_dir, chunk_frames = self._prepare_chunk_frames(
                    current_frame, chunk_end, all_frame_paths
                )
    
                self.logger.info(
                    f'Processing chunk: current_frame = {current_frame}, chunk_end = {chunk_end}, '
                    f'frames in chunk = {len(chunk_frames)}'
                )
    
                # First frame of video needs initial detection
                if current_frame == 0:
                    boxes, labels = self.detection_model.detect(
                        chunk_frames[0],
                        self.text_prompt
                    )
                    
                    if boxes is not None and len(boxes) > 0:
                        frame_bgr = cv2.imread(str(chunk_frames[0]))
                        masks = self._create_masks_from_boxes(frame_bgr, boxes)
                        
                        if masks is not None and len(masks) == len(boxes):
                            stable_ids = [
                                self.object_tracker.get_stable_id(box) 
                                for box in boxes
                            ]
                            
                            # Initialize object classes
                            for i, obj_id in enumerate(stable_ids):
                                if i < len(labels):
                                    self.object_classes[obj_id] = labels[i]
                            
                            # Update tracked objects including masks
                            self._update_tracked_objects(current_frame, boxes, stable_ids, labels, masks)
                        else:
                            boxes = np.array([])
                            masks = np.array([])
                            labels = []
                            stable_ids = []
                    else:
                        boxes = np.array([])
                        masks = np.array([])
                        labels = []
                        stable_ids = []
                        
                    self.frame_data[current_frame] = (boxes, masks, labels, stable_ids)
    
                # Get current objects (either from initial detection or previous merge)
                boxes, masks, labels, stable_ids = self.frame_data[current_frame]
                
                # Check if we have objects to track
                if len(boxes) == 0 or len(masks) == 0:
                    self.logger.info(f"No objects to track at frame {current_frame}. Skipping SAM2 initialization for this chunk.")
                    
                    # Skip to next chunk start but visualize current frame with empty results
                    self.visualizer.visualize_frame(
                        chunk_frames[0],
                        boxes, masks, labels,
                        stable_ids=stable_ids
                    )
                    
                    # Check if we need to run detection on the last frame of this chunk
                    if chunk_end >= 0:
                        # Get new DINO detections on the last frame of chunk
                        new_boxes, new_labels = self.detection_model.detect(
                            chunk_frames[-1],
                            self.text_prompt
                        )
                        
                        if new_boxes is not None and len(new_boxes) > 0:
                            # Create masks for new detections
                            frame_bgr = cv2.imread(str(chunk_frames[-1]))
                            new_masks = self._create_masks_from_boxes(frame_bgr, new_boxes)
                            
                            if new_masks is not None and len(new_masks) == len(new_boxes):
                                # Get stable IDs for new detections
                                new_stable_ids = [
                                    self.object_tracker.get_stable_id(box) 
                                    for box in new_boxes
                                ]
                                
                                # Initialize object classes for new detections
                                for i, obj_id in enumerate(new_stable_ids):
                                    if i < len(new_labels):
                                        self.object_classes[obj_id] = new_labels[i]
                                
                                # Update frame data and tracked objects
                                self.frame_data[chunk_end] = (
                                    new_boxes, new_masks, new_labels, new_stable_ids
                                )
                                self._update_tracked_objects(
                                    chunk_end, new_boxes, new_stable_ids, new_labels, new_masks
                                )
                                
                                # Visualize frame with new detections
                                self.visualizer.visualize_frame(
                                    all_frame_paths[chunk_end],
                                    new_boxes, new_masks, new_labels,
                                    new_stable_ids
                                )
                            else:
                                # Store empty data if masks creation failed
                                self.frame_data[chunk_end] = (
                                    np.array([]), np.array([]), [], []
                                )
                        else:
                            # Store empty data if no new detections
                            self.frame_data[chunk_end] = (
                                np.array([]), np.array([]), [], []
                            )
                    
                    # Clean up chunk directory and move to next chunk
                    try:
                        shutil.rmtree(str(chunk_dir))
                        self.logger.debug(f"Cleaned up chunk directory: {chunk_dir}")
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up chunk directory: {str(e)}")
                    
                    if chunk_end >= total_frames - 1:
                        break  # Exit loop when last frame is reached
                    # Move to next chunk
                    current_frame = chunk_end
                    continue
                
                # Initialize SAM2 tracking with current frame's objects
                state = self.video_predictor.init_state(
                    video_path=str(chunk_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True
                )
    
                # Add objects to tracker
                for i, m in enumerate(masks):
                    mask_t = torch.from_numpy(m).to(self.device)
                    self.video_predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=0,  # Always use 0 as we're working with relative frames in chunk
                        obj_id=stable_ids[i],
                        mask=mask_t
                    )
                    mask_t = mask_t.cpu()
    
                # Visualize current frame
                self.visualizer.visualize_frame(
                    chunk_frames[0],
                    boxes, masks, labels,
                    stable_ids=stable_ids
                )
    
                # Track objects through chunk - note that frame indices are relative to chunk
                relative_max_frames = len(chunk_frames) - 1  # -1 because we start from 0
                for (rel_f_idx, obj_ids, mask_logits) in self.video_predictor.propagate_in_video(
                    inference_state=state,
                    start_frame_idx=0,  # Start from beginning of chunk
                    max_frame_num_to_track=relative_max_frames
                ):
                    # Convert relative frame index to global frame index
                    f_idx = current_frame + rel_f_idx
                    
                    if f_idx >= total_frames:
                        break
                    if len(obj_ids) == 0:
                        continue
    
                    # Process SAM2 tracking results
                    final_masks = self._process_logits_to_masks(mask_logits)
                    final_boxes = self._masks_to_boxes(final_masks)
    
                    # Map tracked object IDs to original labels
                    id_to_idx = {oid: i for i, oid in enumerate(stable_ids)}
                    final_labels = []
                    final_stable_ids = []
                    
                    for oid in obj_ids:
                        if oid in id_to_idx:
                            idx = id_to_idx[oid]
                            final_labels.append(labels[idx])
                            final_stable_ids.append(stable_ids[idx])
                        else:
                            final_labels.append("unknown")
                            final_stable_ids.append(oid)
    
                    # Update tracked objects with masks
                    self._update_tracked_objects(f_idx, final_boxes, final_stable_ids, final_labels, final_masks)
                            
                    # Store and visualize frame results - use actual frame from all_frame_paths for visualization
                    self.frame_data[f_idx] = (
                        final_boxes, final_masks, final_labels, final_stable_ids
                    )
                    self.visualizer.visualize_frame(
                        all_frame_paths[f_idx],
                        final_boxes, final_masks, final_labels,
                        final_stable_ids
                    )
    
                # Last frame of chunk - merge SAM2 results with new DINO detections
                if chunk_end >= 0:
                    # Get SAM2 tracking results for the last frame
                    old_boxes_pf, old_masks_pf, old_labels_pf, old_stable_ids_pf = \
                        self.frame_data[chunk_end]
                    
                    # Get new DINO detections - use the last frame in chunk
                    new_boxes, new_labels = self.detection_model.detect(
                        chunk_frames[-1],
                        self.text_prompt
                    )
                    
                    # Merge tracked and newly detected objects
                    merged_boxes, merged_masks, merged_labels, merged_stable_ids = \
                        self.object_tracker.merge_detections(
                            old_boxes_pf, old_masks_pf, old_labels_pf, old_stable_ids_pf,
                            new_boxes, new_labels,
                            frame_path=chunk_frames[-1],
                            create_masks_fn=self._create_masks_from_boxes
                        )
    
                    # Update tracked objects with merged results and masks
                    self._update_tracked_objects(chunk_end, merged_boxes, merged_stable_ids, merged_labels, merged_masks)
                    
                    # Update last frame with merged results
                    self.frame_data[chunk_end] = (
                        merged_boxes, merged_masks, merged_labels, merged_stable_ids
                    )
                    self.visualizer.visualize_frame(
                        all_frame_paths[chunk_end],
                        merged_boxes, merged_masks, merged_labels,
                        merged_stable_ids
                    )
    
                # Clean up chunk directory after processing
                try:
                    shutil.rmtree(str(chunk_dir))
                    self.logger.debug(f"Cleaned up chunk directory: {chunk_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up chunk directory: {str(e)}")
    
                if chunk_end >= total_frames - 1:
                    break  # Exit loop when last frame is reached
                # Move to next chunk, starting from the last frame we processed
                current_frame = chunk_end
            
            # First filter objects by minimum tracked frames
            self._filter_objects_by_min_frames()
                
            # Create final video if generate_video is True - ONLY after filtering
            if self.generate_video:
                # Get all tracked frames before they're deleted
                tracked_frames = sorted(self.output_dir.glob("*_tracked.jpg"))
                
                if tracked_frames:
                    # First delete any existing tracked video
                    if (self.output_dir / "tracked.mp4").exists():
                        os.remove(self.output_dir / "tracked.mp4")
                
                    # We need to re-visualize frames with only valid objects
                    self.logger.info("Creating final video with only valid objects...")
                    for frame_idx in range(total_frames):
                        if frame_idx in self.frame_data:
                            # Extract valid objects for this frame
                            valid_boxes = []
                            valid_masks = []
                            valid_labels = []
                            valid_ids = []
                            
                            boxes, masks, labels, stable_ids = self.frame_data[frame_idx]
                            
                            for i, obj_id in enumerate(stable_ids):
                                if obj_id in self.tracked_objects and frame_idx in self.tracked_objects[obj_id]["frames"]:
                                    valid_boxes.append(boxes[i])
                                    valid_masks.append(masks[i])
                                    valid_labels.append(labels[i])
                                    valid_ids.append(obj_id)
                                    
                            # Visualize with only valid objects if any exist
                            if valid_boxes:
                                valid_boxes = np.array(valid_boxes)
                                valid_masks = np.array(valid_masks)
                                
                                # Re-visualize frame with only filtered objects
                                self.visualizer.visualize_frame(
                                    all_frame_paths[frame_idx],
                                    valid_boxes, valid_masks, valid_labels,
                                    valid_ids
                                )
                    
                    # Now create video from re-visualized frames
                    tracked_frames = sorted(self.output_dir.glob("*_tracked.jpg"))            
                    self.video_processor.create_video(tracked_frames)
                    
                    # Rename the output video to match original filename
                    source_video = self.output_dir / "tracked.mp4"
                    if source_video.exists():
                        target_video = self.output_dir / f"{self.video_path.stem}_tracked.mp4"
                        os.rename(source_video, target_video)
                        self.logger.info(f"Renamed output video to: {target_video}")
            
            self.logger.info(f"Tracking complete. Found {len(self.tracked_objects)} unique objects after filtering.")
            
            return self.tracked_objects
    
        except Exception as e:
            self.logger.error(f"Error during video processing: {str(e)}", exc_info=True)
            raise
        finally:
            torch.cuda.empty_cache()
            # Always clean up temporary resources
            self._cleanup_resources()