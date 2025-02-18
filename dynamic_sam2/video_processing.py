# video_processing.py
import cv2
import os
import numpy as np
import supervision as sv
from pathlib import Path
import subprocess
from tqdm import tqdm
import logging
from typing import List, Optional, Tuple

class VideoProcessor:
    def __init__(self, video_path: Path, target_fps: int, target_resolution: tuple, 
                 output_dir: Path, frames_dir: Path, logger: logging.Logger):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frames_dir = frames_dir
        self.logger = logger
        
        video_info = sv.VideoInfo.from_video_path(str(video_path))
        self.original_fps = video_info.fps
        self.original_width = video_info.width
        self.original_height = video_info.height
        self.target_fps = target_fps if target_fps else self.original_fps
        
        if target_resolution:
            self.process_width, self.process_height = target_resolution
        else:
            self.process_width, self.process_height = (
                self.original_width, self.original_height
            )

    def extract_frames(self):
        cap = cv2.VideoCapture(str(self.video_path))
        frame_interval = max(1, round(self.original_fps / self.target_fps))
        frame_paths = []
        frame_count = 0
        saved_count = 0

        with sv.ImageSink(self.frames_dir, image_name_pattern="{:05d}.jpg") as sink:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    if (
                        self.process_width != self.original_width
                        or self.process_height != self.original_height
                    ):
                        frame = cv2.resize(
                            frame, 
                            (self.process_width, self.process_height)
                        )
                    sink.save_image(frame)
                    frame_paths.append(self.frames_dir / f"{saved_count:05d}.jpg")
                    saved_count += 1
                frame_count += 1

        cap.release()
        return frame_paths

    def create_video(self, tracked_frames=None):
        """Create output video from tracked frames."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if tracked_frames is None:
            tracked_frames = sorted(self.output_dir.glob("*_tracked.jpg"))
                
        if not tracked_frames:
            self.logger.error("No tracked frames found for video creation")
            return

        # Log first few frames for debugging
        self.logger.debug(f"First frame path: {tracked_frames[0]}")
        self.logger.debug(f"First frame exists: {tracked_frames[0].exists()}")
        
        output_path = self.output_dir / "tracked.mp4"
        list_file = self.output_dir / "frames.txt"
        
        self.logger.info(f"Creating video from {len(tracked_frames)} frames")
        self.logger.info(f"Target FPS: {self.target_fps}")
        self.logger.info(f"Output path: {output_path}")
        
        try:
            # Try OpenCV first as it's more reliable
            self._create_video_opencv(tracked_frames)
        except Exception as e:
            self.logger.error(f"OpenCV video creation failed: {str(e)}")
            self.logger.info("Falling back to ffmpeg")
            self._create_video_ffmpeg(tracked_frames, output_path, list_file)

    def _create_video_ffmpeg(self, tracked_frames, output_path, list_file):
        """Create video using ffmpeg."""
        try:
            # Write frames list
            with open(list_file, "w") as f:
                for frame in tracked_frames:
                    # Use absolute paths
                    abs_path = frame.absolute()
                    f.write(f"file '{abs_path}'\n")
                    f.write(f"duration {1/self.target_fps}\n")

            cmd = (
                f"ffmpeg -y -f concat -safe 0 -i {list_file} "
                f"-c:v libx264 -preset medium -crf 23 "
                f"-r {self.target_fps} "
                f"-pix_fmt yuv420p {output_path}"
            )
            
            self.logger.info(f"Running ffmpeg command: {cmd}")
            process = subprocess.run(cmd, shell=True, check=True, 
                                   capture_output=True, text=True)
            self.logger.info(f"Video successfully saved to {output_path}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ffmpeg error: {e.stderr}")
            raise
        finally:
            if list_file.exists():
                list_file.unlink()
                self.logger.debug("Cleaned up temporary files")

    def _create_video_opencv(self, tracked_frames):
        """Create video using OpenCV."""
        self.logger.info("Creating video with OpenCV")
        if not tracked_frames:
            raise ValueError("No tracked frames found")
            
        first = cv2.imread(str(tracked_frames[0]))
        if first is None:
            raise ValueError(f"Could not read first frame: {tracked_frames[0]}")
            
        h, w = first.shape[:2]
        output_path = self.output_dir / "tracked.mp4"
        self.logger.info(f"Creating video at {self.target_fps} fps, resolution: {w}x{h}")

        codecs = ['mp4v', 'avc1', 'H264', 'XVID', 'MJPG']  # Changed order
        writer = None
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    self.target_fps,
                    (w, h)
                )
                if writer.isOpened():
                    self.logger.info(f"Successfully opened writer with codec {codec}")
                    break
                writer.release()
            except Exception as e:
                self.logger.debug(f"Codec {codec} failed: {str(e)}")
                continue

        if not writer or not writer.isOpened():
            raise RuntimeError("Failed to initialize video writer with any codec")

        try:
            success_count = 0
            for frame_path in tqdm(tracked_frames, desc="Writing frames"):
                frame_bgr = cv2.imread(str(frame_path))
                if frame_bgr is None:
                    self.logger.warning(f"Could not read frame: {frame_path}")
                    continue
                if frame_bgr.shape[:2] != (h, w):
                    frame_bgr = cv2.resize(frame_bgr, (w, h))
                writer.write(frame_bgr)
                success_count += 1
            
            self.logger.info(f"Successfully wrote {success_count}/{len(tracked_frames)} frames")
            self.logger.info(f"OpenCV video saved at: {output_path}")
        finally:
            writer.release()