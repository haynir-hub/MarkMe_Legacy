"""
Player Segmentation using MediaPipe Image Segmentation (Tasks API)
Extracts player silhouette for proper marker layering
"""
import cv2
import numpy as np
import os
import urllib.request
from typing import Tuple, Optional

MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("Warning: MediaPipe not installed. Segmentation disabled.")

# Model URL and local path
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "selfie_segmenter.tflite")


def download_model_if_needed():
    """Download the segmentation model if not present"""
    if os.path.exists(MODEL_PATH):
        return True

    print(f"Downloading segmentation model to {MODEL_PATH}...")
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


class PlayerSegmentation:
    """
    Segments players from background using MediaPipe Tasks API.
    Allows markers to be drawn UNDER the player.
    """

    def __init__(self):
        """Initialize segmentation with MediaPipe Tasks API"""
        self.enabled = False
        self.segmenter = None

        if not MEDIAPIPE_AVAILABLE:
            return

        if not download_model_if_needed():
            return

        try:
            options = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                output_category_mask=False,
                output_confidence_masks=True
            )
            self.segmenter = ImageSegmenter.create_from_options(options)
            self.enabled = True
            print("MediaPipe segmentation (Confidence Mode) initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize segmenter: {e}")

    def get_player_mask(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None,
                        threshold: float = 0.4) -> Optional[np.ndarray]:
        """
        Get binary mask of player in frame using confidence scores.

        Args:
            frame: BGR image
            bbox: Optional (x, y, w, h) to crop region of interest
            threshold: Confidence threshold (0.0 - 1.0)

        Returns:
            Binary mask (0 or 255) same size as frame, or None if failed
        """
        if not self.enabled or self.segmenter is None:
            return None

        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Process frame
            result = self.segmenter.segment(mp_image)

            if result.confidence_masks is None:
                return None

            # Get person mask (index 0 usually for background, index 1 for person in selfie segmenter)
            # Actually, selfie segmenter has two masks: index 0 (background) and index 1 (person)
            person_mask = result.confidence_masks[1].numpy_view()

            # Threshold confidence mask to create binary mask
            mask = (person_mask > threshold).astype(np.uint8) * 255

            # If bbox provided, zero out areas outside bbox (with padding)
            if bbox is not None:
                x, y, w, h = bbox
                # Use smaller padding for person mask to avoid including floor
                pad = int(max(w, h) * 0.15) 

                bbox_mask = np.zeros_like(mask)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                bbox_mask[y1:y2, x1:x2] = 255

                # Combine: only keep segmentation within bbox area
                mask = cv2.bitwise_and(mask, bbox_mask)

            return mask

        except Exception as e:
            print(f"Segmentation error: {e}")
            return None

    def composite_player_over_marker(self, marked_frame: np.ndarray,
                                       original_frame: np.ndarray,
                                       mask: np.ndarray,
                                       bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Surgical Composite: Only restore original pixels where:
        1. A marker was actually drawn (marked_frame != original_frame)
        2. The pixel is part of the player silhouette (mask == 255)
        """
        x, y, w, h = bbox
        pad = int(max(w, h) * 0.3)

        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(original_frame.shape[1], x + w + pad), min(original_frame.shape[0], y + h + pad)

        result = marked_frame.copy()

        # ROI extraction
        mask_roi = mask[y1:y2, x1:x2]
        original_roi = original_frame[y1:y2, x1:x2]
        marked_roi = marked_frame[y1:y2, x1:x2]

        # 1. Identify which pixels were actually modified by the marker
        # We calculate the absolute difference between marked and original
        diff = cv2.absdiff(marked_roi, original_roi)
        # Convert to grayscale and threshold to get a mask of modified pixels
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        modified_mask = (diff_gray > 5).astype(np.uint8) * 255  # 5 is noise threshold

        # 2. Intersection: Modified AND Player Silhouette
        # These are the pixels we want to restore to original (to put player on top)
        restore_mask = cv2.bitwise_and(modified_mask, mask_roi)

        # 3. Soften edges of the restoration mask
        restore_float = restore_mask.astype(np.float32) / 255.0
        restore_float = cv2.GaussianBlur(restore_float, (5, 5), 0)
        restore_3ch = cv2.merge([restore_float] * 3)

        # 4. Composite:
        # result = marked (with marker) * (1 - restore) + original (clean) * restore
        composite_roi = (marked_roi.astype(np.float32) * (1.0 - restore_3ch) +
                        original_roi.astype(np.float32) * restore_3ch)

        result[y1:y2, x1:x2] = np.clip(composite_roi, 0, 255).astype(np.uint8)

        return result

    def render_with_segmentation(self, frame: np.ndarray,
                                  bbox: Tuple[int, int, int, int],
                                  draw_marker_func,
                                  threshold: float = 0.5) -> np.ndarray:
        """
        High-level function: render marker UNDER player.

        Process:
        1. Get player segmentation mask (only within bbox area)
        2. Draw marker on frame
        3. Composite: player pixels from original ON TOP of marked frame

        Args:
            frame: Original frame
            bbox: Player bounding box
            draw_marker_func: Function that draws marker on frame
            threshold: Segmentation threshold

        Returns:
            Frame with marker visible on floor, player on top
        """
        if not self.enabled:
            # Fallback: just draw marker normally
            return draw_marker_func(frame.copy())

        # 1. Get player mask (limited to bbox area)
        mask = self.get_player_mask(frame, bbox, threshold)

        if mask is None or np.sum(mask) < 100:  # No valid segmentation
            return draw_marker_func(frame.copy())

        # 2. Draw marker on copy of frame
        marked_frame = draw_marker_func(frame.copy())

        # 3. Composite: player from original ON TOP of marker
        result = self.composite_player_over_marker(marked_frame, frame, mask, bbox)

        return result

    def close(self):
        """Release resources"""
        if self.segmenter is not None:
            self.segmenter.close()
            self.segmenter = None


# Singleton instance for reuse
_segmenter_instance: Optional[PlayerSegmentation] = None


def get_segmenter() -> PlayerSegmentation:
    """Get or create singleton segmenter instance"""
    global _segmenter_instance
    if _segmenter_instance is None:
        _segmenter_instance = PlayerSegmentation()
    return _segmenter_instance

