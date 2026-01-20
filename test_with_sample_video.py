#!/usr/bin/env python3
"""
Quick Test with Generated Sample Video
בדיקה מהירה עם וידאו דוגמה

This script creates a simple test video with a moving object
and tests the tracking system on it.
"""

import sys
import os
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from src.tracking.tracker_manager import TrackerManager
from src.ui.tracking_review_dialog import TrackingReviewDialog


def create_sample_video(output_path: str, duration_seconds: int = 5):
    """
    Create a sample video with a moving rectangle
    יצירת וידאו לדוגמה עם מלבן נע
    """
    print(f"Creating sample video: {output_path}")

    # Video parameters
    fps = 30
    width, height = 640, 480
    total_frames = fps * duration_seconds

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Rectangle properties
    rect_w, rect_h = 60, 100
    start_x = 50

    for frame_idx in range(total_frames):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background

        # Calculate rectangle position (moves from left to right)
        progress = frame_idx / total_frames
        x = int(start_x + (width - start_x - rect_w) * progress)
        y = height // 2 - rect_h // 2

        # Add some vertical movement (sine wave)
        y += int(50 * np.sin(progress * 4 * np.pi))

        # Draw moving rectangle (blue)
        cv2.rectangle(frame, (x, y), (x + rect_w, y + rect_h), (255, 0, 0), -1)

        # Add some noise/distraction
        for _ in range(5):
            noise_x = np.random.randint(0, width - 20)
            noise_y = np.random.randint(0, height - 20)
            cv2.circle(frame, (noise_x, noise_y), 10, (0, 255, 0), -1)

        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        writer.write(frame)

        if frame_idx % 30 == 0:
            print(f"  Generated {frame_idx}/{total_frames} frames", end='\r')

    writer.release()
    print(f"\n✅ Sample video created: {output_path}")
    print(f"   Frames: {total_frames}, FPS: {fps}, Duration: {duration_seconds}s")

    # Return initial bbox for tracking
    initial_bbox = (start_x, height // 2 - rect_h // 2, rect_w, rect_h)
    return initial_bbox


def test_with_sample_video():
    """
    Test tracking system with generated sample video
    בדיקת מערכת מעקב עם וידאו לדוגמה
    """
    print("=" * 70)
    print("🎬 Quick Test with Sample Video")
    print("בדיקה מהירה עם וידאו לדוגמה")
    print("=" * 70)

    # Create sample video
    sample_video_path = "/tmp/sample_tracking_video.mp4"
    print("\n📹 Step 1: Creating sample video...")
    initial_bbox = create_sample_video(sample_video_path, duration_seconds=5)

    # Create Qt application
    app = QApplication(sys.argv)

    # Load video
    print("\n📹 Step 2: Loading sample video...")
    tracker_manager = TrackerManager()
    if not tracker_manager.load_video(sample_video_path):
        print("❌ Failed to load video")
        return False

    print(f"✅ Video loaded: {tracker_manager.total_frames} frames")

    # Add player
    print("\n👤 Step 3: Adding player with initial bbox...")
    player_id = tracker_manager.add_player(
        name="Moving Rectangle",
        marker_style="circle",
        initial_frame=0,
        bbox=initial_bbox
    )
    print(f"✅ Player added: {player_id}")

    # Generate tracking data
    print("\n🎯 Step 4: Generating tracking data...")
    tracking_data = tracker_manager.generate_tracking_data(
        start_frame=0,
        end_frame=tracker_manager.total_frames - 1,
        progress_callback=lambda curr, total: print(f"  Progress: {curr}/{total}", end='\r')
    )
    print(f"\n✅ Tracking complete!")

    # Open review UI
    print("\n👁️  Step 5: Opening review UI...")
    print("\nInstructions:")
    print("1. Review the confidence graph")
    print("2. Try clicking on different frames")
    print("3. Click 'Fix Frame' and edit the bbox")
    print("4. Try all editing features:")
    print("   - Draw new bbox (click-drag)")
    print("   - Resize from corners")
    print("   - Resize from edges")
    print("   - Move bbox (drag center)")
    print("   - Delete (Delete key)")
    print("   - Cancel (ESC)")
    print("5. Click 'Re-track' after corrections")
    print("6. Click 'Continue to Export' when done")

    review_dialog = TrackingReviewDialog(
        tracker_manager=tracker_manager,
        tracking_data=tracking_data
    )

    result = review_dialog.exec()

    if result == review_dialog.DialogCode.Accepted:
        print("\n✅ Test completed successfully!")
        print("   User approved tracking")

        # Show learning frames
        player = tracker_manager.get_player(player_id)
        if player.learning_frames:
            print(f"\n   Learning frames added: {len(player.learning_frames)}")
            for frame_idx, bbox in sorted(player.learning_frames.items()):
                print(f"     Frame {frame_idx}: {bbox}")
    else:
        print("\n❌ Test cancelled by user")

    # Cleanup
    try:
        os.remove(sample_video_path)
        print(f"\n🗑️  Cleaned up sample video")
    except:
        pass

    print("\n" + "=" * 70)
    print("🎬 Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_with_sample_video()
