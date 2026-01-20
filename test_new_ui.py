"""
Test script for new Two-Phase Tracking UI
סקריפט בדיקה לממשק הדו-שלבי החדש
"""

import sys
import os
import shutil

# FORCE CLEAN: Delete all __pycache__ before loading
print("🧹 Cleaning Python cache...")
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        cache_path = os.path.join(root, '__pycache__')
        shutil.rmtree(cache_path)
        print(f"  Deleted: {cache_path}")
print("✅ Cache cleaned\n")

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from src.tracking.tracker_manager import TrackerManager
from src.ui.two_phase_ui import TwoPhaseTrackingUI


def main():
    """Test the new UI"""
    app = QApplication(sys.argv)

    # Select video file
    video_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select Video File",
        "",
        "Video Files (*.mp4 *.mov *.avi *.mkv)"
    )

    if not video_path:
        print("No video selected")
        return

    print(f"Loading video: {video_path}")

    # Create tracker manager
    tracker_manager = TrackerManager()

    try:
        # Load video
        tracker_manager.load_video(video_path)
        print(f"✅ Video loaded: {tracker_manager.total_frames} frames")

        # Show UI
        dialog = TwoPhaseTrackingUI(tracker_manager)

        if dialog.exec():
            # User clicked "Continue to Export"
            tracking_data = dialog.get_tracking_data()

            print("\n=== Tracking Results ===")
            for player_id, player_data in tracking_data.items():
                player = tracker_manager.players.get(player_id)
                player_name = player.name if player else player_id

                frames_tracked = len([f for f in player_data.values() if f.get('bbox') is not None])
                total_frames = len(player_data)

                print(f"\nPlayer: {player_name}")
                print(f"  Frames tracked: {frames_tracked}/{total_frames}")

                if player_data:
                    confidences = [f.get('confidence', 0) for f in player_data.values() if f.get('bbox')]
                    if confidences:
                        avg_conf = sum(confidences) / len(confidences)
                        print(f"  Average confidence: {avg_conf:.2f}")

            QMessageBox.information(
                None,
                "Tracking Complete",
                f"Tracking complete!\n\n"
                f"Players tracked: {len(tracking_data)}\n\n"
                f"Ready to export video with markers."
            )
        else:
            print("User cancelled")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

        QMessageBox.critical(
            None,
            "Error",
            f"An error occurred:\n\n{str(e)}"
        )


if __name__ == "__main__":
    main()
