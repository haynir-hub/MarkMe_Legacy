"""
Video Markme - Main Application Entry Point
Desktop application for tracking players in videos and adding visual markers
"""
import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from src.ui.main_window import MainWindow


def setup_ffmpeg_path():
    """Ensure FFmpeg is in PATH"""
    # Add user bin directory to PATH (where FFmpeg symlink is located)
    home_bin = os.path.expanduser("~/bin")
    if os.path.exists(home_bin) and home_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{home_bin}:{os.environ.get('PATH', '')}"

    # Try to use imageio-ffmpeg if available
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(ffmpeg_exe):
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            if ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{ffmpeg_dir}:{os.environ.get('PATH', '')}"
            print(f"✅ FFmpeg found: {ffmpeg_exe}")
    except Exception as e:
        # Broad catch to ensure app startup never crashes due to FFmpeg detection issues
        print(f"⚠️  Failed to configure FFmpeg via imageio_ffmpeg: {e}")


def main():
    """Main application entry point"""
    # Setup FFmpeg path before starting app
    setup_ffmpeg_path()

    app = QApplication(sys.argv)
    app.setApplicationName("Video Markme")
    app.setOrganizationName("Video Markme")

    # Force LTR layout for the entire application (fixes slider direction issues)
    app.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


