#!/usr/bin/env python3
"""
Component Tests - Test each component individually
בדיקות רכיבים - בדיקת כל רכיב בנפרד
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all imports work"""
    print("=" * 70)
    print("📦 Test 1: Imports / בדיקת ייבואים")
    print("=" * 70)

    try:
        print("\n1. Testing PyQt6 imports...")
        from PyQt6.QtWidgets import QApplication, QLabel
        from PyQt6.QtCore import Qt, pyqtSignal
        from PyQt6.QtGui import QImage, QPixmap
        print("   ✅ PyQt6 imports successful")

        print("\n2. Testing tracking module imports...")
        from src.tracking.tracker_manager import TrackerManager
        from src.tracking.tracking_analyzer import TrackingAnalyzer, TrackingIssue
        from src.tracking.player_tracker import PlayerTracker
        print("   ✅ Tracking module imports successful")

        print("\n3. Testing UI module imports...")
        from src.ui.bbox_editor import BboxEditor
        from src.ui.tracking_review_dialog import TrackingReviewDialog, ConfidenceGraph
        print("   ✅ UI module imports successful")

        print("\n4. Testing render module imports...")
        from src.render.video_exporter import VideoExporter
        print("   ✅ Render module imports successful")

        print("\n5. Testing OpenCV and NumPy...")
        import cv2
        import numpy as np
        print("   ✅ OpenCV and NumPy imports successful")

        print("\n✅ All imports successful!")
        return True

    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_bbox_editor_creation():
    """Test BboxEditor widget creation"""
    print("\n" + "=" * 70)
    print("🎨 Test 2: BboxEditor Creation / יצירת BboxEditor")
    print("=" * 70)

    try:
        from PyQt6.QtWidgets import QApplication
        from src.ui.bbox_editor import BboxEditor
        import numpy as np

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        print("\n1. Creating BboxEditor widget...")
        editor = BboxEditor()
        print("   ✅ BboxEditor created")

        print("\n2. Testing set_frame with dummy frame...")
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray frame
        editor.set_frame(dummy_frame, bbox=(100, 100, 150, 200))
        print("   ✅ Frame set successfully")

        print("\n3. Testing get_bbox...")
        bbox = editor.get_bbox()
        print(f"   ✅ Got bbox: {bbox}")

        print("\n4. Testing clear_bbox...")
        editor.clear_bbox()
        bbox = editor.get_bbox()
        print(f"   ✅ Bbox cleared: {bbox}")

        print("\n✅ BboxEditor tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ BboxEditor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracking_analyzer():
    """Test TrackingAnalyzer functionality"""
    print("\n" + "=" * 70)
    print("🔍 Test 3: TrackingAnalyzer / מנתח מעקב")
    print("=" * 70)

    try:
        from src.tracking.tracking_analyzer import TrackingAnalyzer

        analyzer = TrackingAnalyzer()
        print("✅ TrackingAnalyzer created")

        # Create dummy tracking data
        print("\n1. Creating dummy tracking data...")
        tracking_data = {
            0: {'bbox': (100, 100, 50, 80), 'confidence': 0.9, 'is_learning_frame': True},
            1: {'bbox': (105, 105, 50, 80), 'confidence': 0.85, 'is_learning_frame': False},
            2: {'bbox': (110, 110, 50, 80), 'confidence': 0.8, 'is_learning_frame': False},
            3: {'bbox': (200, 200, 50, 80), 'confidence': 0.3, 'is_learning_frame': False},  # Jump + low conf
            4: {'bbox': None, 'confidence': 0.0, 'is_learning_frame': False},  # Lost tracking
            5: {'bbox': (210, 210, 50, 80), 'confidence': 0.7, 'is_learning_frame': False},
        }
        print("   ✅ Dummy data created")

        print("\n2. Analyzing tracking data...")
        issues = analyzer.analyze(tracking_data, frame_width=640, frame_height=480)
        print(f"   ✅ Found {len(issues)} issues")

        print("\n3. Getting summary...")
        summary = analyzer.get_summary(issues)
        print(f"   Total issues: {summary['total']}")
        print(f"   By severity: {summary['by_severity']}")
        print(f"   By type: {summary['by_type']}")

        print("\n4. Calculating quality score...")
        quality = analyzer.calculate_tracking_quality_score(tracking_data, issues)
        print(f"   ✅ Quality score: {quality:.2f}")

        print("\n5. Getting correction suggestions...")
        suggestions = analyzer.suggest_corrections(issues, tracking_data)
        print(f"   ✅ Suggestions: {len(suggestions)} frames")
        for frame_idx, reason in suggestions:
            print(f"      Frame {frame_idx}: {reason}")

        print("\n✅ TrackingAnalyzer tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ TrackingAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracker_manager_basic():
    """Test TrackerManager basic functionality"""
    print("\n" + "=" * 70)
    print("🎯 Test 4: TrackerManager Basic / בסיס TrackerManager")
    print("=" * 70)

    try:
        from src.tracking.tracker_manager import TrackerManager

        print("\n1. Creating TrackerManager...")
        manager = TrackerManager()
        print("   ✅ TrackerManager created")

        print("\n2. Testing player management...")
        player_id = manager.add_player(
            name="Test Player",
            marker_style="circle",
            initial_frame=0,
            bbox=(100, 100, 50, 80)
        )
        print(f"   ✅ Player added: {player_id}")

        print("\n3. Getting player...")
        player = manager.get_player(player_id)
        print(f"   ✅ Player retrieved: {player.name}")

        print("\n4. Testing learning frames...")
        manager.add_learning_frame_to_player(player_id, 50, (150, 150, 60, 90))
        print(f"   ✅ Learning frame added")
        print(f"   Learning frames: {player.learning_frames}")

        print("\n✅ TrackerManager basic tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ TrackerManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all component tests"""
    print("\n" + "=" * 70)
    print("🧪 Running All Component Tests")
    print("הרצת כל בדיקות הרכיבים")
    print("=" * 70)

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: BboxEditor
    results.append(("BboxEditor", test_bbox_editor_creation()))

    # Test 3: TrackingAnalyzer
    results.append(("TrackingAnalyzer", test_tracking_analyzer()))

    # Test 4: TrackerManager
    results.append(("TrackerManager", test_tracker_manager_basic()))

    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary / סיכום בדיקות")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! / כל הבדיקות עברו!")
        print("\nNext steps:")
        print("1. Run: python test_with_sample_video.py")
        print("   Or: python test_complete_workflow.py <your-video.mp4>")
        print("2. Test the complete three-phase workflow")
        print("3. Try manual bbox correction")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("Please fix the issues before proceeding")

    return passed == total


if __name__ == "__main__":
    run_all_tests()
