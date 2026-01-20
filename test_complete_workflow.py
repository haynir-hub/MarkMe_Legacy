#!/usr/bin/env python3
"""
Complete Workflow Test - Three-Phase Tracking System
בדיקה מלאה של מערכת המעקב התלת-שלבית

Usage:
    python test_complete_workflow.py <path/to/video.mp4>

This script tests:
1. Phase 1: Tracking data generation
2. Phase 2: Confidence analysis and review UI
3. Phase 3: Manual bbox correction
4. Export with corrected tracking
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication, QMessageBox
from src.tracking.tracker_manager import TrackerManager
from src.tracking.tracking_analyzer import TrackingAnalyzer
from src.ui.tracking_review_dialog_simple import TrackingReviewDialog
from src.render.video_exporter import VideoExporter


def test_complete_workflow(video_path: str):
    """
    Test complete three-phase workflow

    בדיקה של זרימה מלאה:
    1. טעינת וידאו
    2. הוספת שחקן
    3. Phase 1 - יצירת נתוני מעקב
    4. Phase 2 - ניתוח ביטחון
    5. Phase 2+3 - סקירה וממשק תיקון ידני
    6. ייצוא וידאו
    """

    print("=" * 70)
    print("🎬 Complete Three-Phase Tracking Workflow Test")
    print("בדיקת זרימת עבודה מלאה - מעקב תלת-שלבי")
    print("=" * 70)

    # Validate video path
    if not os.path.exists(video_path):
        print(f"\n❌ Error: Video file not found: {video_path}")
        print(f"❌ שגיאה: קובץ וידאו לא נמצא: {video_path}")
        return False

    # Create Qt application
    app = QApplication(sys.argv)

    # Step 1: Load video
    print("\n" + "=" * 70)
    print("📹 Step 1: Loading Video / טעינת וידאו")
    print("=" * 70)

    tracker_manager = TrackerManager()
    if not tracker_manager.load_video(video_path):
        print("❌ Failed to load video / נכשל בטעינת וידאו")
        return False

    print(f"✅ Video loaded successfully!")
    print(f"   Frames: {tracker_manager.total_frames}")
    print(f"   FPS: {tracker_manager.fps}")
    print(f"   Resolution: {tracker_manager.frame_width}x{tracker_manager.frame_height}")
    print(f"   Duration: {tracker_manager.total_frames / tracker_manager.fps:.2f} seconds")

    # Step 2: Add player
    print("\n" + "=" * 70)
    print("👤 Step 2: Adding Player / הוספת שחקן")
    print("=" * 70)
    print("\nNote: Using default bbox in center of frame")
    print("הערה: משתמש ב-bbox ברירת מחדל במרכז הפריים")

    # Default bbox in center
    center_x = tracker_manager.frame_width // 2 - 50
    center_y = tracker_manager.frame_height // 2 - 75
    default_bbox = (center_x, center_y, 100, 150)

    player_id = tracker_manager.add_player(
        name="Test Player",
        marker_style="circle",
        initial_frame=0,
        bbox=default_bbox
    )

    print(f"✅ Player added: {player_id}")
    print(f"   Initial bbox: {default_bbox}")

    # Step 3: Phase 1 - Generate tracking data
    print("\n" + "=" * 70)
    print("🎯 Step 3: Phase 1 - Generating Tracking Data")
    print("שלב 1 - יצירת נתוני מעקב")
    print("=" * 70)

    # Track first 200 frames or less
    end_frame = min(200, tracker_manager.total_frames - 1)
    print(f"\nTracking frames 0 to {end_frame}...")

    tracking_data = tracker_manager.generate_tracking_data(
        start_frame=0,
        end_frame=end_frame,
        progress_callback=lambda curr, total: print(f"  Progress: {curr}/{total} frames ({100*curr//total}%)", end='\r')
    )

    print(f"\n✅ Tracking data generated!")

    # Step 4: Phase 2 - Analyze tracking quality
    print("\n" + "=" * 70)
    print("🔍 Step 4: Phase 2 - Analyzing Tracking Quality")
    print("שלב 2 - ניתוח איכות מעקב")
    print("=" * 70)

    analyzer = TrackingAnalyzer()

    for pid in tracking_data:
        player = tracker_manager.get_player(pid)
        player_data = tracking_data[pid]

        # Analyze issues
        issues = analyzer.analyze(
            player_data,
            tracker_manager.frame_width,
            tracker_manager.frame_height
        )

        # Get summary
        summary = analyzer.get_summary(issues)

        # Calculate quality score
        quality_score = analyzer.calculate_tracking_quality_score(player_data, issues)

        print(f"\n{player.name}:")
        print(f"  📊 Quality Score: {quality_score:.2f} / 1.00")
        print(f"  🔢 Total Issues: {summary['total']}")

        if summary['by_severity']:
            print(f"  📈 By Severity:")
            for severity, count in summary['by_severity'].items():
                emoji = "🔴" if severity == "critical" else "🟠" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                print(f"     {emoji} {severity}: {count}")

        if summary['by_type']:
            print(f"  🏷️  By Type:")
            for issue_type, count in summary['by_type'].items():
                print(f"     - {issue_type}: {count}")

        if summary.get('critical_frames'):
            print(f"  ⚠️  Critical Frames: {len(summary['critical_frames'])}")
            if len(summary['critical_frames']) <= 10:
                print(f"     Frames: {summary['critical_frames']}")
            else:
                print(f"     First 10: {summary['critical_frames'][:10]}...")

        # Get correction suggestions
        suggestions = analyzer.suggest_corrections(issues, player_data)
        if suggestions:
            print(f"  💡 Suggested Corrections: {len(suggestions)} frames")
            for frame_idx, reason in suggestions[:5]:
                print(f"     - Frame {frame_idx}: {reason}")

        # Quality assessment
        print(f"\n  Assessment:")
        if quality_score >= 0.8:
            print(f"     ✅ Excellent tracking quality - minimal corrections needed")
        elif quality_score >= 0.6:
            print(f"     ⚠️  Good tracking - some corrections recommended")
        elif quality_score >= 0.4:
            print(f"     ⚠️  Fair tracking - corrections needed")
        else:
            print(f"     ❌ Poor tracking - significant corrections required")

    # Step 5: Open review UI
    print("\n" + "=" * 70)
    print("👁️  Step 5: Opening Review UI / פתיחת ממשק סקירה")
    print("=" * 70)
    print("\nInstructions / הוראות:")
    print("1. Review confidence graph / סקור גרף ביטחון")
    print("2. Click on problematic frames / לחץ על פריימים בעייתיים")
    print("3. Click 'Fix Frame' to edit bbox / לחץ 'תקן פריים' לעריכת bbox")
    print("4. Draw/edit bbox with mouse / צייר/ערוך bbox עם עכבר")
    print("   - Click-drag to create / לחץ וגרור ליצירה")
    print("   - Drag corners to resize / גרור פינות לשינוי גודל")
    print("   - Drag center to move / גרור אמצע להזזה")
    print("   - ESC to cancel / ESC לביטול")
    print("   - Delete to clear / Delete למחיקה")
    print("5. Bbox saved automatically as learning frame / bbox נשמר אוטומטית")
    print("6. Click 'Re-track' to update / לחץ 'מעקב מחדש' לעדכון")
    print("7. Repeat as needed / חזור לפי הצורך")
    print("8. Click 'Continue to Export' when done / לחץ 'המשך לייצוא' כשמוכן")

    print("\n🎬 Opening dialog...")

    review_dialog = TrackingReviewDialog(
        tracker_manager=tracker_manager,
        tracking_data=tracking_data
    )

    result = review_dialog.exec()

    if result == review_dialog.DialogCode.Accepted:
        print("\n" + "=" * 70)
        print("✅ User approved tracking / משתמש אישר מעקב")
        print("=" * 70)

        # Show learning frames added
        for pid in tracking_data:
            player = tracker_manager.get_player(pid)
            if player.learning_frames:
                print(f"\n{player.name}:")
                print(f"  Learning Frames Added: {len(player.learning_frames)}")
                for frame_idx, bbox in sorted(player.learning_frames.items()):
                    print(f"    Frame {frame_idx}: {bbox}")

        # Step 6: Export
        print("\n" + "=" * 70)
        print("📹 Step 6: Exporting Video / ייצוא וידאו")
        print("=" * 70)

        # Generate output path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.dirname(video_path)
        output_path = os.path.join(output_dir, f"{base_name}_tracked.mp4")

        print(f"\nOutput path: {output_path}")

        # Ask user if they want to export
        reply = QMessageBox.question(
            None,
            "Export Video / ייצוא וידאו",
            f"Export video to:\n{output_path}\n\nThis may take a few minutes.\nזה עשוי לקחת מספר דקות.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            exporter = VideoExporter(tracker_manager)

            print("\nExporting... / מייצא...")
            success = exporter.export_video(
                input_path=video_path,
                output_path=output_path,
                progress_callback=lambda curr, total: print(f"  Export: {curr}/{total} frames ({100*curr//total}%)", end='\r')
            )

            if success:
                print(f"\n\n✅ Video exported successfully!")
                print(f"   Path: {output_path}")

                # Check file size
                if os.path.exists(output_path):
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"   Size: {size_mb:.2f} MB")
            else:
                print(f"\n❌ Export failed")
        else:
            print("\nExport cancelled by user / ייצוא בוטל על ידי משתמש")

    else:
        print("\n" + "=" * 70)
        print("❌ User cancelled / משתמש ביטל")
        print("=" * 70)

    print("\n" + "=" * 70)
    print("🎬 Test Complete / בדיקה הושלמה")
    print("=" * 70)

    return True


if __name__ == "__main__":
    print("\n🎬 Three-Phase Tracking System - Complete Workflow Test")
    print("מערכת מעקב תלת-שלבית - בדיקת זרימת עבודה מלאה\n")

    if len(sys.argv) < 2:
        print("Usage / שימוש:")
        print(f"  python {sys.argv[0]} <path/to/video.mp4>")
        print("\nExample / דוגמה:")
        print(f"  python {sys.argv[0]} ~/Videos/test.mp4")
        print("\nNote: Make sure you have a video file ready for testing")
        print("הערה: וודא שיש לך קובץ וידאו מוכן לבדיקה")
        sys.exit(1)

    video_path = sys.argv[1]
    test_complete_workflow(video_path)
