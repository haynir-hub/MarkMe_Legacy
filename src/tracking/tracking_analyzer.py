"""
Tracking Analyzer - Automatic detection of tracking issues
מנתח מעקב - זיהוי אוטומטי של בעיות במעקב
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class TrackingIssue:
    """Data class for tracking issue"""
    def __init__(self, frame_idx: int, issue_type: str, severity: str, description: str, confidence: float = 0.0):
        self.frame_idx = frame_idx
        self.issue_type = issue_type  # 'lost', 'low_confidence', 'sudden_jump', 'size_change', 'edge'
        self.severity = severity  # 'critical', 'high', 'medium', 'low'
        self.description = description
        self.confidence = confidence

    def __repr__(self):
        return f"TrackingIssue(frame={self.frame_idx}, type={self.issue_type}, severity={self.severity})"


class TrackingAnalyzer:
    """Analyzes tracking data to detect potential issues"""

    def __init__(self):
        # Configurable thresholds
        self.confidence_threshold = 0.5  # Below this is considered low confidence
        self.critical_confidence_threshold = 0.3  # Below this is critical
        self.max_bbox_jump = 100  # Maximum pixel movement between frames
        self.max_size_change_ratio = 2.0  # Maximum size change ratio (e.g., 2x bigger/smaller)
        self.edge_margin = 20  # Pixels from edge to consider as "near edge"

    def analyze(self, tracking_data: Dict[int, Dict[str, any]],
                frame_width: int, frame_height: int) -> List[TrackingIssue]:
        """
        Analyze tracking data and return list of issues

        Args:
            tracking_data: Dictionary {frame_idx: {'bbox': ..., 'confidence': ..., 'is_learning_frame': ...}}
            frame_width: Video frame width
            frame_height: Video frame height

        Returns:
            List of TrackingIssue objects, sorted by frame index
        """
        issues = []

        if not tracking_data:
            return issues

        frames = sorted(tracking_data.keys())
        prev_bbox = None
        prev_frame_idx = None

        for frame_idx in frames:
            data = tracking_data[frame_idx]
            bbox = data.get('bbox')
            confidence = data.get('confidence', 0.0)
            is_learning = data.get('is_learning_frame', False)

            # Skip learning frames (they're manually marked, so they're correct)
            if is_learning:
                prev_bbox = bbox
                prev_frame_idx = frame_idx
                continue

            # Issue 1: Lost tracking
            if bbox is None:
                issues.append(TrackingIssue(
                    frame_idx=frame_idx,
                    issue_type='lost',
                    severity='critical',
                    description='מעקב אבוד לחלוטין - Tracking completely lost',
                    confidence=0.0
                ))
                prev_bbox = None
                prev_frame_idx = frame_idx
                continue

            # Issue 2: Low confidence
            if confidence < self.critical_confidence_threshold:
                issues.append(TrackingIssue(
                    frame_idx=frame_idx,
                    issue_type='low_confidence',
                    severity='critical',
                    description=f'ביטחון קריטי נמוך ({confidence:.2f}) - Critically low confidence',
                    confidence=confidence
                ))
            elif confidence < self.confidence_threshold:
                issues.append(TrackingIssue(
                    frame_idx=frame_idx,
                    issue_type='low_confidence',
                    severity='high',
                    description=f'ביטחון נמוך ({confidence:.2f}) - Low confidence',
                    confidence=confidence
                ))

            # Check bbox-based issues only if we have a valid bbox
            if bbox and prev_bbox:
                # Issue 3: Sudden jump (large movement between consecutive frames)
                prev_x, prev_y, prev_w, prev_h = prev_bbox
                curr_x, curr_y, curr_w, curr_h = bbox

                # Calculate center points
                prev_center_x = prev_x + prev_w / 2
                prev_center_y = prev_y + prev_h / 2
                curr_center_x = curr_x + curr_w / 2
                curr_center_y = curr_y + curr_h / 2

                # Calculate distance
                distance = np.sqrt((curr_center_x - prev_center_x)**2 +
                                 (curr_center_y - prev_center_y)**2)

                # Account for frame gaps (if frames are not consecutive)
                frame_gap = frame_idx - prev_frame_idx if prev_frame_idx else 1
                adjusted_threshold = self.max_bbox_jump * frame_gap

                if distance > adjusted_threshold:
                    severity = 'critical' if distance > adjusted_threshold * 2 else 'high'
                    issues.append(TrackingIssue(
                        frame_idx=frame_idx,
                        issue_type='sudden_jump',
                        severity=severity,
                        description=f'קפיצה חדה ({distance:.0f} פיקסלים) - Sudden jump ({distance:.0f} pixels)',
                        confidence=confidence
                    ))

                # Issue 4: Drastic size change
                prev_size = prev_w * prev_h
                curr_size = curr_w * curr_h

                if prev_size > 0:
                    size_ratio = curr_size / prev_size
                    if size_ratio > self.max_size_change_ratio or size_ratio < (1 / self.max_size_change_ratio):
                        severity = 'high' if size_ratio > self.max_size_change_ratio * 1.5 or size_ratio < (1 / (self.max_size_change_ratio * 1.5)) else 'medium'
                        issues.append(TrackingIssue(
                            frame_idx=frame_idx,
                            issue_type='size_change',
                            severity=severity,
                            description=f'שינוי גודל דרסטי (x{size_ratio:.2f}) - Drastic size change',
                            confidence=confidence
                        ))

            # Issue 5: Bbox near frame edge (might indicate tracking drift)
            if bbox:
                x, y, w, h = bbox
                near_left = x < self.edge_margin
                near_top = y < self.edge_margin
                near_right = (x + w) > (frame_width - self.edge_margin)
                near_bottom = (y + h) > (frame_height - self.edge_margin)

                if any([near_left, near_top, near_right, near_bottom]):
                    edges = []
                    if near_left: edges.append('שמאל-left')
                    if near_top: edges.append('עליון-top')
                    if near_right: edges.append('ימין-right')
                    if near_bottom: edges.append('תחתון-bottom')

                    issues.append(TrackingIssue(
                        frame_idx=frame_idx,
                        issue_type='edge',
                        severity='medium',
                        description=f'קרוב לקצה ({", ".join(edges)}) - Near edge',
                        confidence=confidence
                    ))

            prev_bbox = bbox
            prev_frame_idx = frame_idx

        return issues

    def get_summary(self, issues: List[TrackingIssue]) -> Dict[str, any]:
        """
        Get summary statistics of issues

        Args:
            issues: List of TrackingIssue objects

        Returns:
            Dictionary with summary statistics
        """
        if not issues:
            return {
                'total': 0,
                'by_type': {},
                'by_severity': {},
                'frames_affected': []
            }

        # Count by type
        by_type = {}
        for issue in issues:
            by_type[issue.issue_type] = by_type.get(issue.issue_type, 0) + 1

        # Count by severity
        by_severity = {}
        for issue in issues:
            by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1

        # Get unique affected frames
        frames_affected = sorted(set([issue.frame_idx for issue in issues]))

        return {
            'total': len(issues),
            'by_type': by_type,
            'by_severity': by_severity,
            'frames_affected': frames_affected,
            'critical_frames': [issue.frame_idx for issue in issues if issue.severity == 'critical']
        }

    def suggest_corrections(self, issues: List[TrackingIssue],
                          tracking_data: Dict[int, Dict[str, any]]) -> List[Tuple[int, str]]:
        """
        Suggest frames that should be manually corrected

        Args:
            issues: List of TrackingIssue objects
            tracking_data: Original tracking data

        Returns:
            List of (frame_idx, reason) tuples suggesting manual corrections
        """
        suggestions = []

        # Group issues by frame
        issues_by_frame = {}
        for issue in issues:
            if issue.frame_idx not in issues_by_frame:
                issues_by_frame[issue.frame_idx] = []
            issues_by_frame[issue.frame_idx].append(issue)

        # Prioritize frames for correction
        for frame_idx, frame_issues in sorted(issues_by_frame.items()):
            # Critical issues should always be corrected
            critical_issues = [i for i in frame_issues if i.severity == 'critical']
            if critical_issues:
                reason = ', '.join([i.issue_type for i in critical_issues])
                suggestions.append((frame_idx, f"Critical: {reason}"))
                continue

            # Multiple high-severity issues
            high_issues = [i for i in frame_issues if i.severity == 'high']
            if len(high_issues) >= 2:
                reason = ', '.join([i.issue_type for i in high_issues])
                suggestions.append((frame_idx, f"Multiple issues: {reason}"))
                continue

            # Sudden jumps combined with low confidence
            has_jump = any(i.issue_type == 'sudden_jump' for i in frame_issues)
            has_low_conf = any(i.issue_type == 'low_confidence' for i in frame_issues)
            if has_jump and has_low_conf:
                suggestions.append((frame_idx, "Sudden jump + low confidence"))

        return suggestions

    def find_tracking_gaps(self, tracking_data: Dict[int, Dict[str, any]]) -> List[Tuple[int, int]]:
        """
        Find gaps in tracking (sequences of lost frames)

        Args:
            tracking_data: Tracking data dictionary

        Returns:
            List of (start_frame, end_frame) tuples representing gap ranges
        """
        if not tracking_data:
            return []

        frames = sorted(tracking_data.keys())
        gaps = []
        gap_start = None

        for frame_idx in frames:
            bbox = tracking_data[frame_idx].get('bbox')

            if bbox is None:
                # Lost tracking
                if gap_start is None:
                    gap_start = frame_idx
            else:
                # Tracking resumed
                if gap_start is not None:
                    gaps.append((gap_start, frame_idx - 1))
                    gap_start = None

        # Close any open gap
        if gap_start is not None:
            gaps.append((gap_start, frames[-1]))

        return gaps

    def calculate_tracking_quality_score(self, tracking_data: Dict[int, Dict[str, any]],
                                        issues: List[TrackingIssue]) -> float:
        """
        Calculate overall tracking quality score (0.0 - 1.0)

        Args:
            tracking_data: Tracking data dictionary
            issues: List of detected issues

        Returns:
            Quality score (1.0 = perfect, 0.0 = completely failed)
        """
        if not tracking_data:
            return 0.0

        total_frames = len(tracking_data)

        # Start with perfect score
        score = 1.0

        # Penalty for lost frames
        lost_frames = len([f for f, d in tracking_data.items() if d['bbox'] is None])
        lost_penalty = (lost_frames / total_frames) * 0.5
        score -= lost_penalty

        # Penalty for low confidence
        avg_confidence = sum([d['confidence'] for d in tracking_data.values()]) / total_frames
        confidence_penalty = (1.0 - avg_confidence) * 0.3
        score -= confidence_penalty

        # Penalty for issues
        if issues:
            critical_count = len([i for i in issues if i.severity == 'critical'])
            high_count = len([i for i in issues if i.severity == 'high'])
            medium_count = len([i for i in issues if i.severity == 'medium'])

            issues_penalty = (
                (critical_count / total_frames) * 0.15 +
                (high_count / total_frames) * 0.10 +
                (medium_count / total_frames) * 0.05
            )
            score -= issues_penalty

        return max(0.0, min(1.0, score))
