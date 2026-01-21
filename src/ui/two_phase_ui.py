"""
Two-Phase Tracking UI - Complete single-screen interface
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider,
    QListWidget, QListWidgetItem, QWidget, QProgressBar, QCheckBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFrame, QLineEdit, QMessageBox, QComboBox,
    QSizePolicy, QFileDialog, QGridLayout, QSpacerItem, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor
import cv2
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from ..tracking.tracker_manager import TrackerManager
from ..tracking.person_detector import PersonDetector
from ..render.video_exporter import VideoExporter
from ..render.team_manager import get_team_manager

COLORS = {
    'bg_dark': '#1e1e1e',
    'bg_medium': '#252526',
    'bg_light': '#2d2d30',
    'accent': '#0e639c',
    'success': '#16825d',
    'warning': '#c87d11',
    'error': '#d13438',
    'text': '#ffffff',
    'text_muted': '#cccccc',
}

class CompactConfidenceGraph(QWidget):
    frame_clicked = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracking_data, self.total_frames, self.current_frame = {}, 0, 0
        self.setMinimumHeight(52); self.setMaximumHeight(80)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    def set_data(self, tracking_data, total_frames): self.tracking_data, self.total_frames = tracking_data, total_frames; self.update()
    def set_current_frame(self, frame_idx): self.current_frame = frame_idx; self.update()
    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(COLORS['bg_medium']))
        if not self.tracking_data or self.total_frames == 0: painter.end(); return
        bw = max(1, w / self.total_frames)
        for f_idx in range(self.total_frames):
            x = int(f_idx * bw); d = self.tracking_data.get(f_idx, {})
            conf, is_learn = d.get('confidence', 0.0), d.get('is_learning_frame', False)
            color = QColor(255, 215, 0) if is_learn else QColor(COLORS['success']) if conf >= 0.7 else QColor(COLORS['warning']) if conf >= 0.6 else QColor(COLORS['error']) if conf > 0 else QColor(60, 60, 60)
            bh = int(h * conf) if conf > 0 else 2
            painter.fillRect(x, h - bh, int(bw) + 1, bh, color)
        cx = int(self.current_frame * bw); painter.setPen(QPen(QColor(255, 255, 255), 2)); painter.drawLine(cx, 0, cx, h); painter.end()
    def mousePressEvent(self, event):
        if self.total_frames == 0: return
        f_idx = int((event.pos().x() / self.width()) * self.total_frames)
        self.frame_clicked.emit(max(0, min(f_idx, self.total_frames - 1)))

class VideoPreviewWidget(QLabel):
    person_clicked = pyqtSignal(int); bbox_drawn = pyqtSignal(tuple)
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumSize(800, 450); self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border: 2px solid {COLORS['accent']};")
        self.current_frame, self.detected_people, self.player_bboxes = None, [], []
        self.scale_factor, self.offset_x, self.offset_y = 1.0, 0, 0
        self.manual_drawing_mode, self.drawing, self.start_point, self.current_bbox = False, False, None, None
    def set_frame(self, frame, detected_people=None, player_bboxes=None):
        self.current_frame, self.detected_people, self.player_bboxes = frame.copy(), detected_people or [], player_bboxes or []; self._update_display()
    def set_manual_drawing_mode(self, enabled):
        self.manual_drawing_mode = enabled; self.setCursor(QCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor))
    def _update_display(self):
        if self.current_frame is None: return
        df = self.current_frame.copy()
        for i, (x, y, w, h, conf) in enumerate(self.detected_people):
            cv2.rectangle(df, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(df, f"Person {i + 1} ({conf:.0%})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for x, y, w, h, name, color in self.player_bboxes:
            cv2.rectangle(df, (x, y), (x + w, y + h), color, 2)
            cv2.putText(df, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if self.current_bbox:
            x, y, w, h = self.current_bbox; cv2.rectangle(df, (x, y), (x + w, y + h), (255, 255, 0), 2)
        rgb = cv2.cvtColor(df, cv2.COLOR_BGR2RGB); h, w, ch = rgb.shape; bytes_per_line = ch * w
        px = QPixmap.fromImage(QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888))
        ws = self.size(); s = min(ws.width() / px.width(), ws.height() / px.height())
        sp = px.scaled(int(px.width() * s), int(px.height() * s), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.scale_factor, self.offset_x, self.offset_y = s, (ws.width() - sp.width()) // 2, (ws.height() - sp.height()) // 2; self.setPixmap(sp)
    def mousePressEvent(self, event):
        if self.current_frame is None or self.scale_factor == 0: return
        fx = int((event.pos().x() - self.offset_x) / self.scale_factor); fy = int((event.pos().y() - self.offset_y) / self.scale_factor)
        h, w = self.current_frame.shape[:2]; fx, fy = max(0, min(fx, w - 1)), max(0, min(fy, h - 1))
        if self.manual_drawing_mode: self.drawing, self.start_point, self.current_bbox = True, (fx, fy), None; return
        if self.detected_people:
            for i, (x, y, w, h, conf) in enumerate(self.detected_people):
                if x <= fx <= x + w and y <= fy <= y + h: self.person_clicked.emit(i); return
    def mouseMoveEvent(self, event):
        if not self.drawing or not self.manual_drawing_mode: return
        fx = int((event.pos().x() - self.offset_x) / self.scale_factor); fy = int((event.pos().y() - self.offset_y) / self.scale_factor)
        x1, y1 = self.start_point; self.current_bbox = (min(x1, fx), min(y1, fy), abs(fx - x1), abs(fy - y1)); self._update_display()
    def mouseReleaseEvent(self, event):
        if not self.drawing or not self.manual_drawing_mode: return
        self.drawing = False
        if self.current_bbox and self.current_bbox[2] > 10 and self.current_bbox[3] > 10: self.bbox_drawn.emit(self.current_bbox)
        self.current_bbox = None; self._update_display()

class TrackingThread(QThread):
    progress = pyqtSignal(int, int); finished = pyqtSignal(dict); error = pyqtSignal(str)
    def __init__(self, tm, pids, s, e): super().__init__(); self.tm, self.pids, self.s, self.e, self.stop_flag = tm, pids, s, e, False
    def run(self):
        try:
            rs = self.tm.get_resume_start(self.s); data = self.tm.generate_tracking_data(start_frame=rs, end_frame=self.e, progress_callback=self._cb)
            if not self.stop_flag: self.finished.emit(data)
        except Exception as e:
            if not self.stop_flag: self.error.emit(str(e))
    def _cb(self, c, t):
        if self.stop_flag: raise InterruptedError("Stopped")
        self.progress.emit(c, t)
    def stop(self): self.stop_flag = True

class ExportThread(QThread):
    progress = pyqtSignal(int, str, int, int); finished = pyqtSignal(bool)
    def __init__(self, ve, vp, td, op, s, e): super().__init__(); self.ve, self.vp, self.td, self.op, self.s, self.e, self.cancelled = ve, vp, td, op, s, e, False
    def run(self):
        success = self.ve.export_tracked_video(self.vp, self.td, self.op, progress_callback=lambda c,t,st: self.progress.emit(int(c), st, c, t), tracking_start_frame=self.s, tracking_end_frame=self.e, should_cancel=lambda: self.cancelled)
        self.finished.emit(success)
    def cancel(self): self.cancelled = True

class PlayerListItemWidget(QWidget):
    def __init__(self, pid, name, style, parent=None):
        super().__init__(parent); self.pid = pid; l = QHBoxLayout(); l.setContentsMargins(4, 2, 4, 2)
        self.checkbox = QCheckBox(); self.checkbox.setChecked(True); l.addWidget(self.checkbox)
        icons = {'dynamic_ring_3d': '🟣', 'spotlight_alien': '👽', 'solid_anchor': '⚓', 'radar_defensive': '📡', 'sniper_scope': '🎯'}
        label = QLabel(f"{icons.get(style, '👤')} {name}"); label.setStyleSheet("color: #ffffff; font-weight: 600;"); l.addWidget(label, stretch=1)
        self.setLayout(l)

class TwoPhaseTrackingUI(QDialog):
    def __init__(self, tm, parent=None):
        super().__init__(parent); self.tm, self.pd, self.ve = tm, PersonDetector(), VideoExporter(tm); self.cfi, self.tf, self.td, self.spi, self.tt = 0, tm.total_frames, {}, None, None
        self.cm, self.am, self.rf, self.sf, self.ef = False, "NONE", {}, 0, tm.total_frames - 1
        self.setWindowTitle("Two-Phase Tracking - Phase 1: Review & Track"); self.resize(1300, 800); self.setMinimumSize(1200, 720); self._setup_ui(); self._load_frame(0)
    def _setup_ui(self):
        ml = QHBoxLayout(); ml.setSpacing(16); ml.setContentsMargins(16, 16, 16, 16)
        lw = QWidget(); ll = QVBoxLayout(); ll.setSpacing(10); ll.setContentsMargins(0, 0, 0, 0)
        self.vp = VideoPreviewWidget(); self.vp.person_clicked.connect(self._on_person_clicked); ll.addWidget(self.vp)
        gl = QLabel("📊 Confidence Timeline:"); gl.setStyleSheet("color: #ffffff; font-weight: 600; font-size: 13px;"); ll.addWidget(gl)
        self.cg = CompactConfidenceGraph(); self.cg.frame_clicked.connect(self._jump_to_frame); ll.addWidget(self.cg)
        nl = QHBoxLayout(); self.fl = QLabel("Frame: 0/0"); self.fl.setStyleSheet("color: #ffffff; font-weight: 600;"); nl.addWidget(self.fl); nl.addStretch()
        self.bf, self.bp = QPushButton("⏮"), QPushButton("◀"); [b.setFixedWidth(40) for b in (self.bf, self.bp)]; self.bf.clicked.connect(lambda: self._jump_to_frame(0)); self.bp.clicked.connect(self._prev_frame); nl.addWidget(self.bf); nl.addWidget(self.bp)
        self.fs = QSlider(Qt.Orientation.Horizontal); self.fs.setMinimum(0); self.fs.setMaximum(self.tf - 1); self.fs.valueChanged.connect(self._on_slider_changed); nl.addWidget(self.fs, stretch=1)
        self.bn, self.bl = QPushButton("▶"), QPushButton("⏭"); [b.setFixedWidth(40) for b in (self.bn, self.bl)]; self.bn.clicked.connect(self._next_frame); self.bl.clicked.connect(lambda: self._jump_to_frame(self.tf - 1)); nl.addWidget(self.bn); nl.addWidget(self.bl); ll.addLayout(nl)
        self.il = QLabel("Use 'Detect People' to find and add players →"); self.il.setStyleSheet("color: #ffffff; font-weight: 600; font-size: 13px; padding: 5px;"); ll.addWidget(self.il); lw.setLayout(ll); ml.addWidget(lw, stretch=2)
        ri = QWidget(); rl = QVBoxLayout(); rl.setSpacing(12); rl.setContentsMargins(12, 12, 12, 12); rl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.pg = QGroupBox("👥 Players"); pl = QVBoxLayout(); self.pl = QListWidget(); self.pl.itemClicked.connect(self._on_player_selected); pl.addWidget(self.pl)
        bl = QHBoxLayout(); self.bep, self.brp = QPushButton("✏️ Edit"), QPushButton("🗑️ Remove"); [b.setEnabled(False) for b in (self.bep, self.brp)]; [b.setMinimumSize(QSize(100, 40)) for b in (self.bep, self.brp)]; self.bep.clicked.connect(self._edit_player); self.brp.clicked.connect(self._remove_player); bl.addWidget(self.bep); bl.addWidget(self.brp); pl.addLayout(bl); self.pg.setLayout(pl); rl.addWidget(self.pg)
        rg = QGroupBox("📹 Tracking Range"); rgl = QVBoxLayout(); sl = QHBoxLayout(); sl.addWidget(QLabel("Start:")); self.sfs = QSpinBox(); self.sfs.setMaximum(self.tf - 1); self.sfs.valueChanged.connect(self._on_range_changed); sl.addWidget(self.sfs, stretch=1); rgl.addLayout(sl)
        self.bss = QPushButton("⬅️ Set Start"); self.bss.clicked.connect(lambda: self.sfs.setValue(self.cfi)); rgl.addWidget(self.bss); el = QHBoxLayout(); el.addWidget(QLabel("End:")); self.efs = QSpinBox(); self.efs.setMaximum(self.tf - 1); self.efs.setValue(self.tf - 1); self.efs.valueChanged.connect(self._on_range_changed); el.addWidget(self.efs, stretch=1); rgl.addLayout(el)
        self.bse = QPushButton("➡️ Set End"); self.bse.clicked.connect(lambda: self.efs.setValue(self.cfi)); rgl.addWidget(self.bse); rg.setLayout(rgl); rl.addWidget(rg)
        ag = QGroupBox("⚙️ Actions"); al = QVBoxLayout(); al.setSpacing(10)
        def _sb(txt, var="ghost", h=44): b = QPushButton(txt); b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor)); b.setMinimumHeight(h); b.setProperty("variant", var); return b
        tr = QHBoxLayout(); self.bd = _sb("🔍 Detect People"); self.bd.clicked.connect(self._detect_people); self.bmd = _sb("✏️ Manual Draw"); self.bmd.setCheckable(True); self.bmd.clicked.connect(self._toggle_manual_draw); tr.addWidget(self.bd); tr.addWidget(self.bmd); al.addLayout(tr)
        cr = QHBoxLayout(); self.bac = _sb("✏️ Draw Correction"); self.bac.clicked.connect(self._start_correction_mode); self.bdc = _sb("🔍 Detect Correct"); self.bdc.clicked.connect(self._start_detect_correction_mode); cr.addWidget(self.bac); cr.addWidget(self.bdc); al.addLayout(cr)
        sr = QHBoxLayout(); self.bst = _sb("▶️ Start Tracking", "positive", 48); self.bst.clicked.connect(self._start_tracking); self.bpt = _sb("⏸️ Stop", "danger", 48); self.bpt.setEnabled(False); self.bpt.clicked.connect(self._stop_tracking); sr.addWidget(self.bst); sr.addWidget(self.bpt); al.addLayout(sr)
        self.tmc = QComboBox(); self.tmc.addItems(["Legacy (CSRT only)", "Hybrid (CSRT + YOLO) ⭐"]); self.tmc.setCurrentIndex(1); self.tmc.currentIndexChanged.connect(self._on_tracking_config_changed); al.addWidget(self.tmc)
        thr1 = QHBoxLayout(); self.ispin = QDoubleSpinBox(); self.ispin.setRange(0, 1); self.ispin.setValue(0.15); self.ispin.setPrefix("IoU≥ "); self.sspin = QDoubleSpinBox(); self.sspin.setRange(0, 1.5); self.sspin.setValue(0.35); self.sspin.setPrefix("ScaleΔ "); thr1.addWidget(self.ispin); thr1.addWidget(self.sspin); al.addLayout(thr1)
        self.pbar = QProgressBar(); self.pbar.setVisible(False); al.addWidget(self.pbar); self.brt = _sb("🔄 Re-track"); self.brt.setEnabled(False); self.brt.clicked.connect(self._retrack); al.addWidget(self.brt)
        epr = QHBoxLayout(); self.exprogress = QProgressBar(); self.exprogress.setVisible(False); self.exprogress.setFormat("Exporting… %p%"); self.bce = _sb("❌ Cancel Export", "danger", 40); self.bce.setEnabled(False); self.bce.clicked.connect(self._cancel_export); epr.addWidget(self.exprogress, 1); epr.addWidget(self.bce); al.addLayout(epr); ag.setLayout(al); rl.addWidget(ag); rl.addStretch()
        self.bload = _sb("📂 Load Data", "accent", 45); self.bload.clicked.connect(self._on_load_clicked); rl.addWidget(self.bload)
        self.bex = _sb("✅ Export / Save", "accent-strong", 56); self.bex.setEnabled(False); self.bex.clicked.connect(self._on_export_clicked); rl.addWidget(self.bex)
        rs = QScrollArea(); rs.setWidgetResizable(True); rs.setWidget(ri); rs.setMinimumWidth(360); ml.addWidget(rs, stretch=1); self.setLayout(ml); self._apply_theme()
    def _apply_theme(self): self.setStyleSheet(f"* {{ background-color: {COLORS['bg_dark']}; color: {COLORS['text']}; }} QGroupBox {{ border: 1px solid #3e3e42; border-radius: 6px; margin-top: 12px; padding-top: 12px; }} QPushButton {{ background-color: #273141; border-radius: 10px; padding: 10px; font-weight: 600; }} QPushButton[variant='positive'] {{ background-color: {COLORS['success']}; }} QPushButton[variant='danger'] {{ background-color: {COLORS['error']}; }} QPushButton[variant='accent-strong'] {{ background-color: {COLORS['accent']}; }} QListWidget {{ background-color: {COLORS['bg_light']}; border-radius: 8px; }}")
    def _update_player_list(self):
        cp = {self.pl.itemWidget(self.pl.item(i)).pid for i in range(self.pl.count()) if self.pl.itemWidget(self.pl.item(i)).checkbox.isChecked()}
        ps = self.spi; self.pl.clear(); fi = None
        for pid, p in self.tm.players.items():
            item = QListWidgetItem(self.pl); w = PlayerListItemWidget(pid, p.name, p.marker_style); item.setSizeHint(w.sizeHint()); self.pl.setItemWidget(item, w)
            if pid in cp or not cp: w.checkbox.setChecked(True)
            if fi is None: fi = item
            if ps and pid == ps: self.pl.setCurrentItem(item); self._on_player_selected(item)
        if self.pl.count() > 0 and self.spi is None and fi: self.pl.setCurrentItem(fi); self._on_player_selected(fi)
    def _load_frame(self, fidx):
        self.cfi = fidx; f = self.tm.get_frame(fidx)
        if f is None: return
        pbs = []
        for pid, pd in self.td.items():
            if fidx in pd:
                bb = pd[fidx].get('bbox')
                if bb: p = self.tm.players.get(pid); pbs.append((*bb, p.name, (0, 255, 0) if pid == self.spi else (150, 150, 150)))
        self.vp.set_frame(f, self.detected_people, pbs); self.fl.setText(f"Frame: {fidx}/{self.tf - 1}"); self.fs.blockSignals(True); self.fs.setValue(fidx); self.fs.blockSignals(False)
        if self.spi and self.spi in self.td: self.cg.set_current_frame(fidx); p_data = self.td[self.spi].get(fidx, {}); conf = p_data.get('confidence', 0.0); q = "Good" if conf >= 0.7 else "Med" if conf >= 0.4 else "Poor" if conf > 0 else "Lost"; p = self.tm.players.get(self.spi); self.il.setText(f"Selected: {p.name if p else '?'} | Conf: {conf:.2f} | Q: {q}")
    def _on_slider_changed(self, v): self._load_frame(v)
    def _on_tracking_config_changed(self, *_): self.tm.update_tracking_config(mode="hybrid" if self.tmc.currentIndex() == 1 else "legacy", iou_min=float(self.ispin.value()), scale_change_max=float(self.sspin.value()))
    def _prev_frame(self): self._load_frame(max(0, self.cfi - 1))
    def _next_frame(self): self._load_frame(min(self.tf - 1, self.cfi + 1))
    def _jump_to_frame(self, fi): self._load_frame(fi)
    def _on_player_selected(self, it):
        w = self.pl.itemWidget(it)
        if w: self.spi = w.pid; [b.setEnabled(True) for b in (self.bep, self.brp)]; self.cg.set_data(self.td.get(self.spi, {}), self.tf); self._load_frame(self.cfi)
    def _edit_player(self):
        if not self.spi: return
        from PyQt6.QtWidgets import QInputDialog; nn, ok = QInputDialog.getText(self, "Edit Name", "New name:", text=self.tm.players[self.spi].name)
        if ok and nn.strip(): self.tm.players[self.spi].name = nn.strip(); self._update_player_list(); self._load_frame(self.cfi)
    def _remove_player(self):
        if self.spi and QMessageBox.question(self, "Remove", f"Remove {self.tm.players[self.spi].name}?") == QMessageBox.StandardButton.Yes: self.tm.remove_player(self.spi); self.td.pop(self.spi, None); self.spi = None; self._update_player_list(); self._load_frame(self.cfi)
    def _on_range_changed(self): self.sf, self.ef = self.sfs.value(), self.efs.value(); self.efs.setValue(max(self.sf, self.ef))
    def _detect_people(self):
        if not self.pd.is_available(): QMessageBox.warning(self, "Error", "YOLO not loaded"); return
        f = self.tm.get_frame(self.cfi)
        if f is not None: self.am, self.detected_people = "DETECT_PEOPLE", self.pd.detect_people(f, 0.25); self._load_frame(self.cfi)
    def _toggle_manual_draw(self, c):
        if not c: self.cm, self.am = False, "NONE" if self.am == "MANUAL_DRAW" else self.am
        self.vp.set_manual_drawing_mode(c)
        if c:
            self.am, self.detected_people = "MANUAL_DRAW", []; self.bd.setEnabled(False); self._load_frame(self.cfi)
            try: self.vp.bbox_drawn.disconnect()
            except: pass
            self.vp.bbox_drawn.connect(self._on_bbox_drawn)
        else: self.bd.setEnabled(True)
    def _on_bbox_drawn(self, bb):
        self.bmd.setChecked(False); self.vp.set_manual_drawing_mode(False); self.bd.setEnabled(True)
        if self.cm: self.cm, self.am = False, "NONE"; self.tm.add_learning_frame_to_player(self.spi, self.cfi, bb, bb); return
        self.am = "NONE"; self._show_add_player_dialog(bb, True)
    def _on_person_clicked(self, idx):
        if idx >= len(self.detected_people): return
        bb, conf = self.detected_people[idx][:4], self.detected_people[idx][4]
        if self.cm and self.am == "CORRECTION": self.tm.add_learning_frame_to_player(self.spi, self.cfi, bb, bb); self.cm, self.am, self.detected_people = False, "NONE", []; self._load_frame(self.cfi); return
        self._show_add_player_dialog(bb, False, conf)
    def _show_add_player_dialog(self, bb, manual=False, conf=None):
        d = QDialog(self); d.setWindowTitle("Add Player"); l = QVBoxLayout(); l.addWidget(QLabel(f"Bbox: {bb}" if manual else f"Detected person ({conf:.0%})"))
        nl = QHBoxLayout(); nl.addWidget(QLabel("Name:")); ni = QLineEdit(); ni.setPlaceholderText(f"Player {len(self.tm.players)+1}"); nl.addWidget(ni); l.addLayout(nl)
        sl = QHBoxLayout(); sl.addWidget(QLabel("Marker:")); sc = QComboBox(); sc.addItems(["🟣 Dynamic Ring 3D", "👽 Alien Spotlight", "⚓ Solid Anchor", "📡 Radar Defensive", "🎯 Sniper Scope"]); sl.addWidget(sc); l.addLayout(sl)
        bl = QHBoxLayout(); bc, ba = QPushButton("Cancel"), QPushButton("Add"); bc.clicked.connect(d.reject); ba.clicked.connect(d.accept); bl.addWidget(bc); bl.addWidget(ba); l.addLayout(bl); d.setLayout(l)
        if d.exec() == QDialog.DialogCode.Accepted:
            st_map = {0: "dynamic_ring_3d", 1: "spotlight_alien", 2: "solid_anchor", 3: "radar_defensive", 4: "sniper_scope"}
            pid = self.tm.add_player(ni.text().strip() or ni.placeholderText(), st_map.get(sc.currentIndex(), "dynamic_ring_3d"), self.cfi, bb)
            self.spi, self.detected_people = pid, []; self._update_player_list(); self._load_frame(self.cfi)
    def _start_tracking(self):
        pids = [self.pl.itemWidget(self.pl.item(i)).pid for i in range(self.pl.count()) if self.pl.itemWidget(self.pl.item(i)).checkbox.isChecked()]
        if not pids: QMessageBox.warning(self, "Error", "Select players"); return
        self.tt = TrackingThread(self.tm, pids, self.sf, self.ef); self.tt.progress.connect(lambda c,t: self.pbar.setValue(int(c/t*100))); self.tt.finished.connect(self._on_tracking_finished); self.tt.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.bst.setEnabled(False); self.bpt.setEnabled(True); self.pbar.setVisible(True); self.tt.start()
    def _stop_tracking(self):
        if self.tt: self.tt.stop(); self.bpt.setEnabled(False)
    def _on_tracking_finished(self, data):
        self.td = data; self.tt = None; self.bst.setEnabled(True); self.bpt.setEnabled(False); self.brt.setEnabled(True); self.bex.setEnabled(True); self.pbar.setVisible(False)
        if not self.spi and self.td: self.spi = next(iter(self.td.keys())); self._update_player_list()
        if self.spi in self.td: self.cg.set_data(self.td[self.spi], self.tf)
        self.tm.video_cap.release(); self.tm.video_cap = cv2.VideoCapture(self.tm.video_path); self._load_frame(self.cfi)
    def _retrack(self):
        if QMessageBox.question(self, "Re-track", "Retrack with corrections?") == QMessageBox.StandardButton.Yes: self._start_tracking()
    def _start_correction_mode(self): self.cm, self.am = True, "CORRECTION"; self.bmd.setChecked(True); self._toggle_manual_draw(True)
    def _start_detect_correction_mode(self):
        if not self.spi: return
        self.cm, self.am, f = True, "CORRECTION", self.tm.get_frame(self.cfi)
        if f is not None: self.detected_people = self.pd.detect_people(f, 0.3); self._load_frame(self.cfi)
    def _on_load_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load", "", "JSON Files (*.json)")
        if not path: return
        try:
            with open(path, 'r') as f: ld = json.load(f)
            if 'players' in ld:
                for pid_s, pi in ld['players'].items():
                    pid = int(pid_s)
                    if pid not in self.tm.players: self.tm.players[pid] = PlayerData(pid, pi['name'], pi['marker_style'], 0, (0,0,0,0)); self.tm.next_player_id = max(self.tm.next_player_id, pid + 1)
                    # Load team assignment if present
                    if 'team' in pi and pi['team'] in ('A', 'B'):
                        get_team_manager().assign_team(pid, pi['team'])
                self.td = {int(p): {int(f): v for f, v in fs.items()} for p, fs in ld['tracking_data'].items()}
            self._update_player_list(); self.bex.setEnabled(True); self._load_frame(self.cfi)
        except Exception as e: QMessageBox.critical(self, "Error", str(e))
    def _on_export_clicked(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save", "tracking_data.json", "JSON (*.json)")
        if not p: return
        # Build players dict with team assignments
        players_data = {}
        team_mgr = get_team_manager()
        for pid, pl in self.tm.players.items():
            player_info = {'name': pl.name, 'marker_style': pl.marker_style}
            team = team_mgr.get_team(pid)
            if team: player_info['team'] = team
            players_data[str(pid)] = player_info
        s = {'players': players_data, 'tracking_data': {str(pid): {str(f): v for f, v in fs.items()} for pid, fs in self.td.items()}}
        try:
            with open(p, 'w') as f: json.dump(s, f, indent=2)
            if QMessageBox.question(self, "Render", "Data saved. Export video?") == QMessageBox.StandardButton.Yes:
                vp, _ = QFileDialog.getSaveFileName(self, "Save Video", "tracked_output.mp4", "Video (*.mp4)")
                if vp: self._start_export(vp)
        except Exception as e: QMessageBox.critical(self, "Error", str(e))
    def _start_export(self, vp):
        [b.setEnabled(False) for b in (self.bex, self.bst, self.bpt, self.brt, self.bac, self.bdc, self.bmd, self.bd)]; self.exprogress.setVisible(True); self.bce.setEnabled(True)
        self.et = ExportThread(self.ve, self.tm.video_path, self.td, vp, self.sf, self.ef); self.et.progress.connect(self._on_export_progress); self.et.finished.connect(self._on_export_finished); self.et.start()
    def _on_export_progress(self, p, st, c, t): self.exprogress.setValue(p); self.exprogress.setFormat(f"{st}… {p}%")
    def _on_export_finished(self, s):
        self.exprogress.setVisible(False); self.bce.setEnabled(False); [b.setEnabled(True) for b in (self.bex, self.bst, self.bpt, self.brt, self.bac, self.bdc, self.bmd, self.bd)]
        if s: QMessageBox.information(self, "Success", "Exported"); self.accept()
        else: QMessageBox.warning(self, "Error", "Export failed")
    def _cancel_export(self):
        if hasattr(self, "et"): self.et.cancel(); self.bce.setEnabled(False); self.exprogress.setFormat("Cancelling…")
