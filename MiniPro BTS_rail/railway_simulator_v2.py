from PySide6 import QtCore, QtGui, QtWidgets
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    TZ_BKK = ZoneInfo("Asia/Bangkok")
except Exception:
    TZ_BKK = None
import math, time

# ========================= Theme (Light + Pastel) =========================
INK         = QtGui.QColor("#0F172A")        # text
INK_SOFT    = QtGui.QColor("#334155")
SURFACE     = "#FFFFFF"
BG_GRAD_1   = "#F7FBFF"
BG_GRAD_2   = "#FFFFFF"
BORDER      = "rgba(15, 23, 42, 0.08)"

# Pastel train palette (auto-rotate)
TRAIN_COLORS = [
    QtGui.QColor("#7CBAFF"),  # Sky
    QtGui.QColor("#B9FBC0"),  # Mint
    QtGui.QColor("#FBCFE8"),  # Pink
    QtGui.QColor("#D8B4FE"),  # Lilac
    QtGui.QColor("#FCA5A5"),  # Salmon
]

GREEN_OK  = QtGui.QColor("#10B981")
RED_BUSY  = QtGui.QColor("#EF4444")
GRAY_LINE = QtGui.QColor("#CBD5E1")
TRACK_FREE_FILL = QtGui.QColor("#F1F5F9")    # base pastel for free blocks

def lerp(a,b,t): return a + (b-a)*t
def clamp(x,a,b): return a if x<a else (b if x>b else x)

# ========================= Tiny vector icons =========================
def make_icon(kind: str, size: int = 18) -> QtGui.QIcon:
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm); p.setRenderHint(QtGui.QPainter.Antialiasing, True)

    def pen(c, w): 
        _p = QtGui.QPen(QtGui.QColor(c)); _p.setWidthF(max(1.2, size*w)); return _p
    def brush(hex_): return QtGui.QBrush(QtGui.QColor(hex_))

    if kind == "route_main":
        p.setPen(pen("#7CBAFF", 0.10)); p.drawLine(int(size*0.15), int(size*0.5), int(size*0.85), int(size*0.5))
        p.setBrush(brush("#7CBAFF")); p.setPen(QtCore.Qt.NoPen)
        r = int(size*0.12); p.drawEllipse(QtCore.QPointF(size*0.5, size*0.5), r, r)
    elif kind == "route_loop":
        p.setPen(pen("#FBCFE8", 0.10)); p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(QtCore.QRectF(size*0.18, size*0.18, size*0.64, size*0.64))
        p.setBrush(brush("#FBCFE8")); p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(QtCore.QRectF(size*0.45, size*0.45, size*0.1, size*0.1))
    elif kind == "call":
        p.setBrush(brush("#B9FBC0")); p.setPen(pen("#B9FBC0", 0.11))
        p.drawRoundedRect(QtCore.QRectF(size*0.18, size*0.25, size*0.46, size*0.5), 3, 3)
        p.setPen(QtCore.Qt.NoPen)
        tri = QtGui.QPolygonF([QtCore.QPointF(size*0.7, size*0.5), QtCore.QPointF(size*0.9, size*0.38), QtCore.QPointF(size*0.9, size*0.62)])
        p.drawPolygon(tri)
    elif kind == "release":
        p.setBrush(brush("#FCA5A5")); p.setPen(pen("#FCA5A5", 0.11))
        p.drawRoundedRect(QtCore.QRectF(size*0.36, size*0.25, size*0.46, size*0.5), 3, 3)
        p.setPen(QtCore.Qt.NoPen)
        tri = QtGui.QPolygonF([QtCore.QPointF(size*0.3, size*0.5), QtCore.QPointF(size*0.1, size*0.38), QtCore.QPointF(size*0.1, size*0.62)])
        p.drawPolygon(tri)
    elif kind == "pause":
        p.setBrush(brush("#94A3B8")); p.setPen(QtCore.Qt.NoPen)
        w = size*0.18; h = size*0.6
        p.drawRoundedRect(QtCore.QRectF(size*0.25, size*0.2, w, h), 2, 2)
        p.drawRoundedRect(QtCore.QRectF(size*0.57, size*0.2, w, h), 2, 2)
    elif kind == "resume":
        p.setBrush(brush("#94A3B8")); p.setPen(QtCore.Qt.NoPen)
        tri = QtGui.QPolygonF([QtCore.QPointF(size*0.3, size*0.2), QtCore.QPointF(size*0.8, size*0.5), QtCore.QPointF(size*0.3, size*0.8)])
        p.drawPolygon(tri)
    elif kind == "reset":
        p.setBrush(QtCore.Qt.NoBrush); p.setPen(pen("#D8B4FE", 0.11))
        p.drawArc(QtCore.QRectF(size*0.18, size*0.18, size*0.6, size*0.6), 45*16, 270*16)
        p.setBrush(brush("#D8B4FE")); p.setPen(QtCore.Qt.NoPen)
        tri = QtGui.QPolygonF([QtCore.QPointF(size*0.75, size*0.26), QtCore.QPointF(size*0.9, size*0.26), QtCore.QPointF(size*0.825, size*0.42)])
        p.drawPolygon(tri)
    elif kind == "speed":
        p.setBrush(QtCore.Qt.NoBrush); p.setPen(pen("#9CA3AF", 0.09))
        p.drawLine(int(size*0.18), int(size*0.75), int(size*0.82), int(size*0.75))
        p.drawLine(int(size*0.18), int(size*0.55), int(size*0.65), int(size*0.55))
        p.drawLine(int(size*0.18), int(size*0.35), int(size*0.5), int(size*0.35))
    else:
        p.setBrush(brush("#94A3B8")); p.setPen(QtCore.Qt.NoPen)
        r = size*0.35; p.drawEllipse(QtCore.QPointF(size/2, size/2), r, r)

    p.end()
    return QtGui.QIcon(pm)

# ========================= Zoomable View =========================
class ZoomableView(QtWidgets.QGraphicsView):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.MinimalViewportUpdate)
        self.zoom_clamp = (0.3, 3.0)
        self._zoom = 1.0

    def wheelEvent(self, e: QtGui.QWheelEvent):
        if e.modifiers() & QtCore.Qt.ControlModifier:
            angle = e.angleDelta().y(); factor = 1.0015 ** angle
            self._apply_zoom(factor, e.position())
        else:
            super().wheelEvent(e)

    def _apply_zoom(self, factor, anchor_pos):
        old = self._zoom; new = clamp(old * factor, self.zoom_clamp[0], self.zoom_clamp[1]); factor = new / old; self._zoom = new
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor); self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        view_pos = QtCore.QPointF(anchor_pos); scene_pos = self.mapToScene(view_pos.toPoint())
        self.scale(factor, factor); view_pos_after = self.mapFromScene(scene_pos); delta = view_pos - view_pos_after
        self.translate(delta.x(), delta.y())

    def resetZoom(self):
        self.resetTransform(); self._zoom = 1.0

# ========================= Canva-style Chip =========================
class ChipItem(QtWidgets.QGraphicsItemGroup):
    def __init__(self, text: str, bg: QtGui.QColor, fg: QtGui.QColor = INK, parent=None):
        super().__init__(parent)
        self.rect = QtWidgets.QGraphicsRectItem()
        self.text_item = QtWidgets.QGraphicsTextItem(text)
        font = QtGui.QFont("Inter", 11, QtGui.QFont.DemiBold)
        self.text_item.setFont(font); self.text_item.setDefaultTextColor(fg)
        self.rect.setBrush(QtGui.QBrush(bg))
        pen = QtGui.QPen(QtGui.QColor(0,0,0,25)); pen.setWidthF(1.2); self.rect.setPen(pen)
        self.addToGroup(self.rect); self.addToGroup(self.text_item)
        self.setZValue(12)
        self._bg_color = bg; self._fg_color = fg; self._pad_h = 12; self._pad_v = 6
        self.update_geometry()

    def set_text(self, text: str):
        self.text_item.setPlainText(text); self.update_geometry()

    def set_bg(self, color: QtGui.QColor):
        self._bg_color = color; self.rect.setBrush(QtGui.QBrush(color))

    def update_geometry(self):
        br = self.text_item.boundingRect(); w = br.width() + self._pad_h*2; h = br.height() + self._pad_v*2
        self.rect.setRect(0, 0, w, h); self.text_item.setPos(self._pad_h, self._pad_v)
        path = QtGui.QPainterPath(); path.addRoundedRect(QtCore.QRectF(0,0,w,h), 18, 18); self._shape = path

    def shape(self): return self._shape

# ========================= Track Geometry =========================
class Segment(QtCore.QObject):
    def __init__(self, scene, x1, y1, x2, y2, name, base_fill=TRACK_FREE_FILL, width=24, z=0, ctrl=None):
        super().__init__()
        self.scene = scene; self.name = name; self.base_fill = base_fill
        self.occupied_by = None; self.width = width; self.ctrl = ctrl

        self.path_item = QtWidgets.QGraphicsPathItem()
        # ✅ ขอบรางใช้ปลายเหลี่ยม/มุมมีเทอร์
        pen = QtGui.QPen(GRAY_LINE, 2, QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin)
        pen.setCosmetic(True)
        self.path_item.setPen(pen); self.path_item.setZValue(z); self.scene.addItem(self.path_item)

        self.label_item = self.scene.addText(name, QtGui.QFont("Inter", 10, QtGui.QFont.DemiBold))
        self.label_item.setDefaultTextColor(INK_SOFT); self.label_item.setZValue(z+0.1)

        self.center_path = QtGui.QPainterPath(); self.ux, self.uy = 1.0, 0.0
        self.set_geom(x1,y1,x2,y2, ctrl=self.ctrl)

    def set_geom(self, x1, y1, x2, y2, ctrl=None):
        self.x1, self.y1 = float(round(x1)), float(round(y1))
        self.x2, self.y2 = float(round(x2)), float(round(y2))
        self.ctrl = ctrl

        self.center_path = QtGui.QPainterPath(QtCore.QPointF(self.x1, self.y1))
        if self.ctrl is None:
            self.center_path.lineTo(QtCore.QPointF(self.x2, self.y2))
        else:
            cx, cy = self.ctrl
            self.center_path.quadTo(QtCore.QPointF(float(round(cx)), float(round(cy))), QtCore.QPointF(self.x2, self.y2))

        try:
            self.length = max(1.0, self.center_path.length()); ang0 = self.center_path.angleAtPercent(0.0)
        except Exception:
            dx, dy = self.x2 - self.x1, self.y2 - self.y1
            self.length = math.hypot(dx, dy) if (dx or dy) else 1.0
            ang0 = math.degrees(math.atan2(dy, dx))
        rad = math.radians(ang0); self.ux, self.uy = math.cos(rad), -math.sin(rad)

        self._rebuild(); self._update_brush()

    # โค้ดเดิม
    # โค้ดใหม่
    def _stroked_outline(self):
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(self.width)
        stroker.setCapStyle(QtCore.Qt.SquareCap)
        stroker.setMiterLimit(8.0)

        # ✅ NEW: เช็คชื่อของ Segment
        if self.name in ["B1", "B2"]:
            # ถ้าเป็น B1 หรือ B2 ให้ใช้มุมมน (RoundJoin)
            stroker.setJoinStyle(QtCore.Qt.RoundJoin)
        else:
            # Segment อื่นๆ ใช้มุมเหลี่ยมเหมือนเดิม (MiterJoin)
            stroker.setJoinStyle(QtCore.Qt.MiterJoin)
            
        return stroker.createStroke(self.center_path)

    def _rebuild(self):
        outline = self._stroked_outline(); self.path_item.setPath(outline)
        try:
            pt = self.center_path.pointAtPercent(0.5)
        except Exception:
            px, py = (self.x1 + self.x2)/2, (self.y1 + self.y2)/2; pt = QtCore.QPointF(px, py)
        br = self.label_item.boundingRect(); self.label_item.setPos(pt.x() - br.width()/2, pt.y() - br.height()/2)

    def _update_brush(self):
        # 🎨 DESIGN CHANGE: "Hollow Rails" Style
        # ดีไซน์ใหม่: รางจะเป็นแบบโปร่ง
        
        pen = self.path_item.pen() # ดึง Pen ปัจจุบันมาแก้ไข
        
        if self.occupied_by:
            # --- เมื่อมีรถไฟ ---
            # 1. ตัวราง (Pen) เปลี่ยนเป็นสีเข้มตามสีรถไฟ
            pen.setColor(self.occupied_by.body_color.darker(130))
            pen.setWidthF(2.2) # หนาขึ้นเล็กน้อย
            
            # 2. พื้นหลัง (Brush) เรืองแสงสีอ่อนๆ แบบโปร่งใส
            track_color = QtGui.QColor(self.occupied_by.body_color)
            track_color.setAlpha(60) # ตั้งค่าความโปร่งใส (ประมาณ 24%)
            self.path_item.setBrush(QtGui.QBrush(track_color))
        else:
            # --- เมื่อรางว่าง ---
            # 1. ตัวราง (Pen) กลับไปเป็นสีเทามาตรฐาน
            pen.setColor(GRAY_LINE)
            pen.setWidthF(2)
            
            # 2. พื้นหลัง (Brush) ใส (ไม่เติมสี)
            self.path_item.setBrush(QtCore.Qt.NoBrush)
            
        pen.setCosmetic(True) # ทำให้ความหนาเส้นคงที่ ไม่ว่าจะซูมเท่าไหร่
        self.path_item.setPen(pen)

    def set_occupied(self, train_obj=None):
        self.occupied_by = train_obj; self._update_brush()

    def is_free(self): return self.occupied_by is None

    def point_at(self, t):
        t = clamp(t, 0.0, 1.0)
        try:
            p = self.center_path.pointAtPercent(t); x, y = p.x(), p.y()
        except Exception:
            x = lerp(self.x1, self.x2, t); y = lerp(self.y1, self.y2, t)
        return float(round(x)), float(round(y))

# Simple block controller
class BlockController:
    def __init__(self, segs): self.segs = segs
    def enter(self, seg, train): 
        if seg.is_free(): seg.set_occupied(train); return True
        return False
    def leave(self, seg, train):
        if seg.occupied_by == train: seg.set_occupied(None)

# ========================= Train =========================
class Train(QtCore.QObject):
    statusChanged = QtCore.Signal()
    def __init__(self, scene, name, path, block, stop_segment_name, color, speed=140, stage_offset_px=0):
        super().__init__()
        self.scene, self.name, self.path, self.block = scene, name, path, block
        self.idx, self.t, self.mode = 0, 0.0, "ARRIVAL"
        self.speed = speed
        self.stop_seg_name, self.stop_at_t = stop_segment_name, 0.5
        self.body_color = color

        s0 = self.path[0]
        # ✅ ตัวรถไฟเป็น rounded-rect ผ่าน QGraphicsPathItem (ทุกเครื่องรันได้)
        rect = QtCore.QRectF(-12, -7, 24, 14)
        pth = QtGui.QPainterPath(); pth.addRoundedRect(rect, 6, 6)
        self.body = QtWidgets.QGraphicsPathItem(pth)
        self.body.setBrush(QtGui.QBrush(self.body_color))
        self.body.setPen(QtGui.QPen(self.body_color.darker(130), 1.2))
        self.body.setZValue(5)
        self.scene.addItem(self.body)

        pre_x = s0.x1 - s0.ux*(20 + stage_offset_px)
        pre_y = s0.y1 - s0.uy*(20 + stage_offset_px)
        self.body.setPos(round(pre_x), round(pre_y))

        self.running = False; self.finished = False; self._last = time.time()
        self.start_arrival()

    def cleanup(self):
        if self.body is not None:
            self.scene.removeItem(self.body); self.body = None

    def start_arrival(self):
        first = self.path[0]
        if self.block.enter(first, self):
            self.running = True; self._last = time.time()
            x,y = first.point_at(0.0); self.body.setPos(x,y)
        else:
            self.running = False

    def start_departure(self):
        if self.mode != "AT_STATION" or self.finished: return
        nxt_i = self.idx + 1
        if nxt_i >= len(self.path):
            self.finished = True; self.cleanup(); self.statusChanged.emit(); return
        nxt = self.path[nxt_i]
        if self.block.enter(nxt, self):
            self.mode = "DEPARTURE"; self.running = True; self._last = time.time()
            x,y = nxt.point_at(0.0); self.body.setPos(x,y)
            self.statusChanged.emit()

    def _try_enter_waiting(self):
        if self.mode == "ARRIVAL":
            cur = self.path[self.idx]
            if cur.is_free() and self.block.enter(cur, self):
                self.running = True; self._last = time.time()
                x,y = cur.point_at(self.t); self.body.setPos(x,y)
        elif self.mode == "DEPARTURE":
            cur = self.path[self.idx]
            if cur.occupied_by == self:
                self.running = True; self._last = time.time()
                x,y = cur.point_at(self.t); self.body.setPos(x,y)
            elif cur.is_free() and self.block.enter(cur, self):
                self.running = True; self._last = time.time()
                x,y = cur.point_at(self.t); self.body.setPos(x,y)

    def tick(self, dt):
        if not self.running and not self.finished: self._try_enter_waiting()
        if not self.running or self.finished: return

        EPS = 1e-6; dist = self.speed * dt
        while dist > EPS and self.idx < len(self.path):
            seg = self.path[self.idx]
            remain = (1.0 - self.t) * seg.length
            step = min(remain, dist)
            if seg.length > 0: self.t = max(0.0, min(1.0, self.t + step / seg.length))
            dist -= step

            x, y = seg.point_at(self.t); self.body.setPos(x, y)

            # หยุดกลางสถานี
            if self.mode == "ARRIVAL" and seg.name == self.stop_seg_name and self.t + EPS >= self.stop_at_t:
                x, y = seg.point_at(self.stop_at_t); self.body.setPos(x, y)
                self.running = False; self.mode = "AT_STATION"; self.statusChanged.emit(); break

            # จบ segment
            if self.t + EPS >= 1.0:
                endx, endy = seg.point_at(1.0); self.body.setPos(endx, endy)
                self.block.leave(seg, self)
                self.idx += 1; self.t = 0.0

                if self.idx >= len(self.path):
                    self.running = False; self.finished = True; self.cleanup(); self.statusChanged.emit(); break

                self.running = False; self.statusChanged.emit(); break

        if self.idx >= len(self.path) and not self.running and not self.finished:
            self.finished = True; self.cleanup(); self.statusChanged.emit()

    def current_block_name(self):
        if self.finished: return "DONE"
        if self.idx >= len(self.path): return "END"
        return self.path[self.idx].name

    def position(self):
        c = self.body.sceneBoundingRect().center()
        return int(c.x()), int(c.y())

# ============================== Main Window ==============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Railway Schematic — Pastel Blocks + Connectors")
        self.resize(1280, 720)

        # central
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central); root.setContentsMargins(14,14,14,14); root.setSpacing(12)

        # header
        header = QtWidgets.QFrame(); header.setObjectName("Header")
        h = QtWidgets.QHBoxLayout(header); h.setContentsMargins(18,14,18,14); h.setSpacing(12)
        logo = QtWidgets.QLabel(); logo.setPixmap(make_icon("route_loop", 22).pixmap(22,22))
        title = QtWidgets.QLabel("Klong 6 — Control Panel"); title.setObjectName("Title")
        h.addWidget(logo); h.addWidget(title); h.addStretch(1)
        self.btn_reset = self._btn("Reset", role="danger"); self.btn_reset.setIcon(make_icon("reset"))
        h.addWidget(self.btn_reset); root.addWidget(header)

        # control card
        ctrl_card = QtWidgets.QFrame(); ctrl_card.setObjectName("Card")
        ctrl = QtWidgets.QHBoxLayout(ctrl_card); ctrl.setContentsMargins(16,14,16,14); ctrl.setSpacing(12)
        root.addWidget(ctrl_card)

        self.btn_main = self._btn("MAIN → S1", role="pill"); self.btn_main.setIcon(make_icon("route_main"))
        self.btn_loop = self._btn("LOOP → S2", role="pill"); self.btn_loop.setIcon(make_icon("route_loop"))
        self.btn_main.setCheckable(True); self.btn_loop.setCheckable(True)
        grp = QtWidgets.QButtonGroup(self); grp.setExclusive(True); grp.addButton(self.btn_main); grp.addButton(self.btn_loop)
        self.btn_main.setChecked(True)

        self.btn_call = self._btn(" เรียกรถ", role="primary");   self.btn_call.setIcon(make_icon("call"))
        self.btn_release = self._btn(" ปล่อยรถ", role="secondary"); self.btn_release.setIcon(make_icon("release"))
        self.btn_pause = self._btn(" Pause", role="ghost");       self.btn_pause.setIcon(make_icon("pause"))

        speed_wrap = QtWidgets.QHBoxLayout(); speed_wrap.setSpacing(8)
        speed_icon = QtWidgets.QLabel(); speed_icon.setPixmap(make_icon("speed", 18).pixmap(18,18))
        self.speed_label = QtWidgets.QLabel("Speed 100%"); self.speed_label.setObjectName("DimText")
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.speed_slider.setRange(50, 200); self.speed_slider.setValue(100)
        self.speed_slider.setFixedWidth(180); speed_wrap.addWidget(speed_icon); speed_wrap.addWidget(self.speed_label); speed_wrap.addWidget(self.speed_slider)

        ctrl.addWidget(self.btn_main); ctrl.addWidget(self.btn_loop)
        ctrl.addSpacing(6); ctrl.addWidget(self._divider()); ctrl.addSpacing(6)
        ctrl.addWidget(self.btn_call); ctrl.addWidget(self.btn_release); ctrl.addWidget(self.btn_pause)
        ctrl.addStretch(1); ctrl.addLayout(speed_wrap)

        # graphics card
        gfx_card = QtWidgets.QFrame(); gfx_card.setObjectName("Card")
        gfx = QtWidgets.QVBoxLayout(gfx_card); gfx.setContentsMargins(12,12,12,12); gfx.setSpacing(10)
        root.addWidget(gfx_card, 1)

        # scene/view
        self.scene = QtWidgets.QGraphicsScene(self)
        grad = QtGui.QLinearGradient(0,0,0,460); grad.setColorAt(0.0, QtGui.QColor(BG_GRAD_1)); grad.setColorAt(1.0, QtGui.QColor(BG_GRAD_2))
        self.scene.setBackgroundBrush(QtGui.QBrush(grad))
        self.view = ZoomableView(self.scene); self.view.setSceneRect(0,0,1240,500)
        self.view.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor); self.view.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        gfx.addWidget(self.view, 1)

        # clock
        self._add_clock()

        # status line
        self.status_label = QtWidgets.QLabel("MAIN=FREE | LOOP=FREE | Trains=0"); self.status_label.setObjectName("StatusLine")
        gfx.addWidget(self.status_label)

        # build tracks/signals
        self.build_layout()

        # state
        self.route = "MAIN"; self.trains = []; self.sim_speed = 1.0; self.paused = False; self.next_color_idx = 0

        # chips
        self._add_status_chips()

        # events
        self.btn_main.clicked.connect(lambda: (self.set_main(), self._update_status()))
        self.btn_loop.clicked.connect(lambda: (self.set_loop(), self._update_status()))
        self.btn_call.clicked.connect(self.call_train)
        self.btn_release.clicked.connect(self.release_train)
        self.btn_reset.clicked.connect(self.reset_all)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)

        # shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("M"), self, activated=lambda: (self.set_main(), self._update_status()))
        QtGui.QShortcut(QtGui.QKeySequence("L"), self, activated=lambda: (self.set_loop(), self._update_status()))
        QtGui.QShortcut(QtGui.QKeySequence("C"), self, activated=self.call_train)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self.release_train)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, activated=self.toggle_pause)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+0"), self, activated=self.view.resetZoom)

        # timer
        self._prev = time.time()
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.on_tick); self.timer.start(30)

        # style
        for b in [self.btn_main, self.btn_loop, self.btn_call, self.btn_release, self.btn_pause, self.btn_reset, header, ctrl_card, gfx_card]:
            self.add_shadow(b, radius=24, dx=0, dy=6, alpha=90)
        self.apply_styles()

    # ---------- Style ----------
    def apply_styles(self):
        self.setStyleSheet(f"""
        QWidget {{ font-family: 'Inter', 'Segoe UI', Arial; color:{INK.name()}; font-size: 13px; }}
        QMainWindow {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {BG_GRAD_1}, stop:1 {BG_GRAD_2}); }}

        #Header {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 20px;
        }}
        QLabel#Title {{ font-size: 18px; font-weight: 700; color:{INK.name()}; }}

        #Card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 20px;
        }}

        QPushButton {{
            padding: 9px 16px; border-radius: 999px; border: 1px solid {BORDER};
            background: #F8FAFC; color:{INK.name()};
        }}
        QPushButton:hover {{ background: #F1F5F9; }}
        QPushButton:pressed {{ background: #E2E8F0; }}

        QPushButton#Pill:checked {{ background: #7CBAFF; color:#083344; border: 1px solid rgba(8,51,68,0.15); }}
        QPushButton#Primary {{ background: #7CBAFF; color:#083344; border: 1px solid rgba(8,51,68,0.2); }}
        QPushButton#Primary:hover {{ background: #94CCFF; }}
        QPushButton#Secondary {{ background: #B9FBC0; color:#064E3B; border: 1px solid rgba(6,78,59,0.2); }}
        QPushButton#Secondary:hover {{ background: #D0FDE1; }}
        QPushButton#Ghost {{ background: transparent; }}
        QPushButton#Danger {{ background: #FCA5A5; color:#7A1D1D; border: 1px solid rgba(122,29,29,0.2); }}
        QPushButton#Danger:hover {{ background: #FECACA; }}

        QLabel#StatusLine {{ color: {INK_SOFT.name()}; padding: 4px 6px; font-family: 'JetBrains Mono','Cascadia Mono', Consolas, monospace; }}
        QLabel#DimText {{ color: {INK_SOFT.name()}; }}

        QSlider::groove:horizontal {{ height: 6px; background: #E2E8F0; border-radius: 3px; }}
        QSlider::handle:horizontal {{ width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; background: #7CBAFF; border: 1px solid {BORDER}; }}
        """)

    def add_shadow(self, w, radius=22, dx=0, dy=8, alpha=120):
        eff = QtWidgets.QGraphicsDropShadowEffect(self)
        eff.setBlurRadius(radius); eff.setOffset(dx, dy); eff.setColor(QtGui.QColor(15,23,42,alpha))
        w.setGraphicsEffect(eff)

    def _btn(self, text, role="pill"):
        btn = QtWidgets.QPushButton(text); btn.setCursor(QtCore.Qt.PointingHandCursor)
        if role == "pill": btn.setObjectName("Pill")
        elif role == "primary": btn.setObjectName("Primary")
        elif role == "secondary": btn.setObjectName("Secondary")
        elif role == "ghost": btn.setObjectName("Ghost")
        elif role == "danger": btn.setObjectName("Danger")
        else: btn.setObjectName("")
        return btn

    def _divider(self):
        line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.VLine)
        line.setStyleSheet("color: rgba(15,23,42,0.08);"); return line

    # ---------- Clock (scene top-right) ----------
    def _add_clock(self):
        font = QtGui.QFont("Inter", 16, QtGui.QFont.Bold)
        self.clock_item = self.scene.addText("", font)
        self.clock_item.setDefaultTextColor(INK); self.clock_item.setZValue(10)
        self._last_clock_sec = -1; self._update_clock(force=True)

    def _update_clock(self, force=False):
        dt = datetime.now(TZ_BKK) if TZ_BKK is not None else datetime.fromtimestamp(time.time())
        be_year = dt.year + 543; date_txt = f"{dt.day}/{dt.month}/{be_year}"; time_txt = dt.strftime("%H:%M:%S")
        sec = dt.second
        if (sec != self._last_clock_sec) or force:
            self._last_clock_sec = sec; self.clock_item.setPlainText(f"{time_txt}  {date_txt}")
        br = self.clock_item.boundingRect(); sr = self.view.sceneRect(); margin = 8
        self.clock_item.setPos(sr.right() - br.width() - margin, sr.top() + margin)

    # ---------- Layout ----------
    def build_layout(self):
        self.segs = {}
        TRACK_W = 24
        y_main  = 320
        y_top   = 220
        x_s1_l, x_s1_r = 480, 700

        # main line blocks
        self._add_line("A1",  80, y_main, 200, y_main, TRACK_FREE_FILL, TRACK_W, z=0)
        self._add_line("A2", 220, y_main, 340, y_main, TRACK_FREE_FILL, TRACK_W, z=0)
        self._add_line("A3", 360, y_main, x_s1_l, y_main, TRACK_FREE_FILL, TRACK_W, z=0)
        self._add_line("S1", x_s1_l, y_main, x_s1_r, y_main, TRACK_FREE_FILL,  TRACK_W, z=1)

        total_len_A4_A6 = 1120 - 700
        seg_len = total_len_A4_A6 / 3.0
        self._add_line("A4", x_s1_r, y_main, 700 + seg_len, y_main, TRACK_FREE_FILL, TRACK_W, z=0)
        self._add_line("A5", 700 + seg_len, y_main, 700 + 2*seg_len, y_main, TRACK_FREE_FILL, TRACK_W, z=0)
        self._add_line("A6", 700 + 2*seg_len, y_main, 1120,       y_main, TRACK_FREE_FILL, TRACK_W, z=0)

        # loop path
        s1_cx = (x_s1_l + x_s1_r)/2
        s2_len = 120
        x_s2_l = s1_cx - s2_len/2
        x_s2_r = s1_cx + s2_len/2
        self._add_line("S2", x_s2_l, y_top, x_s2_r, y_top, TRACK_FREE_FILL, TRACK_W, z=1)

        offset_y = -40
        cx_b1 = (x_s1_l + x_s2_l)/2; cy_b1 = (y_main + y_top)/2 + offset_y
        self._add_curve("B1", x_s1_l, y_main, cx_b1, cy_b1, x_s2_l, y_top, TRACK_FREE_FILL, TRACK_W, z=0.5)
        cx_b2 = (x_s2_r + x_s1_r)/2; cy_b2 = (y_top + y_main)/2 + offset_y
        self._add_curve("B2", x_s2_r, y_top, cx_b2, cy_b2, x_s1_r, y_main, TRACK_FREE_FILL, TRACK_W, z=0.5)

        self._ensure_contiguous(["A1","A2","A3","S1","A4","A5","A6"])
        self._draw_connectors(radius=9)   # ✅ วงกลมคั่นทุกจุด
        self._verify_contiguity([
            ["A1","A2","A3","S1","A4","A5","A6"],
            ["A3","B1","S2","B2","A4"]
        ])

        self.block = BlockController(self.segs)

        # Signals
        self.signals = {}
        def add_signal(name, x, y, color):
            sig = self.scene.addEllipse(x, y, 16, 16, QtGui.QPen(QtGui.QColor("#94A3B8"), 2), QtGui.QBrush(QtGui.QColor(color)))
            sig.setZValue(8); self.signals[name] = sig
        add_signal("S1_IN",  x_s1_l - 24, y_main - 30, "#FCA5A5")
        add_signal("S1_OUT", x_s1_r + 8,  y_main - 30, "#FCA5A5")
        add_signal("S2_IN",  x_s2_l - 24, y_top - 30, "#FCA5A5")
        add_signal("S2_OUT", x_s2_r + 8,  y_top - 30, "#FCA5A5")
        add_signal("PANEL_CALL",    20, 20, "#CBD5E1"); txt_call = self.scene.addText("CALL", QtGui.QFont("Inter", 9, QtGui.QFont.DemiBold))
        txt_call.setDefaultTextColor(INK_SOFT); txt_call.setPos(40, 18); txt_call.setZValue(8)
        add_signal("PANEL_RELEASE", 90, 20, "#CBD5E1"); txt_rel  = self.scene.addText("RELEASE", QtGui.QFont("Inter", 9, QtGui.QFont.DemiBold))
        txt_rel.setDefaultTextColor(INK_SOFT); txt_rel.setPos(110, 18); txt_rel.setZValue(8)

        station = self.scene.addText("สถานีคลอง 6 (Klong 6 Station)", QtGui.QFont("Inter", 14, QtGui.QFont.Bold))
        station.setDefaultTextColor(INK); station.setZValue(6)
        br = station.boundingRect(); station.setPos(s1_cx - br.width()/2, y_main - 200)

    def _add_line(self, name, x1,y1,x2,y2, fill, width, z):
        self.segs[name] = Segment(self.scene, x1,y1,x2,y2, name, base_fill=fill, width=width, z=z, ctrl=None)

    def _add_curve(self, name, x1,y1,cx,cy,x2,y2, fill, width, z):
        self.segs[name] = Segment(self.scene, x1,y1,x2,y2, name, base_fill=fill, width=width, z=z, ctrl=(cx,cy))

    def _ensure_contiguous(self, ordered_names):
        for i in range(len(ordered_names)-1):
            a = self.segs[ordered_names[i]]; b = self.segs[ordered_names[i+1]]
            b.set_geom(a.x2, a.y2, b.x2, b.y2, ctrl=b.ctrl)

    def _draw_connectors(self, radius=9):
        # ลบของเก่า หากมี
        if hasattr(self, "_connector_items"):
            for it in self._connector_items:
                try: self.scene.removeItem(it)
                except: pass
        self._connector_items = []

        seen=set()
        def key(x,y): return (round(x), round(y))
        pts=[]
        for s in self.segs.values(): pts += [(s.x1,s.y1),(s.x2,s.y2)]
        for (x,y) in pts:
            k=key(x,y)
            if k in seen: continue
            seen.add(k)
            circ = self.scene.addEllipse(x-radius, y-radius, 2*radius, 2*radius,
                                         QtGui.QPen(QtGui.QColor("#94A3B8"), 2),
                                         QtGui.QBrush(QtGui.QColor("#FFFFFF")))
            circ.setZValue(999)  # อยู่บนรางเสมอ
            self._connector_items.append(circ)

    def _verify_contiguity(self, chains):
        bad = []
        for names in chains:
            for i in range(len(names)-1):
                a = self.segs[names[i]]; b = self.segs[names[i+1]]
                if round(a.x2)!=round(b.x1) or round(a.y2)!=round(b.y1):
                    bad.append((a.name, b.name, (a.x2,a.y2), (b.x1,b.y1)))
        if bad:
            print("⚠️ Non-contiguous joints:")
            for a,b,p,q in bad: print(f"  {a} -> {b}: {p} != {q}")
        else:
            print("✅ All joints contiguous")

    # ---------- Panel & Chips ----------
    def set_signal(self, name, color_hex):
        if name in self.signals: self.signals[name].setBrush(QtGui.QBrush(QtGui.QColor(color_hex)))

    def _add_status_chips(self):
        self.chip_main = ChipItem("MAIN: FREE", GREEN_OK, INK)
        self.chip_loop = ChipItem("LOOP: FREE", GREEN_OK, INK)
        self.scene.addItem(self.chip_main); self.scene.addItem(self.chip_loop)
        self._place_status_chips(); self._update_status_chips()

    def _place_status_chips(self):
        sr = self.view.sceneRect()
        self.chip_main.setPos(sr.left() + 16, sr.top() + 56)
        self.chip_loop.setPos(sr.left() + 16, sr.top() + 96)

    def _update_status_chips(self):
        busy_main = self._route_busy("S1"); busy_loop = self._route_busy("S2")
        self.chip_main.set_text(f"MAIN: {'BUSY' if busy_main else 'FREE'}")
        self.chip_loop.set_text(f"LOOP: {'BUSY' if busy_loop else 'FREE'}")
        self.chip_main.set_bg(RED_BUSY if busy_main else GREEN_OK)
        self.chip_loop.set_bg(RED_BUSY if busy_loop else GREEN_OK)

    # ---------- Routing ----------
    def _path(self):
        if self.route == "MAIN":
            order = ["A1","A2","A3","S1","A4","A5","A6"]; stop = "S1"
        else:
            order = ["A1","A2","A3","B1","S2","B2","A4","A5","A6"]; stop = "S2"
        return [self.segs[n] for n in order], stop

    def set_main(self): self.route="MAIN"; self.btn_main.setChecked(True)
    def set_loop(self): self.route="LOOP"; self.btn_loop.setChecked(True)

    # ---------- Ops ----------
    def _route_busy(self, stop_name):
        return any((not t.finished) and (t.stop_seg_name == stop_name) for t in self.trains)

    def call_train(self):
        self.set_signal("PANEL_CALL", "#22C55E"); QtCore.QTimer.singleShot(800, lambda: self.set_signal("PANEL_CALL", "#CBD5E1"))
        target_stop = "S1" if self.route == "MAIN" else "S2"
        if self._route_busy(target_stop): self._update_status(); return
        target_signal_in = "S1_IN" if self.route == "MAIN" else "S2_IN"
        self.set_signal(target_signal_in, "#22C55E"); QtCore.QTimer.singleShot(800, lambda: self.set_signal(target_signal_in, "#FCA5A5"))
        path, stop = self._path()
        color = TRAIN_COLORS[self.next_color_idx % len(TRAIN_COLORS)]; self.next_color_idx += 1
        t = Train(self.scene, f"T{len(self.trains)+1}", path, self.block, stop, color=color, speed=140, stage_offset_px=0)
        t.statusChanged.connect(self._update_status); self.trains.append(t); self._update_status()

    def release_train(self):
        self.set_signal("PANEL_RELEASE", "#7CBAFF"); QtCore.QTimer.singleShot(800, lambda: self.set_signal("PANEL_RELEASE", "#CBD5E1"))
        at_station = [t for t in self.trains if (t.mode=="AT_STATION" and not t.finished)]
        if not at_station: self._update_status(); return
        t = at_station[0]
        target_signal_out = "S1_OUT" if t.stop_seg_name == "S1" else "S2_OUT"
        self.set_signal(target_signal_out, "#22C55E"); QtCore.QTimer.singleShot(800, lambda: self.set_signal(target_signal_out, "#FCA5A5"))
        t.start_departure(); self._update_status()

    def reset_all(self):
        for t in self.trains: t.cleanup()
        self.trains.clear()
        for s in self.segs.values(): s.set_occupied(None)
        for k in ["S1_IN","S1_OUT","S2_IN","S2_OUT"]: self.set_signal(k, "#FCA5A5")
        self.next_color_idx = 0
        self._update_status()

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText(" Resume" if self.paused else " Pause")
        self.btn_pause.setIcon(make_icon("resume" if self.paused else "pause"))

    def on_speed_changed(self, value):
        self.sim_speed = value / 100.0; self.speed_label.setText(f"Speed {value}%")

    # ---------- Tick ----------
    def on_tick(self):
        now = time.time(); dt = now - getattr(self, "_prev", now); self._prev = now
        if not self.paused:
            dt *= self.sim_speed
            for t in list(self.trains): t.tick(dt)
            self.trains = [t for t in self.trains if not t.finished]
        if int(now*10)%3==0: self._update_status()
        self._update_clock(); self._update_status_chips()

    def _update_status(self):
        busy_main = self._route_busy("S1"); busy_loop = self._route_busy("S2")
        active = len([t for t in self.trains if not t.finished])
        self.status_label.setText(f"MAIN={'BUSY' if busy_main else 'FREE'} | LOOP={'BUSY' if busy_loop else 'FREE'} | Trains={active}")

# --------- run ---------
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv); app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(BG_GRAD_1))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(SURFACE))
    pal.setColor(QtGui.QPalette.Text, INK)
    pal.setColor(QtGui.QPalette.ButtonText, INK)
    app.setPalette(pal)
    w = MainWindow(); w.show()
    sys.exit(app.exec())
