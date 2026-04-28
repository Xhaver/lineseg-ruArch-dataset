#!/usr/bin/env python3
"""
prepare_yolo_dataset.py

Генерирует YOLO-датасет из *_full_annotated.dxf.

  Классы YOLO:
    0 — text      (TEXT, MTEXT)
    1 — dimension (DIMENSION, MULTILEADER)

  Для каждого чертежа:
    1. Растеризует геометрию (LINE, LWPOLYLINE, ARC, CIRCLE, ELLIPSE) → PNG 8-bit
    2. Вычисляет bbox аннотаций → YOLO .txt
    3. Нарезает сеткой (по умолчанию 1280 px, overlap 64 px)
    4. Генерирует зашумлённые варианты (gaussian_noise, salt_pepper, gaussian_blur)
       Rotation исключён по умолчанию (требует трансформации bbox).
    5. Сохраняет структуру YOLO + data.yaml + yolo_manifest.json

  Структура вывода:
    yolo_dataset/
        images/train/   images/val/
        labels/train/   labels/val/
        data.yaml
    yolo_manifest.json

Использование:
    python scripts/prepare_yolo_dataset.py --scale 100 --units m
    python scripts/prepare_yolo_dataset.py --scale 100 --units m --no-noise
    python scripts/prepare_yolo_dataset.py --scale 100 --units m --source sources/1_full_annotated.dxf
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

try:
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.ndimage import gaussian_filter
except ImportError as e:
    print(f"Ошибка импорта: {e}\nУстановите: pip install numpy Pillow scipy", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    from ezdxf.addons.drawing.properties import LayoutProperties
    _HAS_MPL_BACKEND = True
except ImportError:
    _HAS_MPL_BACKEND = False

try:
    import ezdxf
    from ezdxf.document import Drawing
except ImportError:
    print("Ошибка: ezdxf не установлен. pip install ezdxf", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR  = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.parent.resolve()

# ──────────────────────────────────────────────────────────────────────────────
# YOLO-классы
# ──────────────────────────────────────────────────────────────────────────────

CLASS_TEXT      = 0   # TEXT, MTEXT, ATTRIB
CLASS_DIMENSION = 1   # DIMENSION, MULTILEADER, LEADER
CLASS_REGION    = 2   # Таблица / штамп
CLASS_NAMES     = ["text", "dimension", "annotation_region"]

# Слои-маркеры для ручной разметки областей.
#
#   YOLO_TABLE  — таблица из линий с текстом (экспликации, ведомости)
#   YOLO_STAMP  — штамп чертежа (основная надпись)
#
# YOLO_FRAME временно отключён (удалён из REGION_LAYERS).
# Алгоритм _resolve_frame_regions реализован и готов к включению:
# если поставить "YOLO_FRAME" обратно в REGION_LAYERS — заработает
# автоматическое вычисление 4 полос из двух полилиний (внешней + внутренней).
# Текущее решение: рамку удалять из DXF-источников вручную (см. CHECKLIST.md).
REGION_LAYERS = {"YOLO_TABLE", "YOLO_STAMP"}
# REGION_LAYERS = {"YOLO_TABLE", "YOLO_STAMP", "YOLO_FRAME"}  # раскомментировать для детекции рамки

# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация шума (без rotation)
# ──────────────────────────────────────────────────────────────────────────────

NOISE_CONFIGS: dict[str, list[tuple[str, dict]]] = {
    "gaussian_noise": [
        ("L1", {"sigma": 0.05}),
        ("L2", {"sigma": 0.10}),
        ("L3", {"sigma": 0.15}),
        ("L4", {"sigma": 0.25}),
        ("L5", {"sigma": 0.40}),
    ],
    "salt_pepper": [
        ("L1", {"density": 0.0001}),
        ("L2", {"density": 0.001}),
        ("L3", {"density": 0.010}),
        ("L4", {"density": 0.040}),
        ("L5", {"density": 0.100}),
    ],
    "gaussian_blur": [
        ("L1", {"sigma": 0.5}),
        ("L2", {"sigma": 1.0}),
        ("L3", {"sigma": 1.5}),
        ("L4", {"sigma": 2.0}),
        ("L5", {"sigma": 3.0}),
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# Единицы (из make_crops.py)
# ──────────────────────────────────────────────────────────────────────────────

_INSUNITS_TO_MM: dict[int, float] = {
    0: 1.0, 1: 25.4, 2: 304.8, 3: 1_609_344.0,
    4: 1.0, 5: 10.0, 6: 1_000.0, 7: 1_000_000.0,
    8: 2.54e-5, 9: 0.0254, 10: 914.4,
}
_UNITS_ALIASES: dict[str, float] = {
    "mm": 1.0, "cm": 10.0, "m": 1_000.0,
    "km": 1_000_000.0, "inch": 25.4, "in": 25.4, "ft": 304.8,
}


def parse_units_arg(value: str) -> float:
    lower = value.strip().lower()
    if lower in _UNITS_ALIASES:
        return _UNITS_ALIASES[lower]
    try:
        v = float(value)
        if v <= 0:
            raise ValueError
        return v
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Неверное значение --units '{value}'. "
            f"Ожидается: {', '.join(_UNITS_ALIASES)} или число (мм/ед.)."
        )


def resolve_units(doc: Drawing, override_mm: Optional[float]) -> tuple[float, str]:
    if override_mm is not None:
        return override_mm, f"задано вручную ({override_mm} мм/ед.)"
    insunits = int(doc.header.get("$INSUNITS", 0))
    scale = _INSUNITS_TO_MM.get(insunits, 1.0)
    return scale, f"$INSUNITS={insunits} -> {scale} мм/ед."


# ──────────────────────────────────────────────────────────────────────────────
# Геометрия: вспомогательные функции для аппроксимации кривых
# ──────────────────────────────────────────────────────────────────────────────

def _arc_polyline(
    cx: float, cy: float, r: float,
    start_rad: float, end_rad: float,
    n: int,
) -> list[tuple[float, float]]:
    return [
        (cx + r * math.cos(start_rad + (end_rad - start_rad) * i / n),
         cy + r * math.sin(start_rad + (end_rad - start_rad) * i / n))
        for i in range(n + 1)
    ]


def _n_segments(radius_px: float, arc_angle: float, delta_px: float = 0.5) -> int:
    if radius_px < 1e-9:
        return 4
    ratio = max(-1.0, min(1.0, 1.0 - delta_px / radius_px))
    half_step = math.acos(ratio)
    return max(4, math.ceil(arc_angle / (2.0 * half_step)))


def _render_with_matplotlib(
    doc,
    msp,
    x_min: float, y_min: float,
    x_max: float, y_max: float,
    img_w: int, img_h: int,
    dpi: float,
) -> np.ndarray:
    """
    Растеризует DXF через ezdxf + matplotlib backend.

    Возвращает uint8 grayscale массив (0=чёрный, 255=белый) размером img_w×img_h.

    ezdxf Frontend использует собственную нормализованную систему координат,
    отличную от DXF-единиц. Поэтому мы:
      1. Рендерим в высоком разрешении без лимитов осей (auto-scale).
      2. Сохраняем через BytesIO с bbox_inches='tight' (убирает паддинг).
      3. Читаем обратно и масштабируем до img_w×img_h с LANCZOS.
    Это гарантирует заполнение всего изображения при любой внутренней
    нормализации ezdxf.
    """
    from io import BytesIO

    # Скрываем вспомогательные слои разметки
    hidden_layers = []
    for lname in list(REGION_LAYERS) + ["YOLO_FRAME"]:
        try:
            layer = doc.layers.get(lname)
            if layer and not layer.is_off():
                layer.off()
                hidden_layers.append(lname)
        except Exception:
            pass

    # Рендерим в 3× больше (LANCZOS downscale даёт сглаживание)
    render_scale = 3
    fig = plt.figure(
        figsize=(img_w * render_scale / dpi, img_h * render_scale / dpi),
        dpi=dpi,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ctx      = RenderContext(doc)
    lp       = LayoutProperties.from_layout(msp)
    lp.set_colors(bg="#ffffff")
    backend  = MatplotlibBackend(ax)
    frontend = Frontend(ctx, backend)
    frontend.draw_layout(msp, layout_properties=lp, finalize=True)

    # Сохранение → tight bbox (убирает паддинг вокруг контента)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", facecolor="white", pad_inches=0)
    plt.close(fig)

    # Восстанавливаем скрытые слои
    for lname in hidden_layers:
        try:
            doc.layers.get(lname).on()
        except Exception:
            pass

    # Читаем обратно и масштабируем до точного img_w×img_h
    buf.seek(0)
    pil_img = Image.open(buf).convert("L")
    if pil_img.size != (img_w, img_h):
        pil_img = pil_img.resize((img_w, img_h), Image.LANCZOS)

    return np.array(pil_img, dtype=np.uint8)


def _draw_geometry(
    draw: ImageDraw.ImageDraw,
    msp,
    doc,
    x_min: float, y_max: float,
    px_per_unit: float,
    line_width: int,
) -> None:
    """Запасной PIL-рендер (используется если matplotlib недоступен)."""

    def to_px(x: float, y: float) -> tuple[int, int]:
        return (round((x - x_min) * px_per_unit),
                round((y_max - y)  * px_per_unit))

    for e in msp:
        t = e.dxftype()

        if t == "LINE":
            p1 = to_px(e.dxf.start.x, e.dxf.start.y)
            p2 = to_px(e.dxf.end.x,   e.dxf.end.y)
            draw.line([p1, p2], fill=0, width=line_width)

        elif t == "LWPOLYLINE":
            raw = list(e.get_points(format="xyb"))
            n   = len(raw)
            if n < 2:
                continue
            pairs = list(zip(range(n), range(1, n)))
            if e.closed:
                pairs.append((n - 1, 0))
            for i, j in pairs:
                x1, y1, bulge = raw[i]
                x2, y2, _     = raw[j]
                if abs(bulge) < 1e-10:
                    draw.line([to_px(x1, y1), to_px(x2, y2)],
                              fill=0, width=line_width)
                else:
                    chord = math.hypot(x2 - x1, y2 - y1)
                    if chord < 1e-12:
                        continue
                    theta = 4.0 * math.atan(abs(bulge))
                    r     = chord / (2.0 * math.sin(theta / 2.0))
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    d  = math.sqrt(max(0.0, r * r - (chord / 2.0) ** 2))
                    dx_, dy_ = (x2 - x1) / chord, (y2 - y1) / chord
                    nx_, ny_ = -dy_, dx_
                    sign = 1.0 if bulge > 0 else -1.0
                    cx_, cy_ = mx + sign * d * nx_, my + sign * d * ny_
                    s_rad = math.atan2(y1 - cy_, x1 - cx_)
                    e_rad = math.atan2(y2 - cy_, x2 - cx_)
                    if bulge > 0:
                        while e_rad < s_rad:
                            e_rad += 2 * math.pi
                    else:
                        while e_rad > s_rad:
                            e_rad -= 2 * math.pi
                    pts = _arc_polyline(cx_, cy_, r, s_rad, e_rad,
                                        _n_segments(r * px_per_unit, abs(e_rad - s_rad)))
                    for k in range(len(pts) - 1):
                        draw.line([to_px(*pts[k]), to_px(*pts[k + 1])],
                                  fill=0, width=line_width)

        elif t == "ARC":
            r  = e.dxf.radius
            cx_, cy_ = e.dxf.center.x, e.dxf.center.y
            s_rad = math.radians(e.dxf.start_angle)
            e_rad = math.radians(e.dxf.end_angle)
            if e_rad <= s_rad:
                e_rad += 2 * math.pi
            pts = _arc_polyline(cx_, cy_, r, s_rad, e_rad,
                                _n_segments(r * px_per_unit, e_rad - s_rad))
            for k in range(len(pts) - 1):
                draw.line([to_px(*pts[k]), to_px(*pts[k + 1])],
                          fill=0, width=line_width)

        elif t == "CIRCLE":
            r  = e.dxf.radius
            cx_, cy_ = e.dxf.center.x, e.dxf.center.y
            pts = _arc_polyline(cx_, cy_, r, 0, 2 * math.pi,
                                _n_segments(r * px_per_unit, 2 * math.pi))
            for k in range(len(pts) - 1):
                draw.line([to_px(*pts[k]), to_px(*pts[k + 1])],
                          fill=0, width=line_width)

        elif t == "ELLIPSE":
            major = e.dxf.major_axis
            ratio = e.dxf.ratio
            cx_, cy_ = e.dxf.center.x, e.dxf.center.y
            a = math.hypot(major.x, major.y)
            b = a * ratio
            angle_rot = math.atan2(major.y, major.x)
            sp = e.dxf.start_param
            ep = e.dxf.end_param
            if ep <= sp:
                ep += 2 * math.pi
            n  = _n_segments(max(a, b) * px_per_unit, ep - sp)
            cos_r, sin_r = math.cos(angle_rot), math.sin(angle_rot)
            pts = []
            for i in range(n + 1):
                t_ = sp + (ep - sp) * i / n
                lx = a * math.cos(t_)
                ly = b * math.sin(t_)
                pts.append((cx_ + lx * cos_r - ly * sin_r,
                            cy_ + lx * sin_r + ly * cos_r))
            for k in range(len(pts) - 1):
                draw.line([to_px(*pts[k]), to_px(*pts[k + 1])],
                          fill=0, width=line_width)


# ──────────────────────────────────────────────────────────────────────────────
# Bounding boxes аннотаций в единицах DXF
# ──────────────────────────────────────────────────────────────────────────────

BBox = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax  (DXF units)


def _text_bbox(e) -> Optional[BBox]:
    """Bbox для TEXT с учётом поворота."""
    try:
        ix, iy = e.dxf.insert.x, e.dxf.insert.y
        h = e.dxf.height
        if h <= 0:
            return None
        text = e.dxf.text.strip()
        w = max(h * 0.6, len(text) * h * 0.6) if text else h
        angle = math.radians(getattr(e.dxf, "rotation", 0.0))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # 4 угла прямоугольника (локально: x∈[0,w], y∈[0,h])
        corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        rotated = [
            (ix + cx * cos_a - cy * sin_a,
             iy + cx * sin_a + cy * cos_a)
            for cx, cy in corners
        ]
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        return min(xs), min(ys), max(xs), max(ys)
    except Exception:
        return None


def _mtext_bbox(e) -> Optional[BBox]:
    """Bbox для MTEXT."""
    try:
        ix, iy = e.dxf.insert.x, e.dxf.insert.y
        h  = e.dxf.char_height
        if h <= 0:
            return None
        try:
            plain = e.plain_text()
        except Exception:
            plain = ""
        n_lines = max(1, plain.count("\n") + 1)
        total_h = h * n_lines * 1.3
        # Оценка ширины: из атрибута width или по длине текста
        w = e.dxf.width if e.dxf.width > 0 else max(h, len(plain) * h * 0.55)
        # attachment_point: 1=top-left, 2=top-center, 3=top-right
        #                   4=mid-left,  5=mid-center,  6=mid-right
        #                   7=bot-left,  8=bot-center,  9=bot-right
        ap = getattr(e.dxf, "attachment_point", 1)
        dx = {1: 0, 2: -w / 2, 3: -w,   4: 0,      5: -w / 2, 6: -w,
              7: 0, 8: -w / 2, 9: -w}.get(ap, 0)
        dy = {1: -total_h, 2: -total_h, 3: -total_h,
              4: -total_h / 2, 5: -total_h / 2, 6: -total_h / 2,
              7: 0,  8: 0,  9: 0}.get(ap, -total_h)
        x0, y0 = ix + dx, iy + dy
        return x0, y0, x0 + w, y0 + total_h
    except Exception:
        return None


def _dim_bbox(e, doc: Drawing) -> Optional[BBox]:
    """
    Bbox для DIMENSION/MULTILEADER.
    Вычисляется по LINE-примитивам и MTEXT в geometry-блоке.
    Fallback — по defpoint-атрибутам сущности.
    """
    xs: list[float] = []
    ys: list[float] = []

    # Из geometry block
    try:
        block_name = e.dxf.geometry
        block = doc.blocks.get(block_name)
        if block:
            for be in block:
                bt = be.dxftype()
                if bt == "LINE":
                    for pt in (be.dxf.start, be.dxf.end):
                        xs.append(pt.x); ys.append(pt.y)
                elif bt in ("MTEXT", "TEXT"):
                    try:
                        bx, by = be.dxf.insert.x, be.dxf.insert.y
                        bh = (be.dxf.char_height if bt == "MTEXT" else be.dxf.height)
                        bw = bh * 3
                        xs += [bx, bx + bw]
                        ys += [by, by + bh]
                    except Exception:
                        pass
    except Exception:
        pass

    # Fallback: defpoints
    if not xs:
        for attr in ("defpoint", "defpoint2", "defpoint3",
                     "defpoint4", "defpoint5", "text_midpoint"):
            try:
                p = getattr(e.dxf, attr)
                if abs(p.x) > 1e-9 or abs(p.y) > 1e-9:
                    xs.append(p.x); ys.append(p.y)
            except Exception:
                pass

    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def collect_annotations(msp, doc: Drawing) -> list[tuple[int, BBox]]:
    """
    Собирает все аннотации из modelspace.
    Возвращает список (class_id, (xmin, ymin, xmax, ymax)) в единицах DXF.

    Обрабатываемые типы:
      CLASS_TEXT (0)      : TEXT, MTEXT — прямые и через ATTRIB в блоках
      CLASS_DIMENSION (1) : DIMENSION, MULTILEADER, LEADER
      CLASS_REGION (2)    : Области для маскирования (YOLO_TABLE / YOLO_STAMP / YOLO_FRAME)
    """
    result: list[tuple[int, BBox]] = []
    _frame_polys: list[BBox] = []   # накапливаем YOLO_FRAME для парной обработки

    for e in msp:
        t = e.dxftype()

        # ── Текст ─────────────────────────────────────────────────────
        if t in ("TEXT", "MTEXT"):
            fn = _text_bbox if t == "TEXT" else _mtext_bbox
            bb = fn(e)
            if bb and bb[2] > bb[0] and bb[3] > bb[1]:
                result.append((CLASS_TEXT, bb))

        elif t == "ATTDEF":
            # Определение атрибута блока — берём как текст если есть координата
            bb = _text_bbox(e)
            if bb and bb[2] > bb[0] and bb[3] > bb[1]:
                result.append((CLASS_TEXT, bb))

        elif t == "INSERT":
            # Блок с атрибутами (штамп чертежа, экспликация и т.п.)
            # Извлекаем ATTRIB (заполненные текстом атрибуты) из вставки
            try:
                for attrib in e.attribs:
                    bb = _text_bbox(attrib)
                    if bb and bb[2] > bb[0] and bb[3] > bb[1]:
                        result.append((CLASS_TEXT, bb))
            except Exception:
                pass

        # ── Размеры ───────────────────────────────────────────────────
        elif t in ("DIMENSION", "MULTILEADER"):
            bb = _dim_bbox(e, doc)
            if bb and bb[2] > bb[0] and bb[3] > bb[1]:
                result.append((CLASS_DIMENSION, bb))

        elif t == "LEADER":
            # Устаревший LEADER (без MTEXT); берём по точкам
            try:
                pts = list(e.vertices)
                if len(pts) >= 2:
                    xs = [p[0] for p in pts];  ys = [p[1] for p in pts]
                    bb = (min(xs), min(ys), max(xs), max(ys))
                    if bb[2] > bb[0] and bb[3] > bb[1]:
                        result.append((CLASS_DIMENSION, bb))
            except Exception:
                pass

        # ── Области для маскирования (ручная разметка) ───────────────
        elif t == "LWPOLYLINE":
            try:
                layer = e.dxf.layer
                if layer not in REGION_LAYERS:
                    pass
                else:
                    pts_raw = list(e.get_points(format="xy"))
                    if len(pts_raw) >= 2:
                        xs = [p[0] for p in pts_raw]
                        ys = [p[1] for p in pts_raw]
                        bb = (min(xs), min(ys), max(xs), max(ys))
                        if bb[2] > bb[0] and bb[3] > bb[1]:
                            # YOLO_FRAME собираем отдельно (нужна пара)
                            if layer == "YOLO_FRAME":
                                _frame_polys.append(bb)
                            else:
                                result.append((CLASS_REGION, bb))
            except Exception:
                pass

        elif t == "SOLID":
            try:
                if e.dxf.layer in REGION_LAYERS:
                    corners = [e.dxf.vtx0, e.dxf.vtx1,
                               e.dxf.vtx2, e.dxf.vtx3]
                    xs = [v.x for v in corners];  ys = [v.y for v in corners]
                    bb = (min(xs), min(ys), max(xs), max(ys))
                    if bb[2] > bb[0] and bb[3] > bb[1]:
                        result.append((CLASS_REGION, bb))
            except Exception:
                pass

    # ── Обработка YOLO_FRAME ──────────────────────────────────────────
    result.extend(_resolve_frame_regions(_frame_polys))

    return result


def _resolve_frame_regions(frame_polys: list[BBox]) -> list[tuple[int, BBox]]:
    """
    Вычисляет CLASS_REGION аннотации из полилиний на слое YOLO_FRAME.

    1 полилиния  → bbox используется напрямую (старое поведение).
    2 полилинии  → вычисляются 4 полосы между внешней и внутренней рамкой:
                   полоса_верх, полоса_низ, полоса_лево, полоса_право.
                   Интерьер чертежа (внутри внутренней рамки) остаётся открытым.
    3+ полилиний → каждая используется напрямую (нестандартный случай).
    """
    if not frame_polys:
        return []

    if len(frame_polys) == 2:
        # Определяем внешнюю (большую) и внутреннюю (меньшую) по площади
        outer, inner = sorted(
            frame_polys,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )
        ox0, oy0, ox1, oy1 = outer   # внешняя: xmin ymin xmax ymax (DXF, Y↑)
        ix0, iy0, ix1, iy1 = inner   # внутренняя

        strips: list[BBox] = [
            (ox0, iy1, ox1, oy1),    # верхняя полоса
            (ox0, oy0, ox1, iy0),    # нижняя полоса
            (ox0, iy0, ix0, iy1),    # левая  полоса
            (ix1, iy0, ox1, iy1),    # правая полоса
        ]
        return [
            (CLASS_REGION, s)
            for s in strips
            if s[2] > s[0] and s[3] > s[1]   # только ненулевые
        ]

    # 1 полилиния или 3+ — напрямую
    return [
        (CLASS_REGION, bb)
        for bb in frame_polys
        if bb[2] > bb[0] and bb[3] > bb[1]
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Шум (без rotation)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_noise(
    arr: np.ndarray,
    noise_type: str,
    params: dict,
    rng: np.random.Generator,
    edge_blur: float = 1.0,
) -> np.ndarray:
    if noise_type == "gaussian_noise":
        if edge_blur > 0:
            arr = gaussian_filter(arr, sigma=edge_blur)
        return np.clip(arr + rng.normal(0.0, params["sigma"], arr.shape).astype(np.float32),
                       0.0, 1.0)
    if noise_type == "salt_pepper":
        result = arr.copy()
        n      = arr.size
        n_each = max(1, int(n * params["density"] / 2))
        flat   = result.ravel()
        flat[rng.choice(n, size=n_each, replace=False)] = 1.0
        flat[rng.choice(n, size=n_each, replace=False)] = 0.0
        return flat.reshape(arr.shape)
    if noise_type == "gaussian_blur":
        return gaussian_filter(arr, sigma=params["sigma"])
    return arr


def _save_png(arr: np.ndarray, path: Path, noise_type: str = "") -> None:
    """Сохраняет float [0,1] массив как 8-bit grayscale PNG."""
    if noise_type != "gaussian_blur":
        # Бинаризуем все типы кроме blur
        arr = (arr > 0.5).astype(np.float32)
    uint8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
    Image.fromarray(uint8, mode="L").save(str(path), format="PNG", optimize=True)


# ──────────────────────────────────────────────────────────────────────────────
# Конвертация bbox → YOLO
# ──────────────────────────────────────────────────────────────────────────────

def bbox_to_yolo(
    xmin_dxf: float, ymin_dxf: float,
    xmax_dxf: float, ymax_dxf: float,
    x_origin: float, y_max_full: float,
    px_per_unit: float,
    crop_x0: int, crop_y0: int,
    crop_size: int,
) -> Optional[tuple[float, float, float, float]]:
    """
    Конвертирует DXF-bbox в YOLO (x_center, y_center, w, h) нормализованные [0,1].
    DXF: Y↑. PIL: Y↓. YOLO: Y↓, нормализовано на crop_size.
    Возвращает None если bbox полностью вне кропа.
    """
    # DXF → полный растр (Y↓)
    ix0 = (xmin_dxf - x_origin)  * px_per_unit
    ix1 = (xmax_dxf - x_origin)  * px_per_unit
    iy0 = (y_max_full - ymax_dxf) * px_per_unit  # ymax_dxf → меньший Y в image
    iy1 = (y_max_full - ymin_dxf) * px_per_unit

    # Полный растр → локальные координаты кропа
    lx0 = ix0 - crop_x0
    lx1 = ix1 - crop_x0
    ly0 = iy0 - crop_y0
    ly1 = iy1 - crop_y0

    # Clip по границам кропа
    lx0c = max(0.0, min(float(crop_size), lx0))
    lx1c = max(0.0, min(float(crop_size), lx1))
    ly0c = max(0.0, min(float(crop_size), ly0))
    ly1c = max(0.0, min(float(crop_size), ly1))

    if lx1c - lx0c < 1.0 or ly1c - ly0c < 1.0:
        return None  # bbox вне кропа или вырожден

    xc = (lx0c + lx1c) / 2 / crop_size
    yc = (ly0c + ly1c) / 2 / crop_size
    w  = (lx1c - lx0c) / crop_size
    h  = (ly1c - ly0c) / crop_size
    return xc, yc, w, h


# ──────────────────────────────────────────────────────────────────────────────
# Основная функция обработки одного файла
# ──────────────────────────────────────────────────────────────────────────────

def process_file(
    dxf_path:       Path,
    units_mm:       float,
    scale:          float,
    dpi:            float,
    crop_size:      int,
    overlap:        int,
    line_width:     int,
    min_ann:        int,
    no_noise:       bool,
    edge_blur:      float,
    seed:           int,
    output_dir:     Path,
    train_ids:      set[str],
) -> list[dict]:
    """Обрабатывает один DXF, возвращает записи для манифеста."""

    print(f"\n[{dxf_path.name}]")
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # ── Единицы и масштаб ────────────────────────────────────────────────────
    mm_per_px   = 25.4 / dpi

    # ── Предварительный bbox с IQR-фильтрацией выбросов ─────────────────────
    _xs_raw = np.array([c for e in msp if e.dxftype() == "LINE"
                        for c in (e.dxf.start.x, e.dxf.end.x)], dtype=np.float64)
    _ys_raw = np.array([c for e in msp if e.dxftype() == "LINE"
                        for c in (e.dxf.start.y, e.dxf.end.y)], dtype=np.float64)
    if len(_xs_raw) == 0:
        print("  [warn] LINE не найдены, пропускаем.")
        return []

    def _iqr_clip(a: np.ndarray, k: float = 5.0) -> np.ndarray:
        q1, q3 = np.percentile(a, 25), np.percentile(a, 75)
        iqr = q3 - q1
        return a[(a >= q1 - k * iqr) & (a <= q3 + k * iqr)]

    _xs = _iqr_clip(_xs_raw)
    _ys = _iqr_clip(_ys_raw)
    if len(_xs) == 0 or len(_ys) == 0:
        _xs, _ys = _xs_raw, _ys_raw  # fallback

    n_outliers = (len(_xs_raw) - len(_xs)) // 2
    if n_outliers:
        print(f"  [warn] Отфильтровано выбросов (IQR): {n_outliers} сегментов")

    _w = float(_xs.max() - _xs.min())
    _h = float(_ys.max() - _ys.min())

    # Если при заданных единицах растр > 100 МПикс — подбираем корректные единицы
    MAX_MPIX = 100.0
    _ppu = units_mm / scale / mm_per_px
    if _w * _ppu * _h * _ppu > MAX_MPIX * 1e6:
        candidates = [(1.0, "mm"), (10.0, "cm"), (1000.0, "m")]
        chosen_mm, chosen_label = units_mm, f"override ({units_mm} мм/ед.)"
        for cand_mm, cand_label in candidates:
            ppu_c = cand_mm / scale / mm_per_px
            if _w * ppu_c * _h * ppu_c <= MAX_MPIX * 1e6:
                chosen_mm, chosen_label = cand_mm, cand_label
                break
        if chosen_mm != units_mm:
            print(f"  [warn] Растр слишком большой при --units ({units_mm} мм/ед.). "
                  f"Авто-подбор: {chosen_label}")
            units_mm = chosen_mm
        else:
            print(f"  [warn] Растр превышает {MAX_MPIX:.0f} МПикс даже после подбора единиц. "
                  f"Пропускаем файл.")
            return []

    px_per_unit = units_mm / scale / mm_per_px
    print(f"  px/unit = {px_per_unit:.3f}  ({units_mm} мм/ед., 1:{scale:.0f}, {dpi:.0f} DPI)")

    # ── Bbox геометрии (по IQR-очищенным координатам) ────────────────────────
    x_min, y_min = float(_xs.min()), float(_ys.min())
    x_max, y_max = float(_xs.max()), float(_ys.max())
    img_w = max(1, round((x_max - x_min) * px_per_unit))
    img_h = max(1, round((y_max - y_min) * px_per_unit))
    print(f"  Растр: {img_w} × {img_h} px")

    # ── Растеризация ──────────────────────────────────────────────────────────
    print("  Растеризация ...", end=" ", flush=True)
    if _HAS_MPL_BACKEND:
        # ezdxf + matplotlib: реальный текст, DIMENSION, правильные шрифты
        gray_arr = _render_with_matplotlib(
            doc, msp, x_min, y_min, x_max, y_max, img_w, img_h, dpi
        )
        base_arr = gray_arr.astype(np.float32) / 255.0
    else:
        # Запасной вариант: только геометрия через PIL
        img = Image.new("L", (img_w, img_h), color=255)
        _draw_geometry(ImageDraw.Draw(img), msp, doc, x_min, y_max, px_per_unit, line_width)
        base_arr = np.array(img, dtype=np.float32) / 255.0
    print("готово")

    # ── Аннотации ────────────────────────────────────────────────────────────
    annotations = collect_annotations(msp, doc)
    print(f"  Аннотаций: {len(annotations)}  "
          f"(text={sum(1 for c,_ in annotations if c==CLASS_TEXT)}, "
          f"dim={sum(1 for c,_ in annotations if c==CLASS_DIMENSION)})")

    # ── Идентификатор чертежа ─────────────────────────────────────────────────
    raw_id = dxf_path.stem.replace("_full_annotated", "")
    try:
        drawing_id = f"{int(raw_id):03d}"
    except ValueError:
        drawing_id = raw_id

    # ── Сетка кропов ─────────────────────────────────────────────────────────
    step   = max(1, crop_size - overlap)
    cols   = max(1, -(-img_w // step))
    rows   = max(1, -(-img_h // step))
    rng    = np.random.default_rng(seed)

    saved = skipped_ann = 0
    manifest_entries: list[dict] = []

    print(f"  Сетка: {cols}×{rows} = {cols*rows} кропов  (step={step} px)")

    for row in range(rows):
        for col in range(cols):
            x0 = col * step
            y0 = row * step

            # ── Собираем YOLO-разметку для кропа ─────────────────────────────
            yolo_lines: list[str] = []
            for cls_id, (bx0, by0, bx1, by1) in annotations:
                yolo = bbox_to_yolo(
                    bx0, by0, bx1, by1,
                    x_min, y_max, px_per_unit,
                    x0, y0, crop_size,
                )
                if yolo is not None:
                    xc, yc, w, h = yolo
                    yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            if len(yolo_lines) < min_ann:
                skipped_ann += 1
                continue

            crop_name = f"{drawing_id}_crop_{saved:04d}"

            # ── Вырезаем кроп ─────────────────────────────────────────────────
            region_arr = base_arr[y0:y0 + crop_size, x0:x0 + crop_size]
            if region_arr.shape != (crop_size, crop_size):
                padded = np.ones((crop_size, crop_size), dtype=np.float32)
                h_r, w_r = region_arr.shape
                padded[:h_r, :w_r] = region_arr
                region_arr = padded

            # ── Сплит (по drawing_id, не по кропу — чтобы не было утечки) ────
            split = "train" if drawing_id in train_ids else "val"

            img_dir = output_dir / "images" / split
            lbl_dir = output_dir / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            label_content = "\n".join(yolo_lines)

            # Чистый кроп
            variant = "clean"
            png_path = img_dir / f"{crop_name}_{variant}.png"
            txt_path = lbl_dir / f"{crop_name}_{variant}.txt"
            _save_png(region_arr, png_path)
            txt_path.write_text(label_content, encoding="utf-8")

            manifest_entries.append({
                "id":         f"{crop_name}_{variant}",
                "drawing_id": drawing_id,
                "crop_id":    f"crop_{saved:04d}",
                "split":      split,
                "noise_type": None,
                "noise_level": 0,
                "params":     {},
                "image":      str(png_path.relative_to(DATASET_DIR)).replace("\\", "/"),
                "label":      str(txt_path.relative_to(DATASET_DIR)).replace("\\", "/"),
                "n_annotations": len(yolo_lines),
            })

            # Зашумлённые варианты
            if not no_noise:
                for noise_type, levels in NOISE_CONFIGS.items():
                    for level, params in levels:
                        noisy = _apply_noise(region_arr.copy(), noise_type,
                                             params, rng, edge_blur)
                        variant_n = f"{noise_type}_{level}"
                        png_n = img_dir / f"{crop_name}_{variant_n}.png"
                        txt_n = lbl_dir / f"{crop_name}_{variant_n}.txt"
                        _save_png(noisy, png_n, noise_type)
                        txt_n.write_text(label_content, encoding="utf-8")

                        p_str = ", ".join(f"{k}={v}" for k, v in params.items())
                        manifest_entries.append({
                            "id":          f"{crop_name}_{variant_n}",
                            "drawing_id":  drawing_id,
                            "crop_id":     f"crop_{saved:04d}",
                            "split":       split,
                            "noise_type":  noise_type,
                            "noise_level": int(level[1]),
                            "params":      params,
                            "image":       str(png_n.relative_to(DATASET_DIR)).replace("\\", "/"),
                            "label":       str(txt_n.relative_to(DATASET_DIR)).replace("\\", "/"),
                            "n_annotations": len(yolo_lines),
                        })

            saved += 1

    variants_per_crop = 1 + (0 if no_noise else sum(len(v) for v in NOISE_CONFIGS.values()))
    print(f"  Сохранено кропов : {saved}  "
          f"({saved * variants_per_crop} изображений, {skipped_ann} пропущено)")
    return manifest_entries


# ──────────────────────────────────────────────────────────────────────────────
# Сборка и train/val split
# ──────────────────────────────────────────────────────────────────────────────

def assign_splits(
    dxf_files: list[Path],
    val_split: float,
    seed: int,
) -> set[str]:
    """
    Делает split на уровне чертежей (не кропов), чтобы избежать утечки данных.
    Возвращает множество drawing_id, назначенных в train.
    """
    ids: list[str] = []
    for p in dxf_files:
        raw = p.stem.replace("_full_annotated", "")
        try:
            ids.append(f"{int(raw):03d}")
        except ValueError:
            ids.append(raw)

    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val   = max(1, round(len(shuffled) * val_split))
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])
    print(f"\nSplit (по чертежам):  train={sorted(train_ids)}  val={sorted(val_ids)}")
    return train_ids


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генерирует YOLO-датасет из *_full_annotated.dxf.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Конкретный DXF-файл. По умолчанию: все *_full_annotated.dxf в sources/.",
    )
    parser.add_argument(
        "--scale", type=float, required=True,
        help="Знаменатель масштаба (например, 100 для 1:100).",
    )
    parser.add_argument(
        "--units", type=parse_units_arg, default=None, metavar="UNITS",
        help="Единицы DXF: mm|cm|m|inch|ft или мм/ед.",
    )
    parser.add_argument("--dpi",          type=float, default=300.0)
    parser.add_argument("--crop-size",    type=int,   default=1280)
    parser.add_argument("--overlap",      type=int,   default=64)
    parser.add_argument("--line-width",   type=int,   default=2)
    parser.add_argument("--min-annotations", type=int, default=1,
                        help="Минимальное число аннотаций в кропе (0 = не фильтровать).")
    parser.add_argument("--val-split",    type=float, default=0.2,
                        help="Доля чертежей в val-выборке.")
    parser.add_argument("--no-noise",     action="store_true",
                        help="Генерировать только чистые изображения.")
    parser.add_argument("--edge-blur",    type=float, default=1.0,
                        help="Sigma предразмытия краёв перед gaussian_noise.")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Выходная папка. По умолчанию: yolo_dataset/.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (DATASET_DIR / "yolo_dataset")

    # Список файлов
    if args.source is not None:
        if not args.source.exists():
            print(f"Ошибка: файл не найден: {args.source}", file=sys.stderr)
            sys.exit(1)
        dxf_files = [args.source]
    else:
        sources_dir = DATASET_DIR / "sources"
        dxf_files = sorted(sources_dir.glob("*_full_annotated.dxf"))
        if not dxf_files:
            print("Файлы *_full_annotated.dxf не найдены.", file=sys.stderr)
            sys.exit(1)

    print(f"Файлов: {len(dxf_files)}")
    print(f"Параметры: crop={args.crop_size}px, overlap={args.overlap}px, "
          f"dpi={args.dpi}, scale=1:{args.scale:.0f}, "
          f"noise={'нет' if args.no_noise else 'все'}")

    # Train/val split по чертежам
    train_drawing_ids = assign_splits(dxf_files, args.val_split, args.seed)

    # Обрабатываем каждый файл
    all_entries: list[dict] = []
    for dxf_path in dxf_files:
        doc = ezdxf.readfile(str(dxf_path))
        units_mm, _ = resolve_units(doc, args.units)
        entries = process_file(
            dxf_path    = dxf_path,
            units_mm    = units_mm,
            scale       = args.scale,
            dpi         = args.dpi,
            crop_size   = args.crop_size,
            overlap     = args.overlap,
            line_width  = args.line_width,
            min_ann     = args.min_annotations,
            no_noise    = args.no_noise,
            edge_blur   = args.edge_blur,
            seed        = args.seed,
            output_dir  = output_dir,
            train_ids   = train_drawing_ids,
        )
        all_entries.extend(entries)

    # ── data.yaml ────────────────────────────────────────────────────────────
    yaml_path = output_dir / "data.yaml"
    yaml_content = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\nnc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")

    # ── yolo_manifest.json ───────────────────────────────────────────────────
    manifest_path = DATASET_DIR / "yolo_manifest.json"
    manifest = {
        "description": "YOLO dataset — text and dimension detection in architectural drawings",
        "classes":     CLASS_NAMES,
        "samples":     all_entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── Итог ─────────────────────────────────────────────────────────────────
    n_train = sum(1 for e in all_entries if e["split"] == "train" and e["noise_type"] is None)
    n_val   = sum(1 for e in all_entries if e["split"] == "val"   and e["noise_type"] is None)
    print(f"\n{'='*54}")
    print(f"  Готово.")
    print(f"  Чистых кропов : train={n_train}  val={n_val}")
    print(f"  Всего записей : {len(all_entries)}")
    print(f"  data.yaml     : {yaml_path}")
    print(f"  Манифест      : {manifest_path}")
    print(f"{'='*54}")


if __name__ == "__main__":
    main()
