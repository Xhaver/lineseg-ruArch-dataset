#!/usr/bin/env python3
"""
preprocess_dxf.py

Читает DXF из sources/unprocessed_geometry_only/, приводит все примитивы
к набору отдельных LINE-сущностей и записывает результат в sources/geometry_only/.
Исходные файлы НЕ изменяются.

Поддерживаемые примитивы:
  - LINE        -> сохраняется как есть
  - LWPOLYLINE  -> разбивается на LINE-сегменты (bulge-дуги — прямые хорды при --skip-curved)
  - CIRCLE      -> аппроксимируется LINE-сегментами (или пропускается при --skip-curved)
  - ARC         -> аппроксимируется LINE-сегментами (или пропускается при --skip-curved)
  - ELLIPSE     -> аппроксимируется LINE-сегментами (или пропускается при --skip-curved)
  - Прочие      -> удаляются

--skip-curved (по умолчанию включён):
    Пропускает ARC, CIRCLE, ELLIPSE и bulge-сегменты в LWPOLYLINE.
    Убирает дуги дверей и скруглений из GT-данных.
    Оставляет только прямолинейную геометрию.

Единицы чертежа:
    $INSUNITS нередко заполнен некорректно. Параметр --units позволяет
    задать единицы явно: mm | cm | m | inch | ft, или число (мм/ед.).

Использование:
    # Все файлы из unprocessed_geometry_only/ -> geometry_only/
    python preprocess_dxf.py --units m

    # Конкретный файл
    python preprocess_dxf.py path/to/file.dxf --units m --output-dir path/to/out/

    # Оставить дуги (старое поведение)
    python preprocess_dxf.py --units m --no-skip-curved
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import ezdxf
from ezdxf.document import Drawing

from paths import UNPROCESSED_GEOMETRY_DIR, GEOMETRY_ONLY_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Единицы
# ──────────────────────────────────────────────────────────────────────────────

_INSUNITS_TO_MM: dict[int, float] = {
    0:  1.0,           # Unitless — считаем мм
    1:  25.4,          # Inches
    2:  304.8,         # Feet
    3:  1_609_344.0,   # Miles
    4:  1.0,           # Millimeters
    5:  10.0,          # Centimeters
    6:  1_000.0,       # Meters
    7:  1_000_000.0,   # Kilometers
    8:  2.54e-5,       # Microinches
    9:  0.0254,        # Mils
    10: 914.4,         # Yards
    11: 1e-7,          # Angstroms
    12: 1e-6,          # Nanometers
    13: 1e-3,          # Microns
    14: 100.0,         # Decimeters
    15: 10_000.0,      # Decameters
    16: 100_000.0,     # Hectometers
    17: 1e9,           # Gigameters
}

_UNITS_ALIASES: dict[str, float] = {
    "mm":   1.0,
    "cm":   10.0,
    "m":    1_000.0,
    "km":   1_000_000.0,
    "inch": 25.4,
    "in":   25.4,
    "ft":   304.8,
}

_MM_TO_INSUNITS: dict[float, int] = {
    1.0:     4,
    10.0:    5,
    1_000.0: 6,
    25.4:    1,
    304.8:   2,
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
        known = ", ".join(_UNITS_ALIASES.keys())
        raise argparse.ArgumentTypeError(
            f"Неверное значение --units '{value}'. "
            f"Ожидается: {known} или положительное число (мм/ед.)."
        )


def resolve_units(doc: Drawing, override_mm: Optional[float]) -> tuple[float, str]:
    if override_mm is not None:
        return override_mm, f"задано вручную ({override_mm} мм/ед.)"
    insunits = int(doc.header.get("$INSUNITS", 0))
    scale = _INSUNITS_TO_MM.get(insunits, 1.0)
    label = f"$INSUNITS={insunits} -> {scale} мм/ед."
    if insunits not in _INSUNITS_TO_MM:
        label += "  [warn: неизвестный код, принято 1 мм/ед.]"
    return scale, label


# ──────────────────────────────────────────────────────────────────────────────
# Геометрические утилиты
# ──────────────────────────────────────────────────────────────────────────────

def n_segments_for_arc(radius_px: float, arc_angle_rad: float, delta_px: float) -> int:
    """Минимальное n, при котором стрелка хорды < delta_px."""
    if radius_px < 1e-9:
        return 2
    ratio = max(-1.0, min(1.0, 1.0 - delta_px / radius_px))
    half_step = math.acos(ratio)
    if half_step < 1e-12:
        return max(2, int(arc_angle_rad * 1000))
    return max(2, math.ceil(arc_angle_rad / (2.0 * half_step)))


def _attribs(entity) -> dict:
    attribs: dict = {"layer": entity.dxf.layer}
    for attr in ("color", "linetype", "lineweight", "true_color", "transparency"):
        if entity.dxf.hasattr(attr):
            attribs[attr] = entity.dxf.get(attr)
    return attribs


# ──────────────────────────────────────────────────────────────────────────────
# Конвертеры кривых -> список точек (x, y)
# ──────────────────────────────────────────────────────────────────────────────

def _arc_points(
    cx: float, cy: float, r: float,
    start_rad: float, end_rad: float,
    units_to_px: float, delta_px: float,
) -> list[tuple[float, float]]:
    arc_angle = abs(end_rad - start_rad)
    n = n_segments_for_arc(r * units_to_px, arc_angle, delta_px)
    return [
        (cx + r * math.cos(start_rad + (end_rad - start_rad) * i / n),
         cy + r * math.sin(start_rad + (end_rad - start_rad) * i / n))
        for i in range(n + 1)
    ]


def circle_to_lines(
    entity, units_to_px: float, delta_px: float, min_radius_px: float
) -> Optional[list[tuple[tuple[float, float], tuple[float, float]]]]:
    r = entity.dxf.radius
    if r * units_to_px < min_radius_px:
        return None
    cx, cy = entity.dxf.center.x, entity.dxf.center.y
    pts = _arc_points(cx, cy, r, 0, 2 * math.pi, units_to_px, delta_px)
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def arc_to_lines(
    entity, units_to_px: float, delta_px: float, min_radius_px: float
) -> Optional[list[tuple[tuple[float, float], tuple[float, float]]]]:
    r = entity.dxf.radius
    if r * units_to_px < min_radius_px:
        return None
    cx, cy = entity.dxf.center.x, entity.dxf.center.y
    start_rad = math.radians(entity.dxf.start_angle)
    end_rad   = math.radians(entity.dxf.end_angle)
    if end_rad <= start_rad:
        end_rad += 2 * math.pi
    pts = _arc_points(cx, cy, r, start_rad, end_rad, units_to_px, delta_px)
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def ellipse_to_lines(
    entity, units_to_px: float, delta_px: float, min_radius_px: float
) -> Optional[list[tuple[tuple[float, float], tuple[float, float]]]]:
    major_vec = entity.dxf.major_axis
    ratio     = entity.dxf.ratio
    cx, cy    = entity.dxf.center.x, entity.dxf.center.y
    a         = math.hypot(major_vec.x, major_vec.y)
    b         = a * ratio
    angle_rot = math.atan2(major_vec.y, major_vec.x)
    r_eff_px  = max(a, b) * units_to_px
    if r_eff_px < min_radius_px:
        return None
    start_p = entity.dxf.start_param
    end_p   = entity.dxf.end_param
    if end_p <= start_p:
        end_p += 2 * math.pi
    arc_angle = end_p - start_p
    n = n_segments_for_arc(r_eff_px, arc_angle, delta_px)
    cos_rot, sin_rot = math.cos(angle_rot), math.sin(angle_rot)
    pts: list[tuple[float, float]] = []
    for i in range(n + 1):
        t  = start_p + arc_angle * i / n
        lx = a * math.cos(t)
        ly = b * math.sin(t)
        pts.append((cx + lx * cos_rot - ly * sin_rot,
                     cy + lx * sin_rot + ly * cos_rot))
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


# ──────────────────────────────────────────────────────────────────────────────
# Разбивка LWPOLYLINE -> LINE-сегменты
# ──────────────────────────────────────────────────────────────────────────────

def _bulge_arc_points(
    p1: tuple[float, float], p2: tuple[float, float],
    bulge: float, units_to_px: float, delta_px: float,
) -> list[tuple[float, float]]:
    """
    Аппроксимирует bulge-дугу между p1 и p2 точками.
    Bulge = tan(включённый_угол / 4).
    """
    x1, y1 = p1
    x2, y2 = p2
    chord = math.hypot(x2 - x1, y2 - y1)
    if chord < 1e-12:
        return [p1, p2]
    theta = 4.0 * math.atan(abs(bulge))
    r     = chord / (2.0 * math.sin(theta / 2.0))
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    d  = math.sqrt(max(0.0, r * r - (chord / 2.0) ** 2))
    dx, dy = (x2 - x1) / chord, (y2 - y1) / chord
    nx, ny = -dy, dx
    sign = 1.0 if bulge > 0 else -1.0
    cx = mx + sign * d * nx
    cy = my + sign * d * ny
    start_rad = math.atan2(y1 - cy, x1 - cx)
    end_rad   = math.atan2(y2 - cy, x2 - cx)
    if bulge > 0:
        while end_rad < start_rad:
            end_rad += 2 * math.pi
    else:
        while end_rad > start_rad:
            end_rad -= 2 * math.pi
    return _arc_points(cx, cy, r, start_rad, end_rad, units_to_px, delta_px)


def lwpolyline_to_lines(
    entity, units_to_px: float, delta_px: float, skip_curved: bool
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Разбивает LWPOLYLINE на LINE-сегменты.
    При skip_curved=True bulge-дуги заменяются прямыми хордами.
    """
    raw = list(entity.get_points(format="xyb"))
    n   = len(raw)
    if n < 2:
        return []
    indices = list(range(n))
    pairs   = list(zip(indices, indices[1:]))
    if entity.closed:
        pairs.append((n - 1, 0))
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i, j in pairs:
        x1, y1, bulge = raw[i]
        x2, y2, _     = raw[j]
        p1, p2 = (x1, y1), (x2, y2)
        if abs(bulge) < 1e-10 or skip_curved:
            segments.append((p1, p2))
        else:
            pts = _bulge_arc_points(p1, p2, bulge, units_to_px, delta_px)
            for k in range(len(pts) - 1):
                segments.append((pts[k], pts[k + 1]))
    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Обработка одного файла
# ──────────────────────────────────────────────────────────────────────────────

def _robust_bounds(values: list[float], k: float = 5.0) -> tuple[float, float]:
    """Return [median - k*MAD, median + k*MAD].

    Uses Median Absolute Deviation — robust up to 50 % outliers,
    unlike IQR which breaks once >25 % of points are outliers.
    """
    if len(values) < 8:
        return float("-inf"), float("inf")
    s = sorted(values)
    n = len(s)
    median = s[n // 2]
    deviations = sorted(abs(x - median) for x in s)
    mad = deviations[n // 2]
    if mad < 1e-12:
        return float("-inf"), float("inf")
    return median - k * mad, median + k * mad


def process_file(
    src_path:        Path,
    out_path:        Path,
    dpi:             float,
    min_radius_mm:   float,
    delta_px:        float,
    units_mm_override: Optional[float],
    skip_curved:     bool,
) -> None:
    print(f"\n[{src_path.name}]  ->  {out_path}")
    doc = ezdxf.readfile(str(src_path))
    msp = doc.modelspace()

    units_to_mm, units_label = resolve_units(doc, units_mm_override)
    mm_per_px     = 25.4 / dpi
    units_to_px   = units_to_mm / mm_per_px
    min_radius_px = min_radius_mm / mm_per_px

    print(f"  Единицы  : {units_label}")
    print(f"  DPI      : {dpi}  ->  1 px = {mm_per_px:.4f} мм")
    if not skip_curved:
        print(f"  min_r    : {min_radius_mm} мм = {min_radius_px:.1f} px")
        print(f"  delta_px : {delta_px} px")
    print(f"  skip_curved: {skip_curved}")

    if units_mm_override is not None:
        correct_insunits = _MM_TO_INSUNITS.get(units_to_mm)
        if correct_insunits is not None:
            doc.header["$INSUNITS"] = correct_insunits

    entities = list(msp)
    to_delete: list = []
    new_lines: list[tuple[tuple[float, float], tuple[float, float], dict]] = []

    stats: dict[str, int] = {
        "line_kept":          0,
        "lwpoly_segments":    0,
        "lwpoly_count":       0,
        "lwpoly_bulge_chord": 0,   # bulge-сегментов заменено прямыми хордами
        "circle_segments":    0,
        "circle_skipped":     0,
        "arc_segments":       0,
        "arc_skipped":        0,
        "ellipse_segments":   0,
        "ellipse_skipped":    0,
        "skipped_small":      0,
        "skipped_type":       0,
        "post_cleanup":       0,
    }
    skipped_types: dict[str, int] = {}
    post_cleanup_types: dict[str, int] = {}

    for entity in entities:
        etype   = entity.dxftype()
        attribs = _attribs(entity)

        if etype == "LINE":
            stats["line_kept"] += 1

        elif etype == "LWPOLYLINE":
            to_delete.append(entity)
            raw = list(entity.get_points(format="xyb"))
            n_bulge = sum(1 for _, _, b in raw if abs(b) >= 1e-10)
            segs = lwpolyline_to_lines(entity, units_to_px, delta_px, skip_curved)
            for p1, p2 in segs:
                new_lines.append((p1, p2, attribs))
            stats["lwpoly_count"]    += 1
            stats["lwpoly_segments"] += len(segs)
            if skip_curved and n_bulge:
                stats["lwpoly_bulge_chord"] += n_bulge

        elif etype == "CIRCLE":
            to_delete.append(entity)
            if skip_curved:
                stats["circle_skipped"] += 1
            else:
                segs = circle_to_lines(entity, units_to_px, delta_px, min_radius_px)
                if segs is None:
                    stats["skipped_small"] += 1
                else:
                    for p1, p2 in segs:
                        new_lines.append((p1, p2, attribs))
                    stats["circle_segments"] += len(segs)

        elif etype == "ARC":
            to_delete.append(entity)
            if skip_curved:
                stats["arc_skipped"] += 1
            else:
                segs = arc_to_lines(entity, units_to_px, delta_px, min_radius_px)
                if segs is None:
                    stats["skipped_small"] += 1
                else:
                    for p1, p2 in segs:
                        new_lines.append((p1, p2, attribs))
                    stats["arc_segments"] += len(segs)

        elif etype == "ELLIPSE":
            to_delete.append(entity)
            if skip_curved:
                stats["ellipse_skipped"] += 1
            else:
                segs = ellipse_to_lines(entity, units_to_px, delta_px, min_radius_px)
                if segs is None:
                    stats["skipped_small"] += 1
                else:
                    for p1, p2 in segs:
                        new_lines.append((p1, p2, attribs))
                    stats["ellipse_segments"] += len(segs)

        elif etype == "INSERT":
            try:
                for virt_entity in entity.virtual_entities():
                    ve_type = virt_entity.dxftype()
                    if ve_type == "LINE":
                        p1 = (virt_entity.dxf.start.x, virt_entity.dxf.start.y)
                        p2 = (virt_entity.dxf.end.x,   virt_entity.dxf.end.y)
                        new_lines.append((p1, p2, _attribs(virt_entity)))
                        stats["line_kept"] += 1
                    elif ve_type == "LWPOLYLINE":
                        segs = lwpolyline_to_lines(virt_entity, units_to_px, delta_px, skip_curved)
                        for p1, p2 in segs:
                            new_lines.append((p1, p2, attribs))
                        stats["lwpoly_segments"] += len(segs)
                    elif ve_type in ("CIRCLE", "ARC", "ELLIPSE") and not skip_curved:
                        fn = {"CIRCLE": circle_to_lines,
                              "ARC":    arc_to_lines,
                              "ELLIPSE": ellipse_to_lines}[ve_type]
                        segs = fn(virt_entity, units_to_px, delta_px, min_radius_px)
                        if segs:
                            for p1, p2 in segs:
                                new_lines.append((p1, p2, attribs))
            except Exception:
                pass
            to_delete.append(entity)
            skipped_types["INSERT(exploded)"] = skipped_types.get("INSERT(exploded)", 0) + 1

        else:
            to_delete.append(entity)
            stats["skipped_type"] += 1
            skipped_types[etype] = skipped_types.get(etype, 0) + 1

    for entity in to_delete:
        msp.delete_entity(entity)

    # IQR-фильтрация выбросов (артефакты INSERT-блоков на далёких координатах)
    all_xs: list[float] = []
    all_ys: list[float] = []
    for e in msp:
        if e.dxftype() == "LINE":
            all_xs += [e.dxf.start.x, e.dxf.end.x]
            all_ys += [e.dxf.start.y, e.dxf.end.y]
    for p1, p2, _ in new_lines:
        all_xs += [p1[0], p2[0]]
        all_ys += [p1[1], p2[1]]

    x_lo, x_hi = _robust_bounds(all_xs)
    y_lo, y_hi = _robust_bounds(all_ys)

    def _in_bounds(x: float, y: float) -> bool:
        return x_lo <= x <= x_hi and y_lo <= y <= y_hi

    outliers_removed = 0
    for e in list(msp):
        if e.dxftype() == "LINE":
            if not (_in_bounds(e.dxf.start.x, e.dxf.start.y) and
                    _in_bounds(e.dxf.end.x, e.dxf.end.y)):
                msp.delete_entity(e)
                outliers_removed += 1
                stats["line_kept"] -= 1

    before_new = len(new_lines)
    new_lines = [(p1, p2, a) for p1, p2, a in new_lines if _in_bounds(*p1) and _in_bounds(*p2)]
    outliers_removed += before_new - len(new_lines)
    stats["lwpoly_segments"] -= (before_new - len(new_lines))

    if outliers_removed:
        print(f"  IQR-выбросы удалены         : {outliers_removed} LINE-сегментов")

    for p1, p2, attribs in new_lines:
        msp.add_line(start=p1, end=p2, dxfattribs=attribs)

    # Пост-проверка: гарантируем только LINE
    for entity in list(msp):
        if entity.dxftype() != "LINE":
            t = entity.dxftype()
            post_cleanup_types[t] = post_cleanup_types.get(t, 0) + 1
            stats["post_cleanup"] += 1
            msp.delete_entity(entity)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.dxf")
    try:
        doc.saveas(str(tmp))
        tmp.replace(out_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    # Статистика
    total = (stats["line_kept"] + stats["lwpoly_segments"] +
             stats["circle_segments"] + stats["arc_segments"] +
             stats["ellipse_segments"])

    print(f"  LINE  сохранено              : {stats['line_kept']}")
    if stats["lwpoly_count"]:
        print(f"  LWPOLYLINE ({stats['lwpoly_count']} шт.) -> LINE : {stats['lwpoly_segments']}")
        if stats["lwpoly_bulge_chord"]:
            print(f"    в т.ч. bulge -> хорда       : {stats['lwpoly_bulge_chord']}")
    if skip_curved:
        total_skipped_curved = stats["arc_skipped"] + stats["circle_skipped"] + stats["ellipse_skipped"]
        if total_skipped_curved:
            print(f"  Пропущено кривых (--skip-curved):")
            if stats["arc_skipped"]:    print(f"    ARC    : {stats['arc_skipped']}")
            if stats["circle_skipped"]: print(f"    CIRCLE : {stats['circle_skipped']}")
            if stats["ellipse_skipped"]:print(f"    ELLIPSE: {stats['ellipse_skipped']}")
    else:
        if stats["circle_segments"]:  print(f"  CIRCLE  -> LINE              : {stats['circle_segments']}")
        if stats["arc_segments"]:     print(f"  ARC     -> LINE              : {stats['arc_segments']}")
        if stats["ellipse_segments"]: print(f"  ELLIPSE -> LINE              : {stats['ellipse_segments']}")
        if stats["skipped_small"]:    print(f"  Пропущено (r < {min_radius_mm} мм) : {stats['skipped_small']}")
    if skipped_types:
        print(f"  Удалено (не геометрия)       : {stats['skipped_type']}")
        for t, cnt in sorted(skipped_types.items(), key=lambda x: -x[1]):
            print(f"    {t}: {cnt}")
    if post_cleanup_types:
        print(f"  Пост-очистка (не-LINE)       : {stats['post_cleanup']}")
        for t, cnt in sorted(post_cleanup_types.items(), key=lambda x: -x[1]):
            print(f"    {t}: {cnt}")
    print(f"  Итого LINE в выходном файле  : {total}")


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Приводит все примитивы DXF к отдельным LINE-сегментам. "
            "Читает из sources/unprocessed_geometry_only/, "
            "записывает в sources/geometry_only/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path", type=Path, nargs="?",
        help=(
            "DXF-файл или папка с DXF-файлами. "
            f"По умолчанию: {UNPROCESSED_GEOMETRY_DIR.name}/"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, metavar="DIR",
        help=(
            "Папка для выходных файлов. "
            f"По умолчанию: sources/geometry_only/"
        ),
    )
    parser.add_argument(
        "--units", type=parse_units_arg, default=None, metavar="UNITS",
        help="Единицы чертежа: mm | cm | m | inch | ft или мм/ед. числом.",
    )
    parser.add_argument("--dpi",           type=float, default=300.0)
    parser.add_argument("--min-radius-mm", type=float, default=2.0,
                        help="Минимальный радиус (мм) при --no-skip-curved")
    parser.add_argument("--delta-px",      type=float, default=0.5,
                        help="Допуск стрелки хорды (px) при --no-skip-curved")
    parser.add_argument(
        "--skip-curved", default=True, action=argparse.BooleanOptionalAction,
        help="Пропускать ARC, CIRCLE, ELLIPSE и bulge-дуги в LWPOLYLINE (рекомендуется).",
    )
    args = parser.parse_args()

    # Определяем входные файлы
    if args.path is None:
        src_dir = UNPROCESSED_GEOMETRY_DIR
        if not src_dir.is_dir():
            print(f"Ошибка: папка не найдена: {src_dir}", file=sys.stderr)
            sys.exit(1)
        files = sorted(src_dir.glob("*.dxf"))
    elif args.path.is_dir():
        src_dir = args.path
        files = sorted(args.path.glob("*.dxf"))
    elif args.path.is_file():
        src_dir = args.path.parent
        files = [args.path]
    else:
        print(f"Ошибка: путь не найден: {args.path}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print(f"DXF-файлы не найдены в: {src_dir}", file=sys.stderr)
        sys.exit(1)

    # Определяем выходную папку
    out_dir = args.output_dir if args.output_dir else GEOMETRY_ONLY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Входная папка : {src_dir}")
    print(f"Выходная папка: {out_dir}")
    print(f"Найдено файлов: {len(files)}")
    print(f"skip_curved   : {args.skip_curved}")

    for f in files:
        out_path = out_dir / f.name
        process_file(
            src_path=f,
            out_path=out_path,
            dpi=args.dpi,
            min_radius_mm=args.min_radius_mm,
            delta_px=args.delta_px,
            units_mm_override=args.units,
            skip_curved=args.skip_curved,
        )

    print("\nГотово.")


if __name__ == "__main__":
    main()
