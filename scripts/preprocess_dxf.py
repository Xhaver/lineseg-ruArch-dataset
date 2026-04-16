#!/usr/bin/env python3
"""
preprocess_dxf.py

Приводит все примитивы DXF к набору отдельных LINE-сущностей:
  - LINE       -> оставляется как есть
  - LWPOLYLINE -> разбивается на отдельные LINE-сегменты (с обработкой bulge-дуг)
  - CIRCLE     -> аппроксимируется LINE-сегментами
  - ARC        -> аппроксимируется LINE-сегментами
  - ELLIPSE    -> аппроксимируется LINE-сегментами
  - Прочие     -> удаляются из файла

После конвертации выполняется пост-проверка: повторный обход modelspace
гарантирует, что в выходном файле остались исключительно LINE-примитивы.
Это улавливает штриховки (HATCH), размеры (DIMENSION), тексты (TEXT/MTEXT),
блоки (INSERT), SPLINE и другие объекты, которые могли не попасть в первый
проход (например, вложенные или динамически добавленные ezdxf-ом).

Цель: GT-файл содержит только LINE-примитивы, каждый из которых прямо
сопоставим с отрезком, возвращённым детектором (LSD/Hough).

Число сегментов при аппроксимации кривых:
    delta = R * (1 - cos(theta/2)) < delta_px
    n >= arc_angle / (2 * arccos(1 - delta_px / R))

Единицы чертежа:
    $INSUNITS нередко заполнен некорректно. Параметр --units позволяет
    задать единицы явно: mm | cm | m | inch | ft, или число (мм/ед.).

Использование:
    # Все *_geometry_only.dxf в папке sources/ (по умолчанию)
    python preprocess_dxf.py --units m

    # Конкретная папка или файл
    python preprocess_dxf.py path/to/sources/ --units m
    python preprocess_dxf.py path/to/file.dxf --units m
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
    """Общий генератор точек дуги (по часовой или против)."""
    arc_angle = abs(end_rad - start_rad)
    n = n_segments_for_arc(r * units_to_px, arc_angle, delta_px)
    return [
        (cx + r * math.cos(start_rad + (end_rad - start_rad) * i / n),
         cy + r * math.sin(start_rad + (end_rad - start_rad) * i / n))
        for i in range(n + 1)
    ]


def circle_to_lines(
    entity, units_to_px: float, delta_px: float, min_radius_px: float
) -> Optional[list[tuple[tuple[float,float], tuple[float,float]]]]:
    r = entity.dxf.radius
    if r * units_to_px < min_radius_px:
        return None
    cx, cy = entity.dxf.center.x, entity.dxf.center.y
    pts = _arc_points(cx, cy, r, 0, 2 * math.pi, units_to_px, delta_px)
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def arc_to_lines(
    entity, units_to_px: float, delta_px: float, min_radius_px: float
) -> Optional[list[tuple[tuple[float,float], tuple[float,float]]]]:
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
) -> Optional[list[tuple[tuple[float,float], tuple[float,float]]]]:
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
# Разбивка LWPOLYLINE -> LINE-сегменты (с обработкой bulge-дуг)
# ──────────────────────────────────────────────────────────────────────────────

def _bulge_arc_points(
    p1: tuple[float, float], p2: tuple[float, float],
    bulge: float, units_to_px: float, delta_px: float,
) -> list[tuple[float, float]]:
    """
    Аппроксимирует bulge-дугу между p1 и p2 точками.

    Bulge = tan(включённый_угол / 4).
    Знак: + -> CCW (дуга влево от направления p1->p2),
           - -> CW  (дуга вправо).
    """
    x1, y1 = p1
    x2, y2 = p2
    chord = math.hypot(x2 - x1, y2 - y1)
    if chord < 1e-12:
        return [p1, p2]

    theta = 4.0 * math.atan(abs(bulge))          # включённый угол, рад
    r     = chord / (2.0 * math.sin(theta / 2.0)) # радиус

    # Центр дуги
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    d  = math.sqrt(max(0.0, r * r - (chord / 2.0) ** 2))
    dx, dy = (x2 - x1) / chord, (y2 - y1) / chord  # единичный вектор хорды
    nx, ny = -dy, dx                                 # левая нормаль

    sign = 1.0 if bulge > 0 else -1.0
    cx = mx + sign * d * nx
    cy = my + sign * d * ny

    start_rad = math.atan2(y1 - cy, x1 - cx)
    end_rad   = math.atan2(y2 - cy, x2 - cx)

    if bulge > 0:   # CCW: end > start
        while end_rad < start_rad:
            end_rad += 2 * math.pi
    else:           # CW:  end < start
        while end_rad > start_rad:
            end_rad -= 2 * math.pi

    arc_angle = abs(end_rad - start_rad)
    return _arc_points(cx, cy, r, start_rad, end_rad, units_to_px, delta_px)


def lwpolyline_to_lines(
    entity, units_to_px: float, delta_px: float
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Разбивает LWPOLYLINE на отдельные LINE-сегменты.
    Обрабатывает bulge (дуговые сегменты внутри полилинии).
    """
    # format="xyb" -> (x, y, bulge)
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

        if abs(bulge) < 1e-10:
            segments.append((p1, p2))
        else:
            pts = _bulge_arc_points(p1, p2, bulge, units_to_px, delta_px)
            for k in range(len(pts) - 1):
                segments.append((pts[k], pts[k + 1]))

    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Обработка одного файла
# ──────────────────────────────────────────────────────────────────────────────

def process_file(
    path: Path,
    dpi: float,
    min_radius_mm: float,
    delta_px: float,
    units_mm_override: Optional[float],
) -> None:
    print(f"\n[{path.name}]")
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    units_to_mm, units_label = resolve_units(doc, units_mm_override)
    mm_per_px     = 25.4 / dpi
    units_to_px   = units_to_mm / mm_per_px
    min_radius_px = min_radius_mm / mm_per_px

    print(f"  Единицы  : {units_label}")
    print(f"  DPI      : {dpi}  ->  1 px = {mm_per_px:.4f} мм")
    print(f"  min_r    : {min_radius_mm} мм = {min_radius_px:.1f} px")
    print(f"  delta_px : {delta_px} px")

    if units_mm_override is not None:
        correct_insunits = _MM_TO_INSUNITS.get(units_to_mm)
        if correct_insunits is not None:
            doc.header["$INSUNITS"] = correct_insunits

    entities = list(msp)
    to_delete: list = []
    # (start, end, attribs)
    new_lines: list[tuple[tuple[float,float], tuple[float,float], dict]] = []

    stats: dict[str, int] = {
        "line_kept":        0,
        "lwpoly_segments":  0,   # LINE-сегментов из LWPOLYLINE
        "lwpoly_count":     0,   # сколько LWPOLYLINE обработано
        "circle_segments":  0,
        "arc_segments":     0,
        "ellipse_segments": 0,
        "skipped_small":    0,
        "skipped_type":     0,
        "post_cleanup":     0,   # удалено в пост-проверке
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
            segs = lwpolyline_to_lines(entity, units_to_px, delta_px)
            for p1, p2 in segs:
                new_lines.append((p1, p2, attribs))
            stats["lwpoly_count"]    += 1
            stats["lwpoly_segments"] += len(segs)

        elif etype == "CIRCLE":
            to_delete.append(entity)
            segs = circle_to_lines(entity, units_to_px, delta_px, min_radius_px)
            if segs is None:
                stats["skipped_small"] += 1
            else:
                for p1, p2 in segs:
                    new_lines.append((p1, p2, attribs))
                stats["circle_segments"] += len(segs)

        elif etype == "ARC":
            to_delete.append(entity)
            segs = arc_to_lines(entity, units_to_px, delta_px, min_radius_px)
            if segs is None:
                stats["skipped_small"] += 1
            else:
                for p1, p2 in segs:
                    new_lines.append((p1, p2, attribs))
                stats["arc_segments"] += len(segs)

        elif etype == "ELLIPSE":
            to_delete.append(entity)
            segs = ellipse_to_lines(entity, units_to_px, delta_px, min_radius_px)
            if segs is None:
                stats["skipped_small"] += 1
            else:
                for p1, p2 in segs:
                    new_lines.append((p1, p2, attribs))
                stats["ellipse_segments"] += len(segs)

        else:
            to_delete.append(entity)
            stats["skipped_type"] += 1
            skipped_types[etype] = skipped_types.get(etype, 0) + 1

    for entity in to_delete:
        msp.delete_entity(entity)

    for p1, p2, attribs in new_lines:
        msp.add_line(start=p1, end=p2, dxfattribs=attribs)

    # ── Пост-проверка: гарантируем, что в modelspace остались только LINE ──
    # После конвертации повторно обходим пространство модели и удаляем всё,
    # что не является LINE: штриховки, блоки, размеры, тексты и т.д.
    for entity in list(msp):
        if entity.dxftype() != "LINE":
            t = entity.dxftype()
            post_cleanup_types[t] = post_cleanup_types.get(t, 0) + 1
            stats["post_cleanup"] += 1
            msp.delete_entity(entity)

    tmp = path.with_suffix(".tmp.dxf")
    try:
        doc.saveas(str(tmp))
        tmp.replace(path)
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
    if stats["circle_segments"]:
        print(f"  CIRCLE  -> LINE              : {stats['circle_segments']}")
    if stats["arc_segments"]:
        print(f"  ARC     -> LINE              : {stats['arc_segments']}")
    if stats["ellipse_segments"]:
        print(f"  ELLIPSE -> LINE              : {stats['ellipse_segments']}")
    if stats["skipped_small"]:
        print(f"  Пропущено (r < {min_radius_mm} мм)        : {stats['skipped_small']}")
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
            "Изменяет файлы на месте."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path", type=Path, nargs="?",
        help=(
            "DXF-файл или папка. "
            "По умолчанию: папка sources/ рядом со скриптом. "
            "В папке обрабатываются все *_geometry_only.dxf."
        ),
    )
    parser.add_argument(
        "--units", type=parse_units_arg, default=None, metavar="UNITS",
        help="Единицы чертежа: mm | cm | m | inch | ft или мм/ед. числом.",
    )
    parser.add_argument("--dpi",           type=float, default=300.0)
    parser.add_argument("--min-radius-mm", type=float, default=2.0)
    parser.add_argument("--delta-px",      type=float, default=0.5)
    args = parser.parse_args()

    if args.path is None:
        source_dir = Path(__file__).parent.parent / "sources"
        if not source_dir.is_dir():
            print(f"Ошибка: папка sources/ не найдена: {source_dir}", file=sys.stderr)
            sys.exit(1)
        files = sorted(source_dir.glob("*_geometry_only.dxf"))
    elif args.path.is_dir():
        files = sorted(args.path.glob("*_geometry_only.dxf"))
    elif args.path.is_file():
        files = [args.path]
    else:
        print(f"Ошибка: путь не найден: {args.path}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("Файлы *_geometry_only.dxf не найдены.", file=sys.stderr)
        sys.exit(1)

    print(f"Найдено файлов: {len(files)}")
    for f in files:
        process_file(f, args.dpi, args.min_radius_mm, args.delta_px, args.units)

    print("\nГотово.")


if __name__ == "__main__":
    main()
