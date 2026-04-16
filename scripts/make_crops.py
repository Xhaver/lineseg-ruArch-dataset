#!/usr/bin/env python3
"""
make_crops.py

Растеризует DXF (только LINE-примитивы) в 1-бит TIFF и нарезает на кропы.
Для каждого кропа создаёт GT DXF с координатами в пикселях (ось Y — вниз,
начало координат — левый верхний угол кропа).

Ключевые параметры:
    --scale    : знаменатель масштаба чертежа (например, 100 для 1:100)
    --dpi      : разрешение растеризации (по умолчанию 300)
    --crop-size: размер кропа, px (по умолчанию 2048)
    --overlap  : перекрытие кропов, px (по умолчанию 256)
    --dry-run  : только вывести параметры разрешения, не создавать файлы

Пример:
    # Проверить разрешение перед нарезкой
    python make_crops.py sources/1_geometry_only.dxf --scale 100 --dry-run

    # Выполнить нарезку
    python make_crops.py sources/1_geometry_only.dxf --scale 100

Структура вывода:
    drawings/{id:03d}/
        metadata.json
        images/clean/
            crop_0000.tiff
            ...
        gt/
            crop_0000.dxf      <- GT (pixel coords, Y↑)
            ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Ошибка: Pillow не установлен. Выполните: pip install Pillow", file=sys.stderr)
    sys.exit(1)

import ezdxf
from ezdxf.document import Drawing


# ──────────────────────────────────────────────────────────────────────────────
# Единицы (из preprocess_dxf.py)
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
# Геометрия
# ──────────────────────────────────────────────────────────────────────────────

Segment = tuple[float, float, float, float]  # x1, y1, x2, y2


def load_lines(dxf_path: Path) -> list[Segment]:
    """Читает все LINE-сущности из DXF. Возвращает (x1,y1,x2,y2)."""
    doc = ezdxf.readfile(str(dxf_path))
    segs: list[Segment] = []
    for e in doc.modelspace():
        if e.dxftype() == "LINE":
            segs.append((
                e.dxf.start.x, e.dxf.start.y,
                e.dxf.end.x,   e.dxf.end.y,
            ))
    return segs


def bbox(segs: list[Segment]) -> tuple[float, float, float, float]:
    xs = [v for s in segs for v in (s[0], s[2])]
    ys = [v for s in segs for v in (s[1], s[3])]
    return min(xs), min(ys), max(xs), max(ys)


def liang_barsky(
    x1: float, y1: float, x2: float, y2: float,
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> Optional[Segment]:
    """
    Обрезает отрезок прямоугольником. Возвращает None если отрезок вне.
    Алгоритм Liang-Barsky.
    """
    dx, dy = x2 - x1, y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
        elif pi < 0:
            t0 = max(t0, qi / pi)
        else:
            t1 = min(t1, qi / pi)
    if t0 > t1 + 1e-10:
        return None
    return x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy


# ──────────────────────────────────────────────────────────────────────────────
# Основная логика
# ──────────────────────────────────────────────────────────────────────────────

def make_crops(
    dxf_path:     Path,
    units_mm:     float,
    scale:        float,
    dpi:          float,
    crop_size:    int,
    overlap:      int,
    offset:       int,
    line_width:   int,
    min_segments: int,
    output_dir:   Optional[Path],
    dry_run:      bool,
) -> None:

    # ── 1. Загрузка ──────────────────────────────────────────────────────────
    print(f"\nЧертёж: {dxf_path.name}")
    segs = load_lines(dxf_path)
    if not segs:
        print("  [warn] LINE-примитивы не найдены!")
        return

    x_min, y_min, x_max, y_max = bbox(segs)
    w_units = x_max - x_min
    h_units = y_max - y_min
    w_mm = w_units * units_mm
    h_mm = h_units * units_mm

    # ── 2. Масштаб ───────────────────────────────────────────────────────────
    # Физический размер 1 DXF-единицы на бумаге
    paper_mm_per_unit = units_mm / scale        # мм на бумаге за 1 ед. чертежа
    px_per_unit       = paper_mm_per_unit / 25.4 * dpi  # пикселей за 1 ед.

    img_w = max(1, round(w_units * px_per_unit))
    img_h = max(1, round(h_units * px_per_unit))

    # ── 3. Сетка кропов ──────────────────────────────────────────────────────
    out_size = crop_size + 2 * offset  # реальный размер выходного TIFF
    step = crop_size - overlap
    if step <= 0:
        print(f"  Ошибка: overlap ({overlap}) >= crop_size ({crop_size})", file=sys.stderr)
        return

    cols = max(1, -(-img_w // step))  # ceil division
    rows = max(1, -(-img_h // step))
    n_crops = cols * rows

    # ── 4. Информация о разрешении ───────────────────────────────────────────
    crop_mm   = crop_size / dpi * 25.4          # размер кропа в мм на бумаге
    crop_real = crop_mm * scale / 1000          # размер кропа в реальных метрах
    px_per_mm_real = px_per_unit / units_mm * 1000  # px на метр в реальном пространстве

    print()
    print("  Разрешение:")
    print(f"    Масштаб чертежа   : 1:{scale:.0f}")
    print(f"    DPI               : {dpi:.0f}")
    print(f"    px / ед. чертежа  : {px_per_unit:.2f}")
    print(f"    px / метр (реал.) : {px_per_unit / units_mm * 1000:.1f}")
    print(f"    1 px = на бумаге  : {25.4/dpi:.4f} мм")
    print()
    print("  Чертёж:")
    print(f"    Размер DXF        : {w_units:.3f} x {h_units:.3f} ед.")
    print(f"    Размер на бумаге  : {w_mm/units_mm*paper_mm_per_unit:.0f} x {h_mm/units_mm*paper_mm_per_unit:.0f} мм")
    print(f"    Полный растр      : {img_w} x {img_h} px  ({img_w*img_h/1e6:.1f} МПикс)")
    print(f"    Отрезков LINE     : {len(segs)}")
    print()
    print("  Нарезка:")
    print(f"    Размер тайла      : {crop_size} px  (шаг сетки)")
    if offset:
        print(f"    Offset (контекст) : {offset} px с каждой стороны")
        print(f"    Размер TIFF       : {out_size} x {out_size} px")
    else:
        print(f"    Размер TIFF       : {crop_size} x {crop_size} px")
    print(f"    Перекрытие        : {overlap} px  (шаг {step} px)")
    print(f"    Сетка             : {cols} x {rows} = {n_crops} кропов")
    print(f"    Кроп на бумаге    : {crop_mm:.1f} x {crop_mm:.1f} мм")
    print(f"    Кроп реальный     : {crop_real:.2f} x {crop_real:.2f} м")
    print()

    # ── Референс: типовой скан ───────────────────────────────────────────────
    # A1 лист при 300 DPI: ~7016 x 9933 px
    # Кроп 2048px = 2048/7016 * 594 мм ≈ 173 мм (28% ширины листа A1)
    ref_a1_w_px = round(594 / 25.4 * dpi)
    ref_a1_h_px = round(841 / 25.4 * dpi)
    pct_w = crop_size / ref_a1_w_px * 100
    pct_h = crop_size / ref_a1_h_px * 100
    print(f"  Для справки (A1 @ {dpi:.0f} DPI = {ref_a1_w_px}x{ref_a1_h_px} px):")
    print(f"    Кроп занимает     : {pct_w:.1f}% ширины / {pct_h:.1f}% высоты листа")
    print()

    if min_segments > 0:
        print(f"    Мин. сегментов    : {min_segments}  (кропы с меньшим числом пропускаются)")
        print()

    if dry_run:
        print("  [dry-run] Файлы не созданы. Запустите без --dry-run для нарезки.")
        return

    # ── 5. Определяем выходную папку ────────────────────────────────────────
    raw_id = dxf_path.stem.replace("_geometry_only", "")
    try:
        source_id = f"{int(raw_id):03d}"   # "1" -> "001"
    except ValueError:
        source_id = raw_id                  # нечисловые id оставляем как есть

    if output_dir is None:
        output_dir = dxf_path.parent.parent / "drawings" / source_id
    clean_dir = output_dir / "images" / "clean"
    gt_dir    = output_dir / "gt"
    clean_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # ── 6. Растеризация полного чертежа ──────────────────────────────────────
    print(f"  Растеризация {img_w} x {img_h} px ...", end=" ", flush=True)
    img = Image.new("L", (img_w, img_h), color=255)  # белый фон
    draw = ImageDraw.Draw(img)

    for x1, y1, x2, y2 in segs:
        # DXF: Y вверх. Растр: Y вниз.
        px1 = (x1 - x_min) * px_per_unit
        py1 = (y_max - y1) * px_per_unit
        px2 = (x2 - x_min) * px_per_unit
        py2 = (y_max - y2) * px_per_unit
        draw.line([(px1, py1), (px2, py2)], fill=0, width=line_width)

    img_1bit = img.convert("1")
    print("готово")

    # ── 7. Нарезка ───────────────────────────────────────────────────────────
    crops_meta: list[dict] = []
    crop_idx = 0
    skipped   = 0

    print(f"  Нарезка {n_crops} кропов ...")
    for row in range(rows):
        for col in range(cols):
            # Границы тайла (без offset) — используются для сетки и GT-обрезки
            x0 = col * step
            y0 = row * step
            x1 = x0 + crop_size
            y1 = y0 + crop_size

            # Расширенная область с offset — реально вырезаемый регион
            rx0, ry0 = x0 - offset, y0 - offset
            rx1, ry1 = x1 + offset, y1 + offset

            # ── GT DXF (считаем первым, чтобы применить фильтр --min-segments) ──
            # Координатная система GT DXF: пиксели, Y↑ (DXF-конвенция).
            #
            # Преобразование DXF → полный растр (Y↓, image space):
            #   fpx = (x_dxf  - x_min) * px_per_unit
            #   fpy = (y_max  - y_dxf) * px_per_unit    ← Y инвертирован
            #
            # Полный растр → локальные координаты кропа (Y↓):
            #   lx = fpx - rx0
            #   ly = fpy - ry0
            #
            # Локальные (Y↓) → GT DXF (Y↑):
            #   gt_x = lx
            #   gt_y = out_size - ly              ← инверсия Y внутри кропа
            #
            # Итог: gt_y=0 — нижний край кропа, gt_y=out_size — верхний.
            # При открытии в AutoCAD/DXF-вьюере ориентация совпадает с TIFF.
            gt_doc = ezdxf.new("R2010")
            gt_doc.header["$INSUNITS"] = 0  # unitless (пиксели)
            gt_msp = gt_doc.modelspace()
            n_gt = 0

            for sx1, sy1, sx2, sy2 in segs:
                # 1. DXF → полный растр (Y↓)
                fpx1 = (sx1 - x_min) * px_per_unit
                fpy1 = (y_max - sy1) * px_per_unit
                fpx2 = (sx2 - x_min) * px_per_unit
                fpy2 = (y_max - sy2) * px_per_unit
                # 2. Clip по расширенной области (с offset)
                clipped = liang_barsky(fpx1, fpy1, fpx2, fpy2,
                                        rx0, ry0, rx1, ry1)
                if clipped is None:
                    continue
                cx1, cy1, cx2, cy2 = clipped
                # 3. Полный растр → локальные координаты в out_size (Y↓) → GT DXF (Y↑)
                lx1, ly1 = cx1 - rx0, cy1 - ry0
                lx2, ly2 = cx2 - rx0, cy2 - ry0
                gt_msp.add_line(
                    start=(lx1, out_size - ly1),
                    end=(lx2,   out_size - ly2),
                )
                n_gt += 1

            # Фильтр: пропускаем кропы с недостаточным числом сегментов
            if n_gt < min_segments:
                skipped += 1
                continue

            # ── TIFF CCITT Group 4 ────────────────────────────────────────────
            crop_name = f"crop_{crop_idx:04d}"
            tiff_path = clean_dir / f"{crop_name}.tiff"

            region = img_1bit.crop((rx0, ry0, rx1, ry1))
            if region.size != (out_size, out_size):
                padded = Image.new("1", (out_size, out_size), color=1)
                padded.paste(region, (0, 0))
                region = padded

            try:
                region.save(str(tiff_path), format="TIFF",
                            compression="group4", dpi=(int(dpi), int(dpi)))
            except Exception:
                region.save(str(tiff_path), format="TIFF",
                            dpi=(int(dpi), int(dpi)))

            # ── Сохраняем GT ──────────────────────────────────────────────────
            gt_path = gt_dir / f"{crop_name}.dxf"
            gt_doc.saveas(str(gt_path))

            crops_meta.append({
                "id":          crop_name,
                "col":         col,
                "row":         row,
                "x0_px":       x0,
                "y0_px":       y0,
                "x1_px":       x1,
                "y1_px":       y1,
                "n_segments":  n_gt,
            })
            crop_idx += 1

    # ── 8. metadata.json ─────────────────────────────────────────────────────
    meta = {
        "source_file":       dxf_path.name,
        "drawing_id":        source_id,
        "units_mm_per_unit": units_mm,
        "drawing_scale":     scale,
        "dpi":               dpi,
        "px_per_unit":       round(px_per_unit, 4),
        "line_width_px":     line_width,
        "crop_size_px":      crop_size,
        "offset_px":         offset,
        "out_size_px":       out_size,
        "overlap_px":        overlap,
        "mode":              "grid",
        "full_image_px":     [img_w, img_h],
        "bbox_dxf": {
            "x_min": x_min, "y_min": y_min,
            "x_max": x_max, "y_max": y_max,
        },
        "crops": crops_meta,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  Сохранено кропов : {crop_idx}")
    if skipped:
        print(f"  Пропущено (< {min_segments} сегм.): {skipped}")
    print(f"  images/clean/    : {clean_dir}")
    print(f"  gt/              : {gt_dir}")
    print(f"  metadata.json    : {meta_path}")


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Растеризует DXF -> TIFF и нарезает на кропы с GT DXF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dxf", type=Path,
                        help="Входной DXF-файл (geometry_only, только LINE)")
    parser.add_argument(
        "--scale", type=float, required=True,
        help="Знаменатель масштаба чертежа (например, 100 для 1:100, 50 для 1:50)",
    )
    parser.add_argument(
        "--units", type=parse_units_arg, default=None, metavar="UNITS",
        help="Единицы DXF: mm|cm|m|inch|ft или мм/ед. (авто из $INSUNITS если не задано)",
    )
    parser.add_argument("--dpi",        type=float, default=300.0)
    parser.add_argument("--crop-size",  type=int,   default=2048)
    parser.add_argument("--overlap",    type=int,   default=256)
    parser.add_argument("--offset",     type=int,   default=0,
                        help="Контекстный отступ вокруг тайла, px (GT покрывает полный out_size)")
    parser.add_argument("--line-width", type=int,   default=2,
                        help="Толщина линий при растеризации, px")
    parser.add_argument("--min-segments", type=int, default=10,
                        help="Минимальное число LINE-сегментов в GT кропа; "
                             "кропы с меньшим числом пропускаются (0 = отключить)")
    parser.add_argument("--output-dir", type=Path,  default=None,
                        help="Выходная папка (по умолчанию: drawings/{id:03d}/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только показать параметры разрешения, не создавать файлы")
    args = parser.parse_args()

    if not args.dxf.exists():
        print(f"Ошибка: файл не найден: {args.dxf}", file=sys.stderr)
        sys.exit(1)

    # Определяем единицы
    doc = ezdxf.readfile(str(args.dxf))
    units_mm, units_label = resolve_units(doc, args.units)
    print(f"Единицы: {units_label}")

    make_crops(
        dxf_path     = args.dxf,
        units_mm     = units_mm,
        scale        = args.scale,
        dpi          = args.dpi,
        crop_size    = args.crop_size,
        overlap      = args.overlap,
        offset       = args.offset,
        line_width   = args.line_width,
        min_segments = args.min_segments,
        output_dir   = args.output_dir,
        dry_run      = args.dry_run,
    )


if __name__ == "__main__":
    main()
