#!/usr/bin/env python3
"""
extract_geometry.py

Извлекает геометрические сущности из *_full_annotated.dxf и сохраняет
в sources/unprocessed_geometry_only/ как *_geometry_only.dxf.

Сохраняет: LINE, LWPOLYLINE, ARC, CIRCLE, ELLIPSE, SPLINE, INSERT
Удаляет  : TEXT, MTEXT, DIMENSION, LEADER, MULTILEADER, HATCH, POINT,
            а также всё на слоях-аннотациях (YOLO_TABLE, YOLO_STAMP)

После этого запусти preprocess_dxf.py --units m --skip-curved,
чтобы привести геометрию к LINE-only без дуговых аннотаций.

Использование:
    # Все чертежи из sources/full_annotated/
    python scripts/extract_geometry.py

    # Конкретный файл
    python scripts/extract_geometry.py sources/full_annotated/5_full_annotated.dxf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import ezdxf

from paths import FULL_ANNOTATED_DIR, UNPROCESSED_GEOMETRY_DIR

# Слои с аннотациями YOLO — всё на них удаляется
ANNOTATION_LAYERS = {"YOLO_TABLE", "YOLO_STAMP"}

# Типы сущностей, которые точно не являются геометрией чертежа
NON_GEOMETRY_TYPES = {
    "TEXT", "MTEXT", "DIMENSION", "LEADER", "MULTILEADER",
    "HATCH", "SOLID", "POINT", "SHAPE",
    "VIEWPORT", "ATTDEF", "ATTRIB",
    "ACAD_TABLE", "ACDBPLACEHOLDER",
}

# Типы сущностей, которые сохраняем (геометрия)
GEOMETRY_TYPES = {
    "LINE", "LWPOLYLINE", "ARC", "CIRCLE", "ELLIPSE",
    "SPLINE", "INSERT", "POLYLINE", "3DFACE",
}


def extract(src_path: Path, out_path: Path) -> None:
    print(f"\n[{src_path.name}]  ->  {out_path.name}")
    doc = ezdxf.readfile(str(src_path))
    msp = doc.modelspace()

    to_delete = []
    counts: dict[str, int] = {}

    for entity in msp:
        etype = entity.dxftype()
        layer = entity.dxf.layer if entity.dxf.hasattr("layer") else "0"

        # Удаляем аннотационные слои
        if layer in ANNOTATION_LAYERS:
            to_delete.append(entity)
            counts[f"layer:{layer}"] = counts.get(f"layer:{layer}", 0) + 1
            continue

        # Удаляем нежелательные типы
        if etype in NON_GEOMETRY_TYPES:
            to_delete.append(entity)
            counts[etype] = counts.get(etype, 0) + 1
            continue

        # Всё остальное оставляем (включая GEOMETRY_TYPES и неизвестные)

    for e in to_delete:
        msp.delete_entity(e)

    # Статистика оставшегося
    remaining: dict[str, int] = {}
    for e in msp:
        t = e.dxftype()
        remaining[t] = remaining.get(t, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.dxf")
    try:
        doc.saveas(str(tmp))
        tmp.replace(out_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    print(f"  Удалено:")
    for t, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {cnt}")
    print(f"  Сохранено:")
    for t, cnt in sorted(remaining.items(), key=lambda x: -x[1]):
        print(f"    {t}: {cnt}")
    total_geom = sum(remaining.values())
    print(f"  Итого геометрических сущностей: {total_geom}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Извлекает геометрию из *_full_annotated.dxf -> unprocessed_geometry_only/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path", type=Path, nargs="?",
        help="DXF-файл или папка. По умолчанию: sources/full_annotated/",
    )
    args = parser.parse_args()

    if args.path is None:
        src_dir = FULL_ANNOTATED_DIR
        if not src_dir.is_dir():
            print(f"Ошибка: папка не найдена: {src_dir}", file=sys.stderr)
            sys.exit(1)
        files = sorted(src_dir.glob("*_full_annotated.dxf"))
    elif args.path.is_dir():
        files = sorted(args.path.glob("*_full_annotated.dxf"))
    elif args.path.is_file():
        files = [args.path]
    else:
        print(f"Ошибка: путь не найден: {args.path}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("Файлы *_full_annotated.dxf не найдены.", file=sys.stderr)
        sys.exit(1)

    UNPROCESSED_GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Входная папка : {files[0].parent}")
    print(f"Выходная папка: {UNPROCESSED_GEOMETRY_DIR}")
    print(f"Файлов: {len(files)}")

    for f in files:
        out_name = f.name.replace("_full_annotated", "_geometry_only")
        extract(f, UNPROCESSED_GEOMETRY_DIR / out_name)

    print("\nГотово. Следующий шаг:")
    print("  python scripts/preprocess_dxf.py --units m")


if __name__ == "__main__":
    main()
