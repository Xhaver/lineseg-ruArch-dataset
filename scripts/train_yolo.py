#!/usr/bin/env python3
"""
train_yolo.py — обучение YOLO-модели для маскирования аннотаций
================================================================

Детектирует три класса:
  0 — text             (TEXT, MTEXT, ATTRIB)
  1 — dimension        (DIMENSION, MULTILEADER, LEADER)
  2 — annotation_region (таблицы, штампы, рамки — слои YOLO_TABLE/STAMP/FRAME)

Входные данные: yolo_dataset/ сгенерированный prepare_yolo_dataset.py

Использование
-------------
    # Базовый запуск (YOLO11s, 100 эпох)
    python scripts/train_yolo.py

    # Выбор модели и параметров
    python scripts/train_yolo.py --model yolo11m.pt --epochs 150 --batch 8

    # Дообучение существующих весов
    python scripts/train_yolo.py --resume models/yolo/v1/weights/last.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

SCRIPT_DIR  = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.parent.resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Обучение YOLO для детекции аннотационных объектов на архитектурных чертежах.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    type=str, default="yolo11s.pt",
                   help="Базовая модель: yolo11n/s/m.pt или путь к .pt для дообучения.")
    p.add_argument("--data",     type=Path,
                   default=DATASET_DIR / "yolo_dataset" / "data.yaml",
                   help="Путь к data.yaml YOLO-датасета.")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--patience", type=int,   default=20,
                   help="Early stopping: остановка после N эпох без улучшения.")
    p.add_argument("--batch",    type=int,   default=8,
                   help="Размер батча. RTX 3070 Ti 8GB: 8 при imgsz=1280, 16 при imgsz=640.")
    p.add_argument("--imgsz",    type=int,   default=1280,
                   help="Размер входного изображения (должен совпадать с --crop-size в prepare_yolo_dataset.py).")
    p.add_argument("--device",   type=str,   default="0",
                   help="Устройство: '0' (GPU), 'cpu', '0,1' (multi-GPU).")
    p.add_argument("--workers",  type=int,   default=2,
                   help="DataLoader workers. На Windows >2 даёт overhead; используй 0 если медленно.")
    p.add_argument("--project",  type=Path,
                   default=DATASET_DIR / "models" / "yolo",
                   help="Директория для сохранения результатов.")
    p.add_argument("--name",     type=str,   default="v1",
                   help="Имя эксперимента. Результаты: project/name/")
    p.add_argument("--resume",   type=str,   default=None,
                   help="Путь к last.pt для продолжения обучения.")
    p.add_argument("--save-period", type=int, default=10,
                   help="Сохранять checkpoint каждые N эпох.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ошибка: ultralytics не установлен. pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    if not args.data.exists():
        print(f"Ошибка: data.yaml не найден: {args.data}", file=sys.stderr)
        print("Сначала запустите: python scripts/prepare_yolo_dataset.py --scale 100 --units m",
              file=sys.stderr)
        sys.exit(1)

    # ── Загрузка модели ───────────────────────────────────────────────
    if args.resume:
        print(f"Продолжение обучения с: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Базовая модель: {args.model}")
        model = YOLO(args.model)

    # ── Гиперпараметры для архитектурных чертежей ─────────────────────
    # Ключевые отличия от стандартных настроек:
    #   degrees=5: небольшое вращение — имитация скоса скана.
    #              НЕ 0, т.к. ГОСТ допускает текст от 0° до 90°,
    #              но 90° уже есть в обучающих данных органически.
    #   flipud=0: у чертежа есть явный верх (север/фасад).
    #   perspective=0.0002: минимальная трапецеидальная деформация скана.
    #   hsv_s=0 / hsv_h=0: изображения grayscale — цветовые аугментации бесполезны.
    train_args = dict(
        data        = str(args.data),
        epochs      = args.epochs,
        patience    = args.patience,
        batch       = args.batch,
        imgsz       = args.imgsz,
        device      = args.device,
        workers     = args.workers,
        project     = str(args.project),
        name        = args.name,
        save_period = args.save_period,
        exist_ok    = True,

        # ── Аугментации ──────────────────────────────────────────────
        degrees      = 5.0,    # скос скана; 90° текст приходит из данных органически
        fliplr       = 0.5,    # зеркало — допустимо для планов
        flipud       = 0.0,    # верх/низ чертежа определён
        scale        = 0.3,    # масштаб ±30%
        translate    = 0.1,
        shear        = 2.0,    # небольшой сдвиг — деформация скана
        perspective  = 0.0002, # трапецеидальная деформация сканера
        mosaic       = 0.5,    # объединение 4 кропов
        mixup        = 0.0,    # mixup не нужен для детекции документов
        copy_paste   = 0.0,
        hsv_h        = 0.0,    # grayscale — оттенок не нужен
        hsv_s        = 0.0,    # grayscale — насыщенность не нужна
        hsv_v        = 0.2,    # яркость — имитирует разный уровень сканирования

        # ── Оптимизатор ──────────────────────────────────────────────
        optimizer    = "AdamW",
        lr0          = 0.001,
        lrf          = 0.01,   # финальный lr = lr0 × lrf
        weight_decay = 0.0005,
        warmup_epochs = 3,

        # ── Вывод ─────────────────────────────────────────────────────
        verbose      = True,
        plots        = True,   # кривые обучения, confusion matrix
        save         = True,
        val          = True,
    )

    print(f"\nЗапуск обучения: {args.epochs} эпох, batch={args.batch}, imgsz={args.imgsz}")
    print(f"Результаты: {args.project / args.name}\n")

    results = model.train(**train_args)

    # ── Итог ──────────────────────────────────────────────────────────
    best = args.project / args.name / "weights" / "best.pt"
    print(f"\n{'='*54}")
    print(f"  Обучение завершено.")
    print(f"  Лучшие веса: {best}")
    if best.exists():
        print(f"\nДля использования в Pipeline:")
        print(f"  cfg.mask.model_path = '{best}'")
    print(f"{'='*54}")


if __name__ == "__main__":
    main()
