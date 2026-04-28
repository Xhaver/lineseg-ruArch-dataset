#!/usr/bin/env python3
"""
eval_yolo.py — оценка качества маскирования YOLO
=================================================

Два режима:

1. Валидация на val-датасете (mAP, precision, recall по классам):
    python scripts/eval_yolo.py --model models/yolo/v1/weights/best.pt --mode val

2. Визуальная проверка на конкретных кропах:
    python scripts/eval_yolo.py --model models/yolo/v1/weights/best.pt --mode vis
    python scripts/eval_yolo.py --model ... --mode vis --source yolo_dataset/images/val/
    python scripts/eval_yolo.py --model ... --mode vis --source drawings/001/images/clean/

Выходные PNG сохраняются в --out-dir (по умолчанию eval_yolo_output/).
На каждом изображении показывается:
  — Исходный кроп (grayscale)
  — Детекции YOLO (bbox с меткой класса и confidence)
  — Маскированный результат (регионы, которые скрипт Pipeline заблокирует)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

SCRIPT_DIR  = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.parent.resolve()

# Цвета для каждого класса (BGR для OpenCV)
CLASS_COLORS = {
    0: (255, 100,  50),   # text           — синий
    1: ( 50, 200,  50),   # dimension      — зелёный
    2: ( 50,  50, 255),   # annotation_region — красный
}
CLASS_NAMES = ["text", "dimension", "annotation_region"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Оценка и визуализация качества YOLO-маскирования.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  type=Path, required=True,
                   help="Путь к best.pt или last.pt.")
    p.add_argument("--mode",   type=str, default="vis",
                   choices=["val", "vis", "both"],
                   help="val — метрики на val-сете; vis — визуализация; both — оба.")
    p.add_argument("--data",   type=Path,
                   default=DATASET_DIR / "yolo_dataset" / "data.yaml",
                   help="data.yaml для режима val.")
    p.add_argument("--source", type=Path, default=None,
                   help="Папка с PNG для режима vis. По умолчанию: yolo_dataset/images/val/")
    p.add_argument("--conf",   type=float, default=0.25,
                   help="Порог уверенности детекции.")
    p.add_argument("--iou",    type=float, default=0.45,
                   help="IoU порог для NMS.")
    p.add_argument("--max-vis", type=int, default=20,
                   help="Максимальное число изображений для визуализации.")
    p.add_argument("--imgsz",  type=int,  default=1280)
    p.add_argument("--device", type=str,  default="0")
    p.add_argument("--out-dir", type=Path,
                   default=DATASET_DIR / "eval_yolo_output",
                   help="Директория для сохранения PNG.")
    return p.parse_args()


def run_validation(model, args) -> None:
    """Запуск официальной валидации Ultralytics — mAP, P, R по классам."""
    print("\n=== Валидация на val-датасете ===")
    metrics = model.val(
        data    = str(args.data),
        imgsz   = args.imgsz,
        conf    = args.conf,
        iou     = args.iou,
        device  = args.device,
        verbose = True,
        plots   = True,
        save_json = False,
    )

    print(f"\n{'─'*50}")
    print(f"  mAP@0.50      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.50:0.95 : {metrics.box.map:.4f}")
    print()
    for i, name in enumerate(CLASS_NAMES):
        try:
            p  = metrics.box.p[i]
            r  = metrics.box.r[i]
            f1 = 2*p*r/(p+r) if (p+r) else 0.0
            ap = metrics.box.ap50[i]
            print(f"  [{i}] {name:<20} P={p:.3f}  R={r:.3f}  F1={f1:.3f}  AP50={ap:.3f}")
        except (IndexError, AttributeError):
            pass
    print(f"{'─'*50}")


def run_visualization(model, args) -> None:
    """Визуализация детекций и маски на отдельных изображениях."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Ошибка: opencv-python не установлен. pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    source = args.source or (DATASET_DIR / "yolo_dataset" / "images" / "val")
    if not source.exists():
        print(f"Папка не найдена: {source}", file=sys.stderr)
        sys.exit(1)

    images = sorted(source.glob("*.png"))[:args.max_vis]
    if not images:
        images = sorted(source.glob("*.tiff"))[:args.max_vis]
    if not images:
        print(f"PNG/TIFF изображения не найдены в {source}", file=sys.stderr)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Визуализация ({len(images)} изображений) ===")
    print(f"Выход: {args.out_dir}")

    for img_path in images:
        # Загрузка
        img_raw = cv2.imread(str(img_path))
        if img_raw is None:
            continue
        if img_raw.ndim == 3:
            img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_raw
        h, w = img_gray.shape

        # BGR для отрисовки
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        mask    = np.zeros((h, w), dtype=np.uint8)

        # Инференс
        results = model.predict(
            source  = str(img_path),
            conf    = args.conf,
            iou     = args.iou,
            device  = args.device,
            verbose = False,
            imgsz   = args.imgsz,
        )

        n_det = 0
        for r in results:
            if r.boxes is None:
                continue
            for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                cls_id  = int(cls)
                color   = CLASS_COLORS.get(cls_id, (128, 128, 128))
                label   = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

                # Bbox на изображении
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_bgr, label, (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                # Маска (с padding как в pipeline)
                pad = 5
                mx1 = max(0, x1 - pad); my1 = max(0, y1 - pad)
                mx2 = min(w, x2 + pad); my2 = min(h, y2 + pad)
                mask[my1:my2, mx1:mx2] = 255
                n_det += 1

        # Маскированный результат (белые регионы = замаскировано)
        masked_gray = img_gray.copy()
        masked_gray[mask > 0] = 255
        masked_bgr  = cv2.cvtColor(masked_gray, cv2.COLOR_GRAY2BGR)

        # Сборка финального изображения: [оригинал | детекции | маска]
        separator = np.full((h, 4, 3), 180, dtype=np.uint8)
        combined  = np.hstack([img_bgr, separator, img_bgr if n_det == 0 else
                                _blend_detections(img_bgr.copy(), mask), separator, masked_bgr])

        # Подпись
        label_bar = np.full((28, combined.shape[1], 3), 40, dtype=np.uint8)
        cv2.putText(label_bar, f"{img_path.name}   |  детекций: {n_det}  |  conf>={args.conf}",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        combined = np.vstack([label_bar, combined])

        out_path = args.out_dir / f"vis_{img_path.stem}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  {img_path.name:<55} → {n_det} детекций")

    print(f"\nГотово. Сохранено {len(images)} изображений в {args.out_dir}/")


def _blend_detections(img_bgr, mask):
    """Полупрозрачная заливка маскированных регионов."""
    import numpy as np
    overlay = img_bgr.copy()
    overlay[mask > 0] = (200, 180, 255)   # светло-фиолетовый = замаскировано
    return img_bgr


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ошибка: ultralytics не установлен. pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    if not args.model.exists():
        print(f"Модель не найдена: {args.model}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(args.model))
    print(f"Модель загружена: {args.model}")

    if args.mode in ("val", "both"):
        run_validation(model, args)

    if args.mode in ("vis", "both"):
        run_visualization(model, args)


if __name__ == "__main__":
    main()
