#!/usr/bin/env python3
"""
add_noise.py

Генерирует зашумлённые версии чистых кропов.
Читает images/clean/*.tiff из папки чертежа, создаёт images/noisy/{type}_L{n}/.
GT DXF берётся из gt/ и остаётся неизменным (кроме rotation).

Типы шума и параметры:
  gaussian_noise  : Гауссов шум,  sigma in [0,1]
  salt_pepper     : Соль и перец, density = доля пикселей
  gaussian_blur   : Размытие,     sigma в пикселях  (grayscale TIFF)
  rotation        : Поворот,      angle_deg

ВАЖНО для binary-изображений (1-bit):
  Гауссов шум с малым sigma имеет минимальный эффект на чисто-бинарных пикселях.
  --edge-blur (default 1.0) создаёт мягкие переходы вблизи линий перед шумом.
  Установите --edge-blur 0 чтобы отключить.

Структура вывода:
  drawings/{id}/
      images/noisy/
          gaussian_noise_L1/crop_0000.tiff   (1-bit)
          gaussian_blur_L1/crop_0000.tiff    (8-bit grayscale)
          rotation_L1/crop_0000.tiff         (1-bit)
      gt/rotation/
          L1/crop_0000.dxf                   (повёрнутый GT, только для rotation)

Глобальный манифест датасета:
  manifest.json   <- в корне датасета (родитель drawings/), создаётся/обновляется

Примечание:
  gaussian_blur сохраняется как 8-bit grayscale TIFF (не 1-bit), так как
  бинаризация с порогом 0.5 стирает 2px линии при sigma > ~1.1 пкс.
  Остальные типы шума сохраняются как 1-bit TIFF (CCITT Group 4).

Использование:
  python add_noise.py drawings/001
  python add_noise.py drawings/001 --noise gaussian_noise
  python add_noise.py drawings/001 --noise rotation --level L3
  python add_noise.py drawings/001 --edge-blur 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

try:
    from PIL import Image
except ImportError:
    print("Ошибка: Pillow не установлен. pip install Pillow", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.ndimage import gaussian_filter, rotate as scipy_rotate
except ImportError:
    print("Ошибка: scipy не установлен. pip install scipy", file=sys.stderr)
    sys.exit(1)

try:
    import ezdxf
except ImportError:
    print("Ошибка: ezdxf не установлен. pip install ezdxf", file=sys.stderr)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация шума (5 уровней интенсивности, L0 = чистое изображение)
# ──────────────────────────────────────────────────────────────────────────────

NOISE_CONFIGS: dict[str, dict] = {
    "gaussian_noise": {
        "description": "Гауссов шум (sigma в диапазоне [0,1]). "
                       "Перед шумом применяется edge_blur для создания мягких переходов. "
                       "Выход: 1-bit TIFF.",
        "param":       "sigma",
        "levels": {
            "L1": {"sigma": 0.05,  "note": "Едва заметный, качественный скан"},
            "L2": {"sigma": 0.10,  "note": "Лёгкий, хороший офисный сканер"},
            "L3": {"sigma": 0.15,  "note": "Умеренный, старый/бюджетный сканер"},
            "L4": {"sigma": 0.25,  "note": "Сильный, заметно ухудшает восприятие"},
            "L5": {"sigma": 0.40,  "note": "Очень сильный, векторизация затруднена"},
        },
    },
    "salt_pepper": {
        "description": "Импульсный шум 'Соль и перец' (density = доля пикселей)",
        "param":       "density",
        "levels": {
            "L1": {"density": 0.0001, "note": "Редкие точки, почти незаметны"},
            "L2": {"density": 0.001,  "note": "Небольшое кол-во 'битых' пикселей"},
            "L3": {"density": 0.010,  "note": "Заметная 'грязь'"},
            "L4": {"density": 0.040,  "note": "Значительные помехи, грязный сканер"},
            "L5": {"density": 0.100,  "note": "Экстремальный, сильное искажение"},
        },
    },
    "gaussian_blur": {
        "description": "Размытие по Гауссу (sigma в пикселях). "
                       "Сохраняется как 8-bit grayscale TIFF, а не 1-bit, "
                       "поскольку бинаризация стирает тонкие линии при sigma > ~1 пкс.",
        "param":       "sigma",
        "levels": {
            "L1": {"sigma": 0.5, "note": "Едва заметное смягчение контуров"},
            "L2": {"sigma": 1.0, "note": "Лёгкое, небольшая расфокусировка"},
            "L3": {"sigma": 1.5, "note": "Заметное, мелкие детали теряют чёткость"},
            "L4": {"sigma": 2.0, "note": "Сильное, контуры нечёткие"},
            "L5": {"sigma": 3.0, "note": "Очень сильное, мелкие элементы неразличимы"},
        },
    },
    "rotation": {
        "description": "Поворот изображения (angle_deg, CCW)",
        "param":       "angle_deg",
        "levels": {
            "L1": {"angle_deg": 0.5, "note": "Небольшой перекос, ручное сканирование"},
            "L2": {"angle_deg": 1.0, "note": "Лёгкий поворот, заметен при измерении"},
            "L3": {"angle_deg": 2.0, "note": "Умеренный, артефакты при бинаризации"},
            "L4": {"angle_deg": 3.0, "note": "Сильный перекос, небрежное сканирование"},
            "L5": {"angle_deg": 5.0, "note": "Экстремальный, ортогональность нарушена"},
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты: загрузка / сохранение 1-bit TIFF
# ──────────────────────────────────────────────────────────────────────────────

def load_as_float(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Загружает 1-bit TIFF как float32 в [0,1].
    0.0 = чёрный (линия), 1.0 = белый (фон).
    Возвращает (array, dpi).
    """
    img = Image.open(str(path))
    dpi_info = img.info.get("dpi", (300, 300))
    arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    return arr, dpi_info


def save_as_1bit(arr: np.ndarray, path: Path, dpi: tuple[int, int]) -> None:
    """Пороговая бинаризация (threshold=0.5) и сохранение как 1-bit TIFF."""
    binary = (arr > 0.5).astype(np.uint8) * 255
    img = Image.fromarray(binary, mode="L").convert("1")
    try:
        img.save(str(path), format="TIFF", compression="group4", dpi=dpi)
    except Exception:
        img.save(str(path), format="TIFF", dpi=dpi)


def save_as_grayscale(arr: np.ndarray, path: Path, dpi: tuple[int, int]) -> None:
    """Сохранение float [0,1] как 8-bit grayscale TIFF (без бинаризации).
    Используется для gaussian_blur: бинаризация стирает тонкие линии при sigma>1."""
    uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(uint8, mode="L")
    img.save(str(path), format="TIFF", dpi=dpi)


# ──────────────────────────────────────────────────────────────────────────────
# Функции добавления шума
# ──────────────────────────────────────────────────────────────────────────────

def apply_gaussian_noise(
    arr: np.ndarray, sigma: float, edge_blur: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Добавляет Гауссов шум.

    Для бинарных изображений (пиксели = 0 или 1) малые sigma имеют нулевой
    эффект без предварительного размытия краёв. edge_blur создаёт мягкие
    переходы вблизи линий, аналогичные реальному скану.
    """
    if sigma == 0:
        return arr
    if edge_blur > 0:
        arr = gaussian_filter(arr, sigma=edge_blur)
    noise = rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    return np.clip(arr + noise, 0.0, 1.0)


def apply_salt_pepper(
    arr: np.ndarray, density: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Добавляет импульсный шум.
    density = суммарная доля заменяемых пикселей (половина — белые, половина — чёрные).
    """
    if density == 0:
        return arr
    result = arr.copy()
    n_total  = arr.size
    n_noise  = max(1, int(n_total * density))
    n_each   = n_noise // 2

    flat = result.ravel()
    idx_salt   = rng.choice(n_total, size=n_each, replace=False)
    idx_pepper = rng.choice(n_total, size=n_each, replace=False)
    flat[idx_salt]   = 1.0   # соль  (белые точки)
    flat[idx_pepper] = 0.0   # перец (чёрные точки)
    return flat.reshape(arr.shape)


def apply_gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Размытие по Гауссу. sigma — в пикселях."""
    if sigma == 0:
        return arr
    return gaussian_filter(arr, sigma=sigma)


def apply_rotation(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Поворот изображения на angle_deg градусов (CCW при просмотре).
    Зоны за пределами исходного изображения заполняются белым (фон=1.0).
    """
    if angle_deg == 0:
        return arr
    return scipy_rotate(
        arr, angle=angle_deg,
        reshape=False,
        mode="constant",
        cval=1.0,          # белый фон
        order=1,           # билинейная интерполяция
    )


# ──────────────────────────────────────────────────────────────────────────────
# Трансформация GT DXF при повороте
# ──────────────────────────────────────────────────────────────────────────────

def rotate_gt_dxf(
    src_gt: Path,
    dst_gt: Path,
    angle_deg: float,
    image_size: int,
) -> None:
    """
    Создаёт повёрнутый GT DXF для rotation-шума.

    GT DXF хранит координаты в пикселях с Y↑ (DXF-конвенция).
    Поворот изображения CCW на theta соответствует такому же CCW-повороту
    в DXF-пространстве вокруг центра кропа.

    Формулы (стандартный поворот CCW в декартовой системе):
        dx = x - cx,  dy = y - cy
        x' = cx + dx*cos(theta) - dy*sin(theta)
        y' = cy + dx*sin(theta) + dy*cos(theta)
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx = cy = image_size / 2.0

    doc = ezdxf.readfile(str(src_gt))
    out = ezdxf.new("R2010")
    out.header["$INSUNITS"] = 0
    msp = out.modelspace()

    for e in doc.modelspace():
        if e.dxftype() != "LINE":
            continue
        def rot(x: float, y: float) -> tuple[float, float]:
            dx, dy = x - cx, y - cy
            return (cx + dx * cos_t - dy * sin_t,
                    cy + dx * sin_t + dy * cos_t)

        x1r, y1r = rot(e.dxf.start.x, e.dxf.start.y)
        x2r, y2r = rot(e.dxf.end.x,   e.dxf.end.y)
        msp.add_line(start=(x1r, y1r), end=(x2r, y2r))

    out.saveas(str(dst_gt))


# ──────────────────────────────────────────────────────────────────────────────
# Основная функция обработки одного чертежа
# ──────────────────────────────────────────────────────────────────────────────

def process_source(
    drawing_dir:  Path,
    noise_filter: Optional[str],
    level_filter: Optional[str],
    edge_blur:    float,
    seed:         int,
) -> None:
    clean_dir  = drawing_dir / "images" / "clean"
    noisy_dir  = drawing_dir / "images" / "noisy"
    gt_dir     = drawing_dir / "gt"

    tiff_files = sorted(clean_dir.glob("*.tiff"))

    if not tiff_files:
        print(f"  [warn] TIFF-файлы не найдены в {clean_dir}", file=sys.stderr)
        return

    # Размер изображения (нужен для поворота GT)
    sample_img = Image.open(str(tiff_files[0]))
    img_w, img_h = sample_img.size
    image_size = min(img_w, img_h)

    rng = np.random.default_rng(seed)

    # Глобальный манифест лежит в корне датасета (drawings/../manifest.json)
    dataset_root  = drawing_dir.parent.parent
    manifest_path = dataset_root / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {
            "description":   "Vectorization dataset — architectural drawings with noise",
            "noise_configs": NOISE_CONFIGS,
            "samples":       [],
        }

    # drawing_id берём из имени папки (например "001")
    drawing_id = drawing_dir.name

    # Удаляем старые сэмплы этого чертежа (идемпотентный запуск)
    manifest["samples"] = [
        s for s in manifest["samples"]
        if s.get("drawing_id") != drawing_id
    ]

    # Добавляем чистые сэмплы
    for tiff in tiff_files:
        crop_id = tiff.stem
        gt_path = gt_dir / f"{crop_id}.dxf"
        manifest["samples"].append({
            "id":          f"{drawing_id}_{crop_id}_clean",
            "drawing_id":  drawing_id,
            "crop_id":     crop_id,
            "noise_type":  None,
            "noise_level": 0,
            "params":      {},
            "image":       str(tiff.relative_to(dataset_root)).replace("\\", "/"),
            "gt":          str(gt_path.relative_to(dataset_root)).replace("\\", "/")
                           if gt_path.exists() else None,
        })

    existing_ids = {s["id"] for s in manifest["samples"]}

    noise_types = ([noise_filter] if noise_filter
                   else list(NOISE_CONFIGS.keys()))

    for noise_type in noise_types:
        cfg = NOISE_CONFIGS[noise_type]
        levels = ([level_filter] if level_filter
                  else list(cfg["levels"].keys()))

        for level in levels:
            params = cfg["levels"][level]
            out_dir = noisy_dir / f"{noise_type}_{level}"
            out_dir.mkdir(parents=True, exist_ok=True)

            n_created = 0
            for tiff_path in tiff_files:
                crop_id   = tiff_path.stem
                out_tiff  = out_dir / tiff_path.name
                sample_id = f"{drawing_id}_{crop_id}_{noise_type}_{level}"

                arr, dpi_info = load_as_float(tiff_path)

                # Применяем шум
                if noise_type == "gaussian_noise":
                    noisy = apply_gaussian_noise(
                        arr, params["sigma"], edge_blur, rng)
                    save_as_1bit(noisy, out_tiff, dpi_info)
                elif noise_type == "salt_pepper":
                    noisy = apply_salt_pepper(arr, params["density"], rng)
                    save_as_1bit(noisy, out_tiff, dpi_info)
                elif noise_type == "gaussian_blur":
                    noisy = apply_gaussian_blur(arr, params["sigma"])
                    save_as_grayscale(noisy, out_tiff, dpi_info)
                elif noise_type == "rotation":
                    noisy = apply_rotation(arr, params["angle_deg"])
                    save_as_1bit(noisy, out_tiff, dpi_info)
                else:
                    continue
                n_created += 1

                # GT: для rotation создаём повёрнутый GT в gt/rotation/L{n}/;
                #     для остальных — ссылаемся на чистый GT из gt/
                clean_gt = gt_dir / f"{crop_id}.dxf"
                if noise_type == "rotation" and clean_gt.exists():
                    rot_gt_dir = gt_dir / "rotation" / level
                    rot_gt_dir.mkdir(parents=True, exist_ok=True)
                    rot_gt = rot_gt_dir / f"{crop_id}.dxf"
                    rotate_gt_dxf(clean_gt, rot_gt,
                                  params["angle_deg"], image_size)
                    gt_ref = str(rot_gt.relative_to(dataset_root)).replace("\\", "/")
                else:
                    gt_ref = (str(clean_gt.relative_to(dataset_root)).replace("\\", "/")
                              if clean_gt.exists() else None)

                if sample_id not in existing_ids:
                    manifest["samples"].append({
                        "id":          sample_id,
                        "drawing_id":  drawing_id,
                        "crop_id":     crop_id,
                        "noise_type":  noise_type,
                        "noise_level": int(level[1]),    # "L3" -> 3
                        "params":      params,
                        "image":       str(out_tiff.relative_to(dataset_root)).replace("\\", "/"),
                        "gt":          gt_ref,
                    })
                    existing_ids.add(sample_id)

            param_str = ", ".join(f"{k}={v}" for k, v in params.items()
                                  if k != "note")
            print(f"  {noise_type}_{level}  [{param_str}]  -> {n_created} файлов")

    # Сохраняем глобальный манифест
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    n_this = sum(1 for s in manifest["samples"] if s.get("drawing_id") == drawing_id)
    print(f"\n  Манифест: {manifest_path}")
    print(f"  Сэмплов этого чертежа: {n_this}  |  Всего в датасете: {len(manifest['samples'])}")


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генерирует зашумлённые версии чистых кропов.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source_dir", type=Path,
        help="Папка чертежа (например drawings/001). Должна содержать images/clean/ и gt/.",
    )
    parser.add_argument(
        "--noise", default=None,
        choices=list(NOISE_CONFIGS.keys()),
        help="Обрабатывать только этот тип шума (по умолчанию — все).",
    )
    parser.add_argument(
        "--level", default=None,
        choices=["L1", "L2", "L3", "L4", "L5"],
        help="Обрабатывать только этот уровень (по умолчанию — все).",
    )
    parser.add_argument(
        "--edge-blur", type=float, default=1.0,
        help="Sigma предварительного размытия краёв перед Гауссовым шумом. "
             "Создаёт мягкие переходы вблизи линий, аналогичные реальному скану. "
             "0 = отключить (шум малоэффективен на чистых бинарных пикселях).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed генератора случайных чисел для воспроизводимости.",
    )
    args = parser.parse_args()

    clean_check = args.source_dir / "images" / "clean"
    if not clean_check.is_dir():
        print(f"Ошибка: папка images/clean/ не найдена в {args.source_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Чертёж  : {args.source_dir}")
    print(f"Edge-blur: {args.edge_blur}")
    print(f"Seed     : {args.seed}")
    print()

    process_source(
        drawing_dir  = args.source_dir,
        noise_filter = args.noise,
        level_filter = args.level,
        edge_blur    = args.edge_blur,
        seed         = args.seed,
    )

    print("\nГотово.")


if __name__ == "__main__":
    main()
