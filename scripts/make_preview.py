#!/usr/bin/env python3
"""
make_preview.py

Генерирует PNG-превью всех типов и уровней шума.

Входное изображение (в порядке приоритета):
  1. Первый доступный чистый кроп из drawings/*/images/clean/*.tiff
  2. Синтетическое изображение (линии на белом фоне) — не требует предварительной
     нарезки датасета.

Результат сохраняется в docs/noise_preview.png.

Использование:
    python scripts/make_preview.py              # из корня Dataset/
    python scripts/make_preview.py --size 256   # размер ячейки превью
    python scripts/make_preview.py --source path/to/crop.tiff
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import gaussian_filter, rotate as scipy_rotate
except ImportError as e:
    print(f"Ошибка импорта: {e}\nУстановите зависимости: pip install numpy Pillow scipy",
          file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR  = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.parent.resolve()

# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация шума (синхронизирована с add_noise.py)
# ──────────────────────────────────────────────────────────────────────────────

NOISE_CONFIGS: dict[str, dict] = {
    "gaussian_noise": {
        "label":  "Gaussian noise  (σ)",
        "levels": [
            ("L1", {"sigma": 0.05}),
            ("L2", {"sigma": 0.10}),
            ("L3", {"sigma": 0.15}),
            ("L4", {"sigma": 0.25}),
            ("L5", {"sigma": 0.40}),
        ],
    },
    "salt_pepper": {
        "label":  "Salt & pepper  (density)",
        "levels": [
            ("L1", {"density": 0.0001}),
            ("L2", {"density": 0.001}),
            ("L3", {"density": 0.010}),
            ("L4", {"density": 0.040}),
            ("L5", {"density": 0.100}),
        ],
    },
    "gaussian_blur": {
        "label":  "Gaussian blur  (σ px)",
        "levels": [
            ("L1", {"sigma": 0.5}),
            ("L2", {"sigma": 1.0}),
            ("L3", {"sigma": 1.5}),
            ("L4", {"sigma": 2.0}),
            ("L5", {"sigma": 3.0}),
        ],
    },
    "rotation": {
        "label":  "Rotation  (°, CCW)",
        "levels": [
            ("L1", {"angle_deg": 0.5}),
            ("L2", {"angle_deg": 1.0}),
            ("L3", {"angle_deg": 2.0}),
            ("L4", {"angle_deg": 3.0}),
            ("L5", {"angle_deg": 5.0}),
        ],
    },
}

LEVEL_PARAMS: dict[str, dict[str, str]] = {
    "gaussian_noise": {"L1": "σ=0.05", "L2": "σ=0.10", "L3": "σ=0.15",
                       "L4": "σ=0.25", "L5": "σ=0.40"},
    "salt_pepper":    {"L1": "d=0.0001", "L2": "d=0.001", "L3": "d=0.010",
                       "L4": "d=0.040",  "L5": "d=0.100"},
    "gaussian_blur":  {"L1": "σ=0.5px", "L2": "σ=1.0px", "L3": "σ=1.5px",
                       "L4": "σ=2.0px", "L5": "σ=3.0px"},
    "rotation":       {"L1": "0.5°", "L2": "1.0°", "L3": "2.0°",
                       "L4": "3.0°", "L5": "5.0°"},
}


# ──────────────────────────────────────────────────────────────────────────────
# Синтетическое изображение
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic(size: int) -> np.ndarray:
    """
    Создаёт синтетический чертёж: прямоугольники, диагонали, штриховка.
    Возвращает float32 массив [0,1], 0 = линия, 1 = фон.
    """
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    lw = max(1, size // 128)   # толщина линии масштабируется с размером
    m  = size // 8             # отступ

    # Внешний прямоугольник
    draw.rectangle([m, m, size - m, size - m], outline=0, width=lw)

    # Внутренний прямоугольник
    m2 = size // 4
    draw.rectangle([m2, m2, size - m2, size - m2], outline=0, width=lw)

    # Диагонали
    draw.line([(m, m), (size - m, size - m)], fill=0, width=lw)
    draw.line([(size - m, m), (m, size - m)], fill=0, width=lw)

    # Горизонтальные линии (штриховка нижней половины)
    step = size // 10
    for y in range(size // 2, size - m, step):
        draw.line([(m, y), (size - m, y)], fill=0, width=lw)

    # Вертикальные линии (штриховка правой четверти)
    for x in range(size * 3 // 4, size - m, step):
        draw.line([(x, m), (x, size // 2)], fill=0, width=lw)

    return np.array(img, dtype=np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Применение шума
# ──────────────────────────────────────────────────────────────────────────────

def apply_noise(arr: np.ndarray, noise_type: str, params: dict,
                rng: np.random.Generator, edge_blur: float = 1.0) -> np.ndarray:
    if noise_type == "gaussian_noise":
        if edge_blur > 0:
            arr = gaussian_filter(arr, sigma=edge_blur)
        noise = rng.normal(0.0, params["sigma"], arr.shape).astype(np.float32)
        return np.clip(arr + noise, 0.0, 1.0)

    elif noise_type == "salt_pepper":
        result = arr.copy()
        n_total = arr.size
        n_each  = max(1, int(n_total * params["density"] / 2))
        flat    = result.ravel()
        flat[rng.choice(n_total, size=n_each, replace=False)] = 1.0
        flat[rng.choice(n_total, size=n_each, replace=False)] = 0.0
        return flat.reshape(arr.shape)

    elif noise_type == "gaussian_blur":
        return gaussian_filter(arr, sigma=params["sigma"])

    elif noise_type == "rotation":
        return scipy_rotate(arr, angle=params["angle_deg"],
                            reshape=False, mode="constant",
                            cval=1.0, order=1)
    return arr


def to_pil(arr: np.ndarray, noise_type: str) -> Image.Image:
    """Конвертирует float массив в PIL: gaussian_blur → grayscale, остальные → 1-bit."""
    if noise_type == "gaussian_blur":
        uint8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(uint8, mode="L").convert("RGB")
    else:
        binary = (arr > 0.5).astype(np.uint8) * 255
        return Image.fromarray(binary, mode="L").convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Сборка сетки превью
# ──────────────────────────────────────────────────────────────────────────────

def build_grid(
    source: np.ndarray,
    cell:   int,
    pad:    int,
    rng:    np.random.Generator,
) -> Image.Image:
    """
    Строит сетку: строки = типы шума (4), столбцы = clean + L1..L5 (6).
    Заголовки строк и подписи столбцов рисуются средствами PIL.
    """
    n_rows   = len(NOISE_CONFIGS)
    n_cols   = 6                  # clean + L1..L5
    row_h    = cell + pad
    label_w  = 180                # ширина подписи строки
    header_h = 28                 # высота подписи уровня сверху

    total_w = label_w + n_cols * (cell + pad) + pad
    total_h = header_h + n_rows * row_h + pad

    grid = Image.new("RGB", (total_w, total_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(grid)

    # Попытка загрузить шрифт; fallback на дефолтный
    try:
        font_label = ImageFont.truetype("arial.ttf", 13)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_label = ImageFont.load_default()
        font_small = font_label

    # Заголовки столбцов
    col_labels = ["clean", "L1", "L2", "L3", "L4", "L5"]
    for ci, lbl in enumerate(col_labels):
        x = label_w + pad + ci * (cell + pad) + cell // 2
        draw.text((x, 4), lbl, fill=(60, 60, 60), font=font_label, anchor="mt")

    for ri, (noise_type, cfg) in enumerate(NOISE_CONFIGS.items()):
        y_base = header_h + ri * row_h

        # Подпись строки
        draw.text((pad, y_base + cell // 2), cfg["label"],
                  fill=(30, 30, 30), font=font_label, anchor="lm")

        # Чистый кроп (col 0)
        clean_pil = Image.fromarray(
            (source * 255).astype(np.uint8), mode="L"
        ).convert("RGB").resize((cell, cell), Image.LANCZOS)
        grid.paste(clean_pil, (label_w + pad, y_base))

        # Зашумлённые варианты (col 1..5)
        for ci, (level, params) in enumerate(cfg["levels"]):
            noisy = apply_noise(source.copy(), noise_type, params, rng)
            tile  = to_pil(noisy, noise_type).resize((cell, cell), Image.LANCZOS)
            x = label_w + pad + (ci + 1) * (cell + pad)
            grid.paste(tile, (x, y_base))

            # Подпись параметра под тайлом
            param_str = LEVEL_PARAMS[noise_type][level]
            tx = x + cell // 2
            ty = y_base + cell - 2
            draw.text((tx, ty), param_str, fill=(100, 100, 100),
                      font=font_small, anchor="mb")

    return grid


# ──────────────────────────────────────────────────────────────────────────────

def find_clean_crop() -> Path | None:
    drawings = DATASET_DIR / "drawings"
    if drawings.is_dir():
        for tiff in sorted(drawings.rglob("images/clean/*.tiff")):
            return tiff
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генерирует PNG-превью всех типов и уровней шума.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Входной TIFF-кроп. По умолчанию: первый доступный из drawings/, "
             "или синтетическое изображение.",
    )
    parser.add_argument(
        "--size", type=int, default=192,
        help="Размер одной ячейки превью в пикселях.",
    )
    parser.add_argument(
        "--pad", type=int, default=6,
        help="Отступ между ячейками в пикселях.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Путь для сохранения PNG. По умолчанию: docs/noise_preview.png.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Загружаем / генерируем исходный кроп
    if args.source is not None:
        src_path = args.source
        print(f"Источник: {src_path}")
        arr = np.array(Image.open(str(src_path)).convert("L"),
                       dtype=np.float32) / 255.0
    else:
        crop = find_clean_crop()
        if crop is not None:
            print(f"Источник (авто): {crop.relative_to(DATASET_DIR)}")
            arr = np.array(Image.open(str(crop)).convert("L"),
                           dtype=np.float32) / 255.0
        else:
            print("Источник: синтетическое изображение (drawings/ не найден)")
            arr = make_synthetic(512)

    # Выходной путь
    out_path = args.output or (DATASET_DIR / "docs" / "noise_preview.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Сборка сетки превью ...")
    grid = build_grid(arr, cell=args.size, pad=args.pad, rng=rng)
    grid.save(str(out_path), format="PNG", optimize=True)
    print(f"Сохранено: {out_path}  ({grid.width}×{grid.height} px)")


if __name__ == "__main__":
    main()
