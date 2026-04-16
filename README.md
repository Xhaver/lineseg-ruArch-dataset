# Vectorization Dataset — Architectural Drawings

> **[English](#english) | [Русский](#русский)**

---

<a name="english"></a>

## English

**Navigation:**
[Dataset structure](#dataset-structure) ·
[Noise types](#noise-types-and-levels) ·
[Requirements](#requirements) ·
[Quick start](#quick-start) ·
[Scripts reference](#scripts-reference) ·
[manifest.json](#manifestjson-format) ·
[Adding a drawing](#adding-a-new-drawing)

Synthetic dataset for research on the influence of line segment clustering algorithms
(DBSCAN, K-Means, MeanShift) on hybrid vectorization accuracy at various noise levels.

Each sample consists of a rasterized architectural drawing crop (TIFF) and a
ground-truth DXF file containing the exact LINE segments in pixel coordinates.

---

### Dataset structure

```
Dataset/
├── sources/                        # Preprocessed DXF source files
│   └── 1_geometry_only.dxf
│
├── drawings/
│   └── 001/                        # One folder per drawing (zero-padded)
│       ├── metadata.json           # Rasterization parameters and crop grid info
│       ├── images/
│       │   ├── clean/              # Clean rasterized crops (1-bit TIFF)
│       │   │   ├── crop_0000.tiff
│       │   │   └── ...
│       │   └── noisy/              # Noisy variants
│       │       ├── gaussian_noise_L1/   # 1-bit TIFF
│       │       ├── gaussian_noise_L2/
│       │       ├── ...
│       │       ├── salt_pepper_L1/      # 1-bit TIFF
│       │       ├── ...
│       │       ├── gaussian_blur_L1/    # 8-bit grayscale TIFF *
│       │       ├── ...
│       │       ├── rotation_L1/         # 1-bit TIFF
│       │       └── ...
│       └── gt/
│           ├── crop_0000.dxf       # Ground truth — pixel coords, Y-up
│           └── rotation/           # Rotated GT (only for rotation noise)
│               ├── L1/
│               │   └── crop_0000.dxf
│               └── ...
│
├── manifest.json                   # Global dataset manifest (all drawings)
└── scripts/
    ├── run.py                      # Interactive menu (entry point)
    ├── preprocess_dxf.py
    ├── make_crops.py
    └── add_noise.py
```

> \* `gaussian_blur` crops are saved as 8-bit grayscale because binary thresholding
> at 0.5 erases 2 px lines at σ > ~1.1 px. All other noise types produce 1-bit TIFF.

---

### Noise types and levels

| Type | Parameter | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| `gaussian_noise` | sigma | 0.05 | 0.10 | 0.15 | 0.25 | 0.40 |
| `salt_pepper` | density | 0.0001 | 0.001 | 0.010 | 0.040 | 0.100 |
| `gaussian_blur` | sigma (px) | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
| `rotation` | angle (deg, CCW) | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 |

For `rotation` the ground truth DXF is also rotated and stored in `gt/rotation/L{n}/`.
All other noise types share the clean GT from `gt/`.

Per drawing: 6 crops × (1 clean + 20 noisy) = **126 samples**.

---

### Requirements

```bash
pip install ezdxf Pillow numpy scipy
```

Python 3.10+.

---

### Quick start

```bash
cd Dataset/
python scripts/run.py
```

The interactive menu guides through all pipeline steps.

---

### Scripts reference

#### `run.py` — interactive menu

```bash
python scripts/run.py
```

Menu options:

| # | Option |
|---|---|
| 1 | Preprocess DXF — convert all primitives to LINE segments |
| 2 | Rasterize & crop one drawing |
| 3 | Generate noisy variants for one drawing |
| 4 | Full pipeline (steps 1–3) for one drawing |
| 5 | Dataset statistics |

---

#### `preprocess_dxf.py` — DXF preprocessing

Converts all geometry primitives to individual LINE entities **in-place**:
`LWPOLYLINE` → segments (including bulge arcs), `CIRCLE` / `ARC` / `ELLIPSE` → chord approximation.

```bash
# Process all *_geometry_only.dxf in sources/ (default)
python scripts/preprocess_dxf.py --units m

# Specific folder or file
python scripts/preprocess_dxf.py path/to/sources/ --units m
python scripts/preprocess_dxf.py path/to/file.dxf --units m
```

| Argument | Default | Description |
|---|---|---|
| `path` | `sources/` | File or folder to process |
| `--units` | from `$INSUNITS` | Drawing units: `mm`, `cm`, `m`, `inch`, `ft`, or a number (mm/unit) |
| `--dpi` | `300` | Target DPI for arc approximation tolerance |
| `--min-radius-mm` | `1.0` | Circles/arcs smaller than this (mm on paper) are skipped |
| `--delta-px` | `0.5` | Max chord deviation in pixels for arc approximation |

> **Note:** the `$INSUNITS` header in many CAD files is incorrect.
> Always specify `--units` explicitly if you know the coordinate units.

---

#### `make_crops.py` — rasterization and cropping

Rasterizes a preprocessed DXF to a 1-bit TIFF and slices it into a grid of crops.
For each crop a GT DXF is generated with line coordinates in pixels (Y-up, DXF convention).

```bash
# Dry-run: show resolution info without creating files
python scripts/make_crops.py sources/1_geometry_only.dxf --scale 100 --units m --dry-run

# Rasterize and crop
python scripts/make_crops.py sources/1_geometry_only.dxf --scale 100 --units m
```

| Argument | Default | Description |
|---|---|---|
| `dxf` | required | Path to the source DXF file |
| `--scale` | required | Drawing scale denominator (e.g. `100` for 1:100) |
| `--units` | from `$INSUNITS` | Drawing units (same as `preprocess_dxf.py`) |
| `--dpi` | `300` | Rasterization DPI |
| `--crop-size` | `2048` | Grid tile size in pixels (step basis) |
| `--overlap` | `256` | Tile overlap in pixels; step = crop_size − overlap |
| `--offset` | `0` | Context padding around each tile (px); output TIFF = crop_size + 2×offset |
| `--line-width` | `2` | Line width in pixels during rasterization |
| `--output-dir` | `drawings/{id:03d}/` | Override output directory |
| `--dry-run` | off | Print resolution info only, do not create files |

**Output** is written to `drawings/{id:03d}/`:

```
drawings/001/
    metadata.json
    images/clean/crop_0000.tiff  ...
    gt/crop_0000.dxf  ...
```

**GT DXF coordinate system:** pixel coordinates, origin at the top-left corner of the
crop, Y-axis pointing up (DXF convention). `y = 0` is the bottom edge,
`y = out_size` is the top edge.

---

#### `add_noise.py` — noise generation

Generates all noisy variants of the clean crops for one drawing.
Reads from `drawings/{id}/images/clean/`, writes to `drawings/{id}/images/noisy/`.
Updates the global `manifest.json` (idempotent — reruns replace old entries for the drawing).

```bash
# All noise types and levels
python scripts/add_noise.py drawings/001

# One noise type only
python scripts/add_noise.py drawings/001 --noise gaussian_blur

# One noise type, one level
python scripts/add_noise.py drawings/001 --noise rotation --level L3

# Disable edge-blur pre-softening
python scripts/add_noise.py drawings/001 --edge-blur 0
```

| Argument | Default | Description |
|---|---|---|
| `drawing_dir` | required | Path to drawing folder, e.g. `drawings/001` |
| `--noise` | all | Process only this noise type |
| `--level` | all | Process only this level (`L1`–`L5`) |
| `--edge-blur` | `1.0` | Gaussian pre-softening σ before `gaussian_noise` (simulates scan digitization) |
| `--seed` | `42` | Random seed for reproducibility |

---

### manifest.json format

The global manifest accumulates entries from all drawings.
Each entry in `samples` describes one (image, GT) pair:

```json
{
  "id":           "001_crop_0000_gaussian_noise_L3",
  "drawing_id":   "001",
  "crop_id":      "crop_0000",
  "noise_type":   "gaussian_noise",
  "noise_level":  3,
  "params":       { "sigma": 0.15 },
  "image":        "drawings/001/images/noisy/gaussian_noise_L3/crop_0000.tiff",
  "gt":           "drawings/001/gt/crop_0000.dxf"
}
```

For clean samples `noise_type` is `null` and `noise_level` is `0`.
For `rotation` samples `gt` points to the rotated GT in `gt/rotation/L{n}/`.
All paths are relative to the `Dataset/` root and use forward slashes.

---

### Adding a new drawing

1. Place the converted DXF as `sources/N_geometry_only.dxf`.
2. Run the full pipeline:

```bash
python scripts/preprocess_dxf.py sources/N_geometry_only.dxf --units m
python scripts/make_crops.py     sources/N_geometry_only.dxf --scale 100 --units m
python scripts/add_noise.py      drawings/00N
```

The drawing is assigned ID `00N` automatically and its samples are appended to `manifest.json`.

---
---

<a name="русский"></a>

## Русский

**Навигация:**
[Структура датасета](#структура-датасета) ·
[Типы шума](#типы-шума-и-уровни) ·
[Зависимости](#зависимости) ·
[Быстрый старт](#быстрый-старт) ·
[Справочник по скриптам](#справочник-по-скриптам) ·
[Формат manifest.json](#формат-manifestjson) ·
[Добавление чертежа](#добавление-нового-чертежа)

Синтетический датасет для исследования влияния алгоритмов кластеризации отрезков
(DBSCAN, K-Means, MeanShift) на точность гибридной векторизации архитектурных чертежей
при различных уровнях шума.

Каждый сэмпл — растровый кроп архитектурного чертежа (TIFF) и GT DXF с точными
LINE-сегментами в пиксельных координатах.

---

### Структура датасета

```
Dataset/
├── sources/                        # Препроцессированные DXF-источники
│   └── 1_geometry_only.dxf
│
├── drawings/
│   └── 001/                        # Папка каждого чертежа (нулевой padding)
│       ├── metadata.json           # Параметры растеризации и сетки кропов
│       ├── images/
│       │   ├── clean/              # Чистые кропы (1-bit TIFF)
│       │   │   ├── crop_0000.tiff
│       │   │   └── ...
│       │   └── noisy/              # Зашумлённые варианты
│       │       ├── gaussian_noise_L1/   # 1-bit TIFF
│       │       ├── gaussian_noise_L2/
│       │       ├── ...
│       │       ├── salt_pepper_L1/      # 1-bit TIFF
│       │       ├── ...
│       │       ├── gaussian_blur_L1/    # 8-bit grayscale TIFF *
│       │       ├── ...
│       │       ├── rotation_L1/         # 1-bit TIFF
│       │       └── ...
│       └── gt/
│           ├── crop_0000.dxf       # Ground truth — пиксельные координаты, Y↑
│           └── rotation/           # Повёрнутые GT (только для rotation)
│               ├── L1/
│               │   └── crop_0000.dxf
│               └── ...
│
├── manifest.json                   # Глобальный манифест (все чертежи)
└── scripts/
    ├── run.py                      # Интерактивное меню (точка входа)
    ├── preprocess_dxf.py
    ├── make_crops.py
    └── add_noise.py
```

> \* Кропы `gaussian_blur` сохраняются как 8-bit grayscale: бинаризация с порогом 0.5
> стирает линии толщиной 2 px при σ > ~1.1 px. Все остальные типы шума — 1-bit TIFF.

---

### Типы шума и уровни

| Тип | Параметр | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| `gaussian_noise` | sigma | 0.05 | 0.10 | 0.15 | 0.25 | 0.40 |
| `salt_pepper` | density | 0.0001 | 0.001 | 0.010 | 0.040 | 0.100 |
| `gaussian_blur` | sigma (px) | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
| `rotation` | угол (град, CCW) | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 |

Для `rotation` GT DXF также поворачивается и хранится в `gt/rotation/L{n}/`.
Все остальные типы шума используют один GT из `gt/`.

На каждый чертёж: 6 кропов × (1 чистый + 20 зашумлённых) = **126 сэмплов**.

---

### Зависимости

```bash
pip install ezdxf Pillow numpy scipy
```

Python 3.10+.

---

### Быстрый старт

```bash
cd Dataset/
python scripts/run.py
```

Интерактивное меню проведёт через все шаги пайплайна.

---

### Справочник по скриптам

#### `run.py` — интерактивное меню

```bash
python scripts/run.py
```

Пункты меню:

| # | Действие |
|---|---|
| 1 | Препроцессировать DXF — привести все примитивы к LINE-сегментам |
| 2 | Нарезать чертёж на кропы |
| 3 | Сгенерировать шумовые варианты для одного чертежа |
| 4 | Полный пайплайн (шаги 1–3) для одного чертежа |
| 5 | Статистика датасета |

---

#### `preprocess_dxf.py` — препроцессирование DXF

Приводит все геометрические примитивы к отдельным LINE-сущностям **in-place**:
`LWPOLYLINE` → отрезки (включая bulge-дуги), `CIRCLE` / `ARC` / `ELLIPSE` → хордовая аппроксимация.

```bash
# Все *_geometry_only.dxf в sources/ (по умолчанию)
python scripts/preprocess_dxf.py --units m

# Конкретная папка или файл
python scripts/preprocess_dxf.py path/to/sources/ --units m
python scripts/preprocess_dxf.py path/to/file.dxf --units m
```

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `path` | `sources/` | Файл или папка для обработки |
| `--units` | из `$INSUNITS` | Единицы чертежа: `mm`, `cm`, `m`, `inch`, `ft` или число (мм/ед.) |
| `--dpi` | `300` | DPI для расчёта точности аппроксимации дуг |
| `--min-radius-mm` | `1.0` | Окружности/дуги меньше этого значения (мм на бумаге) пропускаются |
| `--delta-px` | `0.5` | Максимальное отклонение хорды от дуги в пикселях |

> **Важно:** заголовок `$INSUNITS` во многих CAD-файлах заполнен некорректно.
> Всегда указывайте `--units` явно, если единицы чертежа известны.

---

#### `make_crops.py` — растеризация и нарезка

Растеризует препроцессированный DXF в 1-bit TIFF и нарезает по сетке.
Для каждого кропа создаётся GT DXF с координатами линий в пикселях (Y↑, DXF-конвенция).

```bash
# Dry-run: показать параметры без создания файлов
python scripts/make_crops.py sources/1_geometry_only.dxf --scale 100 --units m --dry-run

# Растеризация и нарезка
python scripts/make_crops.py sources/1_geometry_only.dxf --scale 100 --units m
```

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `dxf` | обязательный | Путь к исходному DXF-файлу |
| `--scale` | обязательный | Знаменатель масштаба (например `100` для 1:100) |
| `--units` | из `$INSUNITS` | Единицы чертежа (аналогично `preprocess_dxf.py`) |
| `--dpi` | `300` | DPI растеризации |
| `--crop-size` | `2048` | Размер тайла в пикселях (основа шага сетки) |
| `--overlap` | `256` | Перекрытие тайлов в пикселях; шаг = crop_size − overlap |
| `--offset` | `0` | Контекстный отступ вокруг тайла (px); итоговый TIFF = crop_size + 2×offset |
| `--line-width` | `2` | Толщина линий при растеризации (px) |
| `--output-dir` | `drawings/{id:03d}/` | Переопределить выходную папку |
| `--dry-run` | выкл. | Только показать параметры, не создавать файлы |

**Результат** записывается в `drawings/{id:03d}/`:

```
drawings/001/
    metadata.json
    images/clean/crop_0000.tiff  ...
    gt/crop_0000.dxf  ...
```

**Система координат GT DXF:** пиксельные координаты, начало — левый верхний угол
кропа, ось Y направлена вверх (DXF-конвенция). `y = 0` — нижний край, `y = out_size` — верхний.

---

#### `add_noise.py` — генерация шума

Создаёт все зашумлённые варианты чистых кропов для одного чертежа.
Читает из `drawings/{id}/images/clean/`, записывает в `drawings/{id}/images/noisy/`.
Обновляет глобальный `manifest.json` (идемпотентно — при повторном запуске старые записи заменяются).

```bash
# Все типы и уровни шума
python scripts/add_noise.py drawings/001

# Только один тип шума
python scripts/add_noise.py drawings/001 --noise gaussian_blur

# Один тип и один уровень
python scripts/add_noise.py drawings/001 --noise rotation --level L3

# Отключить предварительное размытие краёв
python scripts/add_noise.py drawings/001 --edge-blur 0
```

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `drawing_dir` | обязательный | Папка чертежа, например `drawings/001` |
| `--noise` | все | Обрабатывать только этот тип шума |
| `--level` | все | Обрабатывать только этот уровень (`L1`–`L5`) |
| `--edge-blur` | `1.0` | Sigma предварительного размытия краёв перед `gaussian_noise` (имитирует оцифровку скана) |
| `--seed` | `42` | Seed генератора случайных чисел для воспроизводимости |

---

### Формат manifest.json

Глобальный манифест накапливает записи по всем чертежам.
Каждая запись в `samples` описывает одну пару (изображение, GT):

```json
{
  "id":           "001_crop_0000_gaussian_noise_L3",
  "drawing_id":   "001",
  "crop_id":      "crop_0000",
  "noise_type":   "gaussian_noise",
  "noise_level":  3,
  "params":       { "sigma": 0.15 },
  "image":        "drawings/001/images/noisy/gaussian_noise_L3/crop_0000.tiff",
  "gt":           "drawings/001/gt/crop_0000.dxf"
}
```

У чистых сэмплов `noise_type` равен `null`, `noise_level` равен `0`.
У сэмплов `rotation` поле `gt` ссылается на повёрнутый GT в `gt/rotation/L{n}/`.
Все пути относительны корня `Dataset/` и используют прямые слэши.

---

### Добавление нового чертежа

1. Разместите конвертированный DXF как `sources/N_geometry_only.dxf`.
2. Запустите полный пайплайн:

```bash
python scripts/preprocess_dxf.py sources/N_geometry_only.dxf --units m
python scripts/make_crops.py     sources/N_geometry_only.dxf --scale 100 --units m
python scripts/add_noise.py      drawings/00N
```

Чертёж автоматически получит ID `00N`, его сэмплы будут добавлены в `manifest.json`.
