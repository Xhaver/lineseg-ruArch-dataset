#!/usr/bin/env python3
"""
run.py  —  Интерактивное меню для работы с датасетом.

Запуск:
    python scripts/run.py          # из корня Dataset/
    python run.py                  # из папки scripts/
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# Корень датасета: папка, где лежат sources/, drawings/, scripts/
SCRIPT_DIR  = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.parent.resolve()
PYTHON      = sys.executable


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────

def run(cmd: list[str], pause: bool = True) -> None:
    """Запускает команду и ждёт завершения."""
    print()
    print("$ " + " ".join(str(c) for c in cmd))
    print()
    subprocess.run(cmd, cwd=DATASET_DIR)
    print()
    if pause:
        input("Нажмите Enter для возврата в меню...")


def ask(prompt: str, default: str = "") -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def list_drawings() -> list[Path]:
    drawings_dir = DATASET_DIR / "drawings"
    if not drawings_dir.exists():
        return []
    return sorted(drawings_dir.iterdir())


def list_sources() -> list[Path]:
    sources_dir = DATASET_DIR / "sources"
    if not sources_dir.exists():
        return []
    return sorted(sources_dir.glob("*_geometry_only.dxf"))


def choose_drawing(prompt: str = "Выберите чертёж") -> Path | None:
    drawings = list_drawings()
    if not drawings:
        print("  [!] Папка drawings/ пуста. Сначала выполните нарезку.")
        return None
    print(f"\n{prompt}:")
    for i, d in enumerate(drawings, 1):
        n_samples = _count_samples(d.name)
        print(f"  {i}. {d.name}   ({n_samples} сэмплов в манифесте)")
    idx = ask("Номер", "1")
    try:
        return drawings[int(idx) - 1]
    except (ValueError, IndexError):
        print("  [!] Неверный выбор.")
        return None


def choose_source(prompt: str = "Выберите исходный DXF") -> Path | None:
    sources = list_sources()
    if not sources:
        print("  [!] Папка sources/ пуста или нет *_geometry_only.dxf.")
        return None
    print(f"\n{prompt}:")
    for i, s in enumerate(sources, 1):
        print(f"  {i}. {s.name}")
    idx = ask("Номер", "1")
    try:
        return sources[int(idx) - 1]
    except (ValueError, IndexError):
        print("  [!] Неверный выбор.")
        return None


def _count_samples(drawing_id: str) -> int:
    manifest = DATASET_DIR / "manifest.json"
    if not manifest.exists():
        return 0
    with open(manifest, encoding="utf-8") as f:
        data = json.load(f)
    return sum(1 for s in data.get("samples", [])
               if s.get("drawing_id") == drawing_id)


# ──────────────────────────────────────────────────────────────────────────────
# Пункты меню
# ──────────────────────────────────────────────────────────────────────────────

def menu_preprocess() -> None:
    """1. Препроцессировать DXF (preprocess_dxf.py)"""
    print("\n=== Препроцессирование DXF ===")
    print("Разбивает все примитивы (LWPOLYLINE, CIRCLE, ARC, ELLIPSE)")
    print("на отдельные LINE-сегменты. Изменяет файл источника IN-PLACE.")
    print()
    sources = list_sources()
    if not sources:
        print("  [!] Нет файлов *_geometry_only.dxf в sources/.")
        input("\nEnter для возврата...")
        return
    print("Будут обработаны файлы:")
    for s in sources:
        print(f"  - {s.name}")

    units = ask("\nЕдиницы чертежа (mm/cm/m/inch/ft)", "m")
    dpi   = ask("DPI для оценки точности аппроксимации дуг", "300")

    run([PYTHON, SCRIPT_DIR / "preprocess_dxf.py",
         DATASET_DIR / "sources",
         "--units", units,
         "--dpi", dpi])


def menu_make_crops() -> None:
    """2. Нарезать чертёж на кропы (make_crops.py)"""
    print("\n=== Нарезка чертежа на кропы ===")
    src = choose_source()
    if src is None:
        input("\nEnter для возврата...")
        return

    scale     = ask("Масштаб (знаменатель, например 100 для 1:100)", "100")
    units     = ask("Единицы чертежа", "m")
    dpi       = ask("DPI", "300")
    crop_size = ask("Размер тайла, px", "2048")
    overlap   = ask("Перекрытие, px", "256")
    offset    = ask("Offset (контекстный отступ), px", "0")
    min_seg   = ask("Минимум сегментов в кропе (0 = не фильтровать)", "10")

    dry = ask("\nСначала dry-run (показать параметры без создания файлов)? [y/n]", "y")
    if dry.lower() == "y":
        run([PYTHON, SCRIPT_DIR / "make_crops.py", src,
             "--scale", scale, "--units", units,
             "--dpi", dpi, "--crop-size", crop_size,
             "--overlap", overlap, "--offset", offset,
             "--min-segments", min_seg,
             "--dry-run"])
        confirm = ask("Продолжить и создать файлы? [y/n]", "y")
        if confirm.lower() != "y":
            return

    run([PYTHON, SCRIPT_DIR / "make_crops.py", src,
         "--scale", scale, "--units", units,
         "--dpi", dpi, "--crop-size", crop_size,
         "--overlap", overlap, "--offset", offset,
         "--min-segments", min_seg])


def menu_add_noise() -> None:
    """3. Сгенерировать шум (add_noise.py)"""
    print("\n=== Генерация шумовых вариантов ===")
    drawing = choose_drawing()
    if drawing is None:
        input("\nEnter для возврата...")
        return

    noise_types = ["gaussian_noise", "salt_pepper", "gaussian_blur", "rotation"]
    print("\nТип шума:")
    print("  0. Все типы")
    for i, t in enumerate(noise_types, 1):
        print(f"  {i}. {t}")
    noise_choice = ask("Выбор", "0")

    noise_arg: list[str] = []
    if noise_choice != "0":
        try:
            noise_arg = ["--noise", noise_types[int(noise_choice) - 1]]
        except (ValueError, IndexError):
            print("  [!] Неверный выбор, будут обработаны все типы.")

    levels = ["L1", "L2", "L3", "L4", "L5"]
    print("\nУровень шума:")
    print("  0. Все уровни")
    for i, lv in enumerate(levels, 1):
        print(f"  {i}. {lv}")
    level_choice = ask("Выбор", "0")

    level_arg: list[str] = []
    if level_choice != "0":
        try:
            level_arg = ["--level", levels[int(level_choice) - 1]]
        except (ValueError, IndexError):
            print("  [!] Неверный выбор, будут обработаны все уровни.")

    edge_blur = ask("Edge-blur sigma для Гауссова шума", "1.0")
    seed      = ask("Random seed", "42")

    run([PYTHON, SCRIPT_DIR / "add_noise.py", drawing,
         *noise_arg, *level_arg,
         "--edge-blur", edge_blur,
         "--seed", seed])


def menu_full_pipeline() -> None:
    """4. Полный пайплайн для одного чертежа (preprocess → crops → noise)"""
    print("\n=== Полный пайплайн ===")
    print("Шаги: preprocess_dxf -> make_crops -> add_noise")
    print()

    src = choose_source("Выберите исходный DXF")
    if src is None:
        input("\nEnter для возврата...")
        return

    units     = ask("Единицы чертежа", "m")
    scale     = ask("Масштаб (знаменатель)", "100")
    dpi       = ask("DPI", "300")
    crop_size = ask("Размер тайла, px", "2048")
    overlap   = ask("Перекрытие, px", "256")
    offset    = ask("Offset, px", "0")
    min_seg   = ask("Минимум сегментов в кропе (0 = не фильтровать)", "10")
    seed      = ask("Random seed", "42")

    print("\n--- Шаг 1: препроцессирование DXF ---")
    run([PYTHON, SCRIPT_DIR / "preprocess_dxf.py", src,
         "--units", units, "--dpi", dpi], pause=False)

    print("\n--- Шаг 2: нарезка на кропы ---")
    run([PYTHON, SCRIPT_DIR / "make_crops.py", src,
         "--scale", scale, "--units", units,
         "--dpi", dpi, "--crop-size", crop_size,
         "--overlap", overlap, "--offset", offset,
         "--min-segments", min_seg], pause=False)

    # Определяем drawing_id так же, как make_crops.py
    raw_id = src.stem.replace("_geometry_only", "")
    try:
        drawing_id = f"{int(raw_id):03d}"
    except ValueError:
        drawing_id = raw_id
    drawing_dir = DATASET_DIR / "drawings" / drawing_id

    print("\n--- Шаг 3: генерация шума ---")
    run([PYTHON, SCRIPT_DIR / "add_noise.py", drawing_dir,
         "--seed", seed], pause=False)

    input("Нажмите Enter для возврата в меню...")


def menu_full_pipeline_all() -> None:
    """5. Полный пайплайн для ВСЕХ исходников"""
    print("\n=== Полный пайплайн для всех исходников ===")
    sources = list_sources()
    if not sources:
        print("  [!] Папка sources/ пуста или нет *_geometry_only.dxf.")
        input("\nEnter для возврата...")
        return

    print("Будут обработаны файлы:")
    for s in sources:
        print(f"  - {s.name}")
    print()

    units     = ask("Единицы чертежа", "m")
    scale     = ask("Масштаб (знаменатель)", "100")
    dpi       = ask("DPI", "300")
    crop_size = ask("Размер тайла, px", "2048")
    overlap   = ask("Перекрытие, px", "256")
    offset    = ask("Offset, px", "0")
    min_seg   = ask("Минимум сегментов в кропе (0 = не фильтровать)", "10")
    seed      = ask("Random seed", "42")

    confirm = ask(f"\nЗапустить полный пайплайн для {len(sources)} файлов? [y/n]", "y")
    if confirm.lower() != "y":
        return

    for idx, src in enumerate(sources, 1):
        print(f"\n{'='*54}")
        print(f"  Исходник {idx}/{len(sources)}: {src.name}")
        print(f"{'='*54}")

        print("\n--- Шаг 1: препроцессирование DXF ---")
        run([PYTHON, SCRIPT_DIR / "preprocess_dxf.py", src,
             "--units", units, "--dpi", dpi], pause=False)

        print("\n--- Шаг 2: нарезка на кропы ---")
        run([PYTHON, SCRIPT_DIR / "make_crops.py", src,
             "--scale", scale, "--units", units,
             "--dpi", dpi, "--crop-size", crop_size,
             "--overlap", overlap, "--offset", offset,
             "--min-segments", min_seg], pause=False)

        raw_id = src.stem.replace("_geometry_only", "")
        try:
            drawing_id = f"{int(raw_id):03d}"
        except ValueError:
            drawing_id = raw_id
        drawing_dir = DATASET_DIR / "drawings" / drawing_id

        print("\n--- Шаг 3: генерация шума ---")
        run([PYTHON, SCRIPT_DIR / "add_noise.py", drawing_dir,
             "--seed", seed], pause=False)

    print(f"\n  Обработано исходников: {len(sources)}")
    input("Нажмите Enter для возврата в меню...")


def menu_clean() -> None:
    """6. Очистить все сгенерированные данные"""
    print("\n=== Очистка сгенерированных данных ===")

    drawings_dir  = DATASET_DIR / "drawings"
    manifest_path = DATASET_DIR / "manifest.json"

    targets = []
    if drawings_dir.exists():
        subdirs = sorted(drawings_dir.iterdir())
        targets += subdirs
    if manifest_path.exists():
        targets.append(manifest_path)

    if not targets:
        print("  Нечего удалять: папка drawings/ и manifest.json отсутствуют.")
        input("\nEnter для возврата...")
        return

    print("\nБудет удалено:")
    total_files = 0
    for t in targets:
        if t.is_dir():
            files = list(t.rglob("*"))
            n = sum(1 for f in files if f.is_file())
            total_files += n
            print(f"  [dir]  {t.relative_to(DATASET_DIR)}  ({n} файлов)")
        else:
            total_files += 1
            print(f"  [file] {t.relative_to(DATASET_DIR)}")

    print(f"\n  Итого файлов к удалению: {total_files}")
    confirm = ask("\nВы уверены? Это действие необратимо! [yes/no]", "no")
    if confirm.lower() != "yes":
        print("  Отменено.")
        input("\nEnter для возврата...")
        return

    import shutil
    removed_dirs = 0
    removed_files = 0

    if drawings_dir.exists():
        for sub in sorted(drawings_dir.iterdir()):
            if sub.is_dir():
                shutil.rmtree(sub)
                removed_dirs += 1
            else:
                sub.unlink()
                removed_files += 1
        # Удаляем саму папку drawings/, если пустая
        try:
            drawings_dir.rmdir()
            removed_dirs += 1
        except OSError:
            pass

    if manifest_path.exists():
        manifest_path.unlink()
        removed_files += 1

    print(f"\n  Удалено папок : {removed_dirs}")
    print(f"  Удалено файлов: {removed_files}")
    print("  Готово.")
    input("\nEnter для возврата...")


def menu_yolo() -> None:
    """7. Сгенерировать YOLO-датасет (prepare_yolo_dataset.py)"""
    print("\n=== Генерация YOLO-датасета ===")
    print("Источник: *_full_annotated.dxf из sources/")
    print("Классы:   0=text (TEXT/MTEXT),  1=dimension (DIMENSION/MULTILEADER)")
    print()

    sources = sorted((DATASET_DIR / "sources").glob("*_full_annotated.dxf"))
    if not sources:
        print("  [!] Нет файлов *_full_annotated.dxf в sources/.")
        input("\nEnter для возврата...")
        return

    print(f"Найдено файлов: {len(sources)}")
    for s in sources:
        print(f"  - {s.name}")

    scale     = ask("\nМасштаб (знаменатель)", "100")
    units     = ask("Единицы чертежа", "m")
    dpi       = ask("DPI", "300")
    crop_size = ask("Размер кропа, px", "1280")
    overlap   = ask("Перекрытие, px", "64")
    min_ann   = ask("Минимум аннотаций в кропе", "1")
    val_split = ask("Доля val-выборки (0.0–1.0)", "0.2")
    no_noise  = ask("Только чистые изображения (без шума)? [y/n]", "n")
    seed      = ask("Random seed", "42")

    cmd = [PYTHON, SCRIPT_DIR / "prepare_yolo_dataset.py",
           "--scale", scale, "--units", units,
           "--dpi", dpi, "--crop-size", crop_size,
           "--overlap", overlap, "--min-annotations", min_ann,
           "--val-split", val_split, "--seed", seed]
    if no_noise.lower() == "y":
        cmd.append("--no-noise")

    run(cmd)


def menu_train_yolo() -> None:
    """8. Обучить YOLO-модель (train_yolo.py)"""
    print("\n=== Обучение YOLO ===")
    yolo_data = DATASET_DIR / "yolo_dataset" / "data.yaml"
    if not yolo_data.exists():
        print("  [!] yolo_dataset/data.yaml не найден. Сначала запустите пункт 7.")
        input("\nEnter для возврата...")
        return

    model   = ask("Базовая модель (yolo11n/s/m.pt)", "yolo11s.pt")
    epochs  = ask("Эпох", "100")
    batch   = ask("Batch size", "16")
    imgsz   = ask("Размер входа, px", "1280")
    name    = ask("Имя эксперимента", "v1")
    device  = ask("Устройство (0=GPU, cpu)", "0")

    run([PYTHON, SCRIPT_DIR / "train_yolo.py",
         "--model", model, "--epochs", epochs,
         "--batch", batch, "--imgsz", imgsz,
         "--name", name, "--device", device])


def menu_eval_yolo() -> None:
    """9. Оценить YOLO-модель"""
    print("\n=== Оценка YOLO-модели ===")
    models_dir = DATASET_DIR / "models" / "yolo"

    # Найти последнюю обученную модель
    best_pts = sorted(models_dir.rglob("best.pt")) if models_dir.exists() else []
    if best_pts:
        default_model = str(best_pts[-1])
        print(f"  Найдена модель: {best_pts[-1].relative_to(DATASET_DIR)}")
    else:
        default_model = ""
        print("  [!] best.pt не найден. Сначала запустите обучение (пункт 8).")

    model_path = ask("Путь к модели (.pt)", default_model)
    if not model_path:
        input("\nEnter для возврата...")
        return

    mode = ask("Режим (val / vis / both)", "both")
    conf = ask("Confidence threshold", "0.25")

    cmd = [PYTHON, SCRIPT_DIR / "eval_yolo.py",
           "--model", model_path, "--mode", mode, "--conf", conf]

    if mode in ("vis", "both"):
        source = ask("Папка с PNG для визуализации (Enter = val-сет)", "")
        if source:
            cmd += ["--source", source]

    run(cmd)


def menu_stats() -> None:
    """5. Статистика датасета"""
    print("\n=== Статистика датасета ===")
    manifest_path = DATASET_DIR / "manifest.json"
    if not manifest_path.exists():
        print("  manifest.json не найден. Сначала выполните нарезку и генерацию шума.")
        input("\nEnter для возврата...")
        return

    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)

    samples = m.get("samples", [])
    drawings = sorted({s["drawing_id"] for s in samples})
    noise_types = sorted({s["noise_type"] for s in samples if s["noise_type"]})

    print(f"\n  Чертежей       : {len(drawings)}")
    print(f"  Всего сэмплов  : {len(samples)}")
    print()

    print("  По чертежам:")
    for d in drawings:
        n = sum(1 for s in samples if s["drawing_id"] == d)
        meta_path = DATASET_DIR / "drawings" / d / "metadata.json"
        src_name = "?"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            src_name = meta.get("source_file", "?")
            n_crops = len(meta.get("crops", []))
        else:
            n_crops = 0
        print(f"    {d}  ({src_name})  :  {n_crops} кропов  x  {n} сэмплов")

    print()
    print("  По типам шума:")
    clean_n = sum(1 for s in samples if s["noise_type"] is None)
    print(f"    clean          : {clean_n}")
    for nt in noise_types:
        n = sum(1 for s in samples if s["noise_type"] == nt)
        print(f"    {nt:<20}: {n}")

    print()
    # Проверяем наличие файлов
    missing = 0
    for s in samples:
        img_path = DATASET_DIR / s["image"]
        if not img_path.exists():
            missing += 1
    if missing:
        print(f"  [!] Отсутствует файлов изображений: {missing}")
    else:
        print("  Все файлы изображений на месте.")

    input("\nEnter для возврата...")


# ──────────────────────────────────────────────────────────────────────────────
# Главное меню
# ──────────────────────────────────────────────────────────────────────────────

MENU = [
    ("1", "Препроцессировать DXF",                             menu_preprocess),
    ("2", "Нарезать чертёж на кропы",                          menu_make_crops),
    ("3", "Сгенерировать шумовые варианты",                     menu_add_noise),
    ("4", "Полный пайплайн — один исходник",                   menu_full_pipeline),
    ("5", "Полный пайплайн — ВСЕ исходники",                   menu_full_pipeline_all),
    ("6", "Статистика датасета",                                menu_stats),
    ("7", "Сгенерировать YOLO-датасет",                        menu_yolo),
    ("8", "Обучить YOLO-модель",                               menu_train_yolo),
    ("9", "Оценить YOLO-модель (метрики + визуализация)",      menu_eval_yolo),
    ("c", "Очистить все сгенерированные данные",                menu_clean),
    ("0", "Выход",                                              None),
]


def print_menu() -> None:
    manifest_path = DATASET_DIR / "manifest.json"
    total = 0
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        total     = len(data.get("samples", []))
        n_drawing = len({s["drawing_id"] for s in data.get("samples", [])})
    else:
        n_drawing = 0

    print("\n" + "=" * 54)
    print("  Dataset builder — Vectorization research dataset")
    print("=" * 54)
    print(f"  Корень : {DATASET_DIR}")
    print(f"  Чертежей в датасете : {n_drawing}   Сэмплов : {total}")
    print("=" * 54)
    for key, label, _ in MENU:
        print(f"  {key}.  {label}")
    print("=" * 54)


def main() -> None:
    while True:
        print_menu()
        choice = input("Выбор: ").strip()
        for key, _, action in MENU:
            if choice == key:
                if action is None:
                    print("Выход.\n")
                    sys.exit(0)
                action()
                break
        else:
            print("  [!] Неверный ввод.")


if __name__ == "__main__":
    main()
