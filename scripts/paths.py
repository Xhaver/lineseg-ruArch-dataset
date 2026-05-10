"""
paths.py — Централизованные пути для всех скриптов датасета.

Импортируй нужные константы:
    from paths import GEOMETRY_ONLY_DIR, DRAWINGS_DIR, ...
"""
from pathlib import Path

SCRIPTS_DIR              = Path(__file__).parent.resolve()
DATASET_DIR              = SCRIPTS_DIR.parent.resolve()
SOURCES_DIR              = DATASET_DIR / "sources"

# ── Подкаталоги sources/ ────────────────────────────────────────────────────
FULL_ANNOTATED_DIR       = SOURCES_DIR / "full_annotated"        # исходники с аннотациями
UNPROCESSED_GEOMETRY_DIR = SOURCES_DIR / "unprocessed_geometry_only"  # геометрия до preprocess
GEOMETRY_ONLY_DIR        = SOURCES_DIR / "geometry_only"         # выход preprocess_dxf.py
DWG_DIR                  = SOURCES_DIR / "dwg"                   # оригинальные DWG

# ── Выходные каталоги датасета ──────────────────────────────────────────────
DRAWINGS_DIR             = DATASET_DIR / "drawings"
MANIFEST_PATH            = DATASET_DIR / "manifest.json"
YOLO_DATASET_DIR         = DATASET_DIR / "yolo_dataset"
YOLO_MANIFEST_PATH       = DATASET_DIR / "yolo_manifest.json"
MODELS_DIR               = DATASET_DIR / "models"
