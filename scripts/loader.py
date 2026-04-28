"""
loader.py — Dataset API
=======================

Единственная точка доступа к датасету.  Читает manifest.json и
предоставляет ленивый итератор объектов Sample, каждый из которых
умеет загружать изображение как numpy-массив и GT как список отрезков.

Быстрый старт
-------------
    from scripts.loader import DatasetLoader

    loader = DatasetLoader("path/to/Dataset")  # или просто DatasetLoader()
                                               # если запускаем из Dataset/

    # все чистые сэмплы первого чертежа
    for sample in loader.iter(noise_type=None, drawing_id="001"):
        img = sample.image()        # np.ndarray uint8, HxW
        gt  = sample.gt_segments()  # list[(x1,y1,x2,y2)]  — пиксели, Y↑
        print(sample, img.shape, len(gt), "отрезков GT")

    # случайный батч
    batch = loader.random_batch(size=8, noise_type="gaussian_blur", seed=0)
    images = [s.image() for s in batch]

    # статистика
    loader.summary()
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image

try:
    import ezdxf
except ImportError:
    ezdxf = None  # GT loading will fail gracefully with a clear message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_root(hint: Optional[str | Path] = None) -> Path:
    """
    Ищет корень датасета в порядке приоритета:
      1. явный аргумент hint
      2. переменная окружения DATASET_ROOT
      3. родительский каталог этого скрипта (Dataset/scripts/../ = Dataset/)
    """
    import os

    candidates: list[Path] = []
    if hint is not None:
        candidates.append(Path(hint))
    env = os.environ.get("DATASET_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).parent.parent.resolve())  # Dataset/

    for c in candidates:
        if (c / "manifest.json").exists():
            return c

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"manifest.json не найден. Проверьте: {searched}\n"
        "Или задайте переменную окружения DATASET_ROOT."
    )


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """
    Один сэмпл датасета (пара изображение + GT).

    Атрибуты соответствуют полям manifest.json, плюс удобные методы
    для загрузки данных.
    """
    id: str
    drawing_id: str
    crop_id: str
    noise_type: Optional[str]   # None → чистый сэмпл
    noise_level: int             # 0 → чистый, 1–5 → зашумлённый
    params: dict
    _image_path: Path
    _gt_path: Path

    # ------------------------------------------------------------------
    def image(self) -> np.ndarray:
        """
        Загружает изображение как uint8 grayscale numpy-массив (HxW).
        Чистые и большинство зашумлённых кропов — 1-bit TIFF (0/255).
        gaussian_blur — 8-bit grayscale.
        """
        with Image.open(self._image_path) as pil:
            if pil.mode == "1":
                return np.array(pil, dtype=bool).astype(np.uint8) * 255
            return np.array(pil.convert("L"), dtype=np.uint8)

    def gt_segments(self) -> list[tuple[float, float, float, float]]:
        """
        Загружает GT LINE-сегменты из DXF.

        Координаты — пиксельные, Y↑ (DXF-конвенция):
          x ∈ [0, crop_width],  y=0 — нижний край, y=crop_height — верхний.

        Для сравнения с результатами алгоритмов нужен flip:
          img_y = img_height - gt_y
        Это делает DatasetBridge в Pipeline автоматически.
        """
        if ezdxf is None:
            raise ImportError("ezdxf не установлен: pip install ezdxf")
        doc = ezdxf.readfile(str(self._gt_path))
        segs: list[tuple[float, float, float, float]] = []
        for e in doc.modelspace():
            if e.dxftype() == "LINE":
                s, en = e.dxf.start, e.dxf.end
                segs.append((float(s.x), float(s.y), float(en.x), float(en.y)))
        return segs

    # ------------------------------------------------------------------
    @property
    def is_clean(self) -> bool:
        return self.noise_type is None

    @property
    def image_path(self) -> Path:
        return self._image_path

    @property
    def gt_path(self) -> Path:
        return self._gt_path

    def __str__(self) -> str:
        noise = f"{self.noise_type}_L{self.noise_level}" if self.noise_type else "clean"
        return f"Sample({self.drawing_id}/{self.crop_id}/{noise})"


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Точка входа для работы с датасетом.

    Параметры
    ---------
    root : путь к директории Dataset/ (содержит manifest.json).
           Можно не указывать, если запускаем из Dataset/ или задан
           DATASET_ROOT.
    """

    def __init__(self, root: Optional[str | Path] = None) -> None:
        self.root = _find_root(root)
        with open(self.root / "manifest.json", encoding="utf-8") as f:
            data = json.load(f)
        self._samples = self._parse(data["samples"])

    # ------------------------------------------------------------------
    def _parse(self, raw: list[dict]) -> list[Sample]:
        samples = []
        for r in raw:
            samples.append(Sample(
                id           = r["id"],
                drawing_id   = r["drawing_id"],
                crop_id      = r["crop_id"],
                noise_type   = r["noise_type"],
                noise_level  = r["noise_level"],
                params       = r.get("params", {}),
                _image_path  = self.root / r["image"],
                _gt_path     = self.root / r["gt"],
            ))
        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, sample_id: str) -> Sample:
        for s in self._samples:
            if s.id == sample_id:
                return s
        raise KeyError(f"Сэмпл '{sample_id}' не найден в манифесте.")

    def __repr__(self) -> str:
        return f"DatasetLoader(root={self.root}, samples={len(self)})"

    # ------------------------------------------------------------------
    def iter(
        self,
        *,
        noise_type: Optional[str]  = ...,   # ... = не фильтровать
        noise_level: Optional[int] = None,
        drawing_id: Optional[str]  = None,
        crop_id: Optional[str]     = None,
    ) -> Iterator[Sample]:
        """
        Ленивый итератор с опциональной фильтрацией.

        noise_type=None      → только чистые сэмплы
        noise_type="gaussian_blur" → только этот тип шума
        noise_type=...       → (по умолчанию) без фильтрации по типу
        """
        for s in self._samples:
            if noise_type is not ...:
                if s.noise_type != noise_type:
                    continue
            if noise_level is not None and s.noise_level != noise_level:
                continue
            if drawing_id is not None and s.drawing_id != drawing_id:
                continue
            if crop_id is not None and s.crop_id != crop_id:
                continue
            yield s

    def clean(self, **kw) -> Iterator[Sample]:
        """Только чистые сэмплы (noise_type=None)."""
        return self.iter(noise_type=None, **kw)

    def noisy(self, noise_type: Optional[str] = None, **kw) -> Iterator[Sample]:
        """Только зашумлённые сэмплы. noise_type=None → все типы шума."""
        if noise_type is not None:
            return self.iter(noise_type=noise_type, **kw)
        # Все зашумлённые: исключаем чистые вручную
        def _gen():
            for s in self.iter(**kw):
                if s.noise_type is not None:
                    yield s
        return _gen()

    def drawing_ids(self) -> list[str]:
        return sorted({s.drawing_id for s in self._samples})

    def noise_types(self) -> list[Optional[str]]:
        return sorted({s.noise_type for s in self._samples}, key=lambda x: x or "")

    # ------------------------------------------------------------------
    def random_batch(
        self,
        size: int,
        seed: int = 42,
        **filter_kw,
    ) -> list[Sample]:
        """Случайные size сэмплов (с фильтрацией)."""
        pool = list(self.iter(**filter_kw))
        rng  = random.Random(seed)
        return rng.sample(pool, min(size, len(pool)))

    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Печатает краткую статистику датасета."""
        from collections import Counter
        total   = len(self._samples)
        by_type = Counter(s.noise_type for s in self._samples)
        by_drw  = Counter(s.drawing_id for s in self._samples)

        print(f"DatasetLoader  root={self.root}")
        print(f"  Всего сэмплов : {total}")
        print(f"  Чертежей      : {len(by_drw)}  {sorted(by_drw)}")
        print(f"  Типы шума:")
        for t, n in sorted(by_type.items(), key=lambda x: x[0] or ""):
            label = t if t else "clean"
            print(f"    {label:<22} {n:>4}")
