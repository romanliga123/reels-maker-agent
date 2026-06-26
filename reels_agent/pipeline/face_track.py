"""Кроп 16:9 → 9:16.

Раньше здесь было автоопределение лица через OpenCV (Haar Cascade, сэмплинг
кадров через cv2.VideoCapture) для центрирования crop-window по говорящему.
Убрали целиком: сам OpenCV — это лишний импортируемый процесс (+ память,
которая остаётся в процессе навсегда после первого использования, как было
с librosa/numba) ради эффекта, который для большинства подкастов/стримов с
говорящим в центре кадра не сильно отличается от простого центрированного
кропа. Надёжность на бесплатном тире важнее точного кропа по лицу.
"""
from dataclasses import dataclass


@dataclass
class CropPlan:
    x: int
    y: int
    width: int
    height: int


def compute_crop_plan(
    src_width: int,
    src_height: int,
    target_w: int = 1080,
    target_h: int = 1920,
) -> CropPlan:
    crop_w = int(round(src_height * target_w / target_h))
    crop_w = min(crop_w, src_width)
    crop_h = src_height
    x = (src_width - crop_w) // 2
    return CropPlan(x=x, y=0, width=crop_w, height=crop_h)
