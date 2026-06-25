"""Автокроп 16:9 → 9:16 с учётом положения лица в кадре.

Сэмплируем кадры внутри клипа примерно раз в 0.5с, ищем лицо через OpenCV
Haar Cascade (работает офлайн, без скачивания моделей — в отличие от
mediapipe Tasks API, который на Windows требует отдельной загрузки .tflite).
Берём триммированное среднее по X-координатам центра лица — один статичный
crop-window на весь клип, без покадрового панорамирования (вне scope v1).

Один seek (cap.set) на весь клип, дальше — последовательные cap.read() с
пропуском кадров между сэмплами. Раньше делали cap.set() на КАЖДЫЙ сэмпл —
для удалённого видео (presigned URL) каждый seek может означать новый сетевой
запрос/пересинхронизацию декодера, и десятки seek'ов на один клип ощутимо
тормозили рендер без какой-либо обратной связи пользователю.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


@dataclass
class CropPlan:
    x: int
    y: int
    width: int
    height: int


def compute_crop_plan(
    video_path: str | Path,
    start: float,
    end: float,
    src_width: int,
    src_height: int,
    target_w: int = 1080,
    target_h: int = 1920,
    sample_interval: float = 0.5,
    on_progress: Callable[[float], None] | None = None,
) -> CropPlan:
    crop_w = int(round(src_height * target_w / target_h))
    crop_w = min(crop_w, src_width)
    crop_h = src_height

    detector = cv2.CascadeClassifier(_CASCADE_PATH)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    sample_every = max(1, round(sample_interval * fps))
    total_frames = max(1, int((end - start) * fps))

    centers: list[float] = []
    frame_idx = 0
    while frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces):
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                centers.append(fx + fw / 2)
        frame_idx += 1
        if on_progress:
            on_progress(min(1.0, frame_idx / total_frames))
    cap.release()

    if centers:
        center_x = float(np.mean(_trim_outliers(centers)))
    else:
        center_x = src_width / 2.0

    x = int(round(center_x - crop_w / 2))
    x = max(0, min(x, src_width - crop_w))
    return CropPlan(x=x, y=0, width=crop_w, height=crop_h)


def _trim_outliers(values: list[float], trim_frac: float = 0.1) -> list[float]:
    if len(values) < 5:
        return values
    s = sorted(values)
    k = max(1, int(len(s) * trim_frac))
    return s[k:-k] if len(s) > 2 * k else s
