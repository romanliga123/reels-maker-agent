"""Автокроп 16:9 → 9:16 с учётом положения лица в кадре.

Сэмплируем кадры внутри клипа примерно раз в 0.5с, ищем лицо через OpenCV
Haar Cascade (работает офлайн, без скачивания моделей — в отличие от
mediapipe Tasks API, который на Windows требует отдельной загрузки .tflite).
Берём триммированное среднее по X-координатам центра лица — один статичный
crop-window на весь клип, без покадрового панорамирования (вне scope v1).
"""
from dataclasses import dataclass
from pathlib import Path

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
    video_path: Path,
    start: float,
    end: float,
    src_width: int,
    src_height: int,
    target_w: int = 1080,
    target_h: int = 1920,
    sample_interval: float = 0.5,
) -> CropPlan:
    crop_w = int(round(src_height * target_w / target_h))
    crop_w = min(crop_w, src_width)
    crop_h = src_height

    detector = cv2.CascadeClassifier(_CASCADE_PATH)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    centers: list[float] = []
    t = start
    while t < end:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces):
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                centers.append(fx + fw / 2)
        t += sample_interval
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
