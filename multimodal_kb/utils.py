import base64
import hashlib
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def compute_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def encode_file_base64(file_path: str) -> str:
    with open(file_path, "rb") as file_obj:
        return base64.b64encode(file_obj.read()).decode("utf-8")


def encode_frame_base64(frame: np.ndarray) -> str:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_video_thumbnail_base64(video_path: str) -> Optional[str]:
    cap = cv2.VideoCapture(video_path)
    try:
        ret, frame = cap.read()
        if not ret:
            return None
        return encode_frame_base64(frame)
    finally:
        cap.release()
