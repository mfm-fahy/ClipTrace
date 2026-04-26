"""
Augmentation Pipeline
Simulates real-world video edits applied during training to make
the embedding model robust to transformations.
"""
import cv2
import numpy as np
import random
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import config


# ── Individual augmentations ──────────────────────────────────────────────────

def random_crop_resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = random.uniform(0.6, 0.95)
    new_h, new_w = int(h * scale), int(w * scale)
    y = random.randint(0, h - new_h)
    x = random.randint(0, w - new_w)
    cropped = img[y:y+new_h, x:x+new_w]
    return cv2.resize(cropped, (w, h))


def random_blur(img: np.ndarray) -> np.ndarray:
    ksize = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def jpeg_compression(img: np.ndarray) -> np.ndarray:
    quality = random.randint(20, 60)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def color_jitter(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.5, 1.5))
    pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.5, 1.5))
    pil = ImageEnhance.Color(pil).enhance(random.uniform(0.0, 2.0))
    pil = ImageEnhance.Sharpness(pil).enhance(random.uniform(0.5, 1.5))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def add_noise(img: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, random.uniform(5, 25), img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_text_overlay(img: np.ndarray) -> np.ndarray:
    overlay = img.copy()
    texts = ["LIVE", "BREAKING", "HD", "© 2024", "SPORTS", "REPLAY"]
    text = random.choice(texts)
    x = random.randint(5, img.shape[1] // 2)
    y = random.randint(20, img.shape[0] - 10)
    font_scale = random.uniform(0.5, 1.2)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    alpha = random.uniform(0.5, 1.0)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def horizontal_flip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def grayscale_convert(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def resize_distort(img: np.ndarray) -> np.ndarray:
    """Simulate resolution downgrade then upscale."""
    h, w = img.shape[:2]
    scale = random.uniform(0.3, 0.6)
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ── Composed augmentation ─────────────────────────────────────────────────────

_AUGMENTATIONS = [
    random_crop_resize,
    random_blur,
    jpeg_compression,
    color_jitter,
    add_noise,
    add_text_overlay,
    horizontal_flip,
    grayscale_convert,
    resize_distort,
]


def apply_random_augmentations(img: np.ndarray, n_augs: int = 2) -> np.ndarray:
    """Apply n_augs randomly chosen augmentations to a frame."""
    chosen = random.sample(_AUGMENTATIONS, min(n_augs, len(_AUGMENTATIONS)))
    for aug in chosen:
        img = aug(img)
    return img


# ── PyTorch transforms ────────────────────────────────────────────────────────

def get_train_transform():
    return T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.3),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform():
    return T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
