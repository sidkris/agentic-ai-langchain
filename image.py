#!/usr/bin/env python3
"""
Auto-enhance images (clarity, brightness, color) with sensible defaults.

Features
- Auto-orient via EXIF
- White balance (gray-world)
- Gentle brightness & contrast boost (percentile-based; CLAHE if OpenCV available)
- Denoise (median/bilateral) depending on noise estimate
- Light unsharp mask for clarity
- Batch processing; preserves EXIF if possible

Dependencies
- Required: Pillow, numpy
- Optional: opencv-python (if installed, uses CLAHE + better denoise)

Install:
  pip install pillow numpy
  # optional:
  pip install opencv-python
"""
import argparse
import os
from pathlib import Path
import math

from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ExifTags
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def exif_transpose(im: Image.Image) -> Image.Image:
    # Same as ImageOps.exif_transpose, but safe on older Pillow
    try:
        return ImageOps.exif_transpose(im)
    except Exception:
        return im


def to_numpy_rgb(im: Image.Image) -> np.ndarray:
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.array(im)


def from_numpy_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def estimate_noise_gray(img_gray: np.ndarray) -> float:
    # Simple noise proxy: median absolute deviation of Laplacian
    lap = cv2.Laplacian(img_gray, cv2.CV_64F) if _HAS_CV2 else np.gradient(img_gray.astype(np.float32))[0]
    mad = np.median(np.abs(lap - np.median(lap)))
    return float(mad)


def grayworld_white_balance(rgb: np.ndarray) -> np.ndarray:
    # Scale each channel so mean equals grand mean
    eps = 1e-6
    means = rgb.reshape(-1, 3).mean(axis=0) + eps
    gmean = means.mean()
    gains = gmean / means
    balanced = np.clip(rgb * gains, 0, 255)
    return balanced


def auto_gamma(rgb: np.ndarray) -> np.ndarray:
    # Map median luminance to ~0.5 to avoid over/under-exposure
    lum = 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]
    med = np.percentile(lum, 50)
    if med <= 1:  # too dark
        return rgb
    target = 128.0  # 0.5 on 0..255
    gamma = math.log(target/255.0) / math.log(max(med/255.0, 1e-6))
    gamma = np.clip(gamma, 0.7, 1.4)  # keep it subtle
    # Apply gamma
    lut = (np.linspace(0, 1, 256) ** gamma) * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(rgb.astype(np.uint8), lut) if _HAS_CV2 else lut[rgb]


def clahe_contrast(rgb: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L2 = clahe.apply(L)
        lab2 = cv2.merge([L2, A, B])
        rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return rgb2
    else:
        # Pillow fallback: autocontrast + slight brightness
        pil = from_numpy_rgb(rgb)
        pil = ImageOps.autocontrast(pil, cutoff=0.5)
        pil = ImageEnhance.Brightness(pil).enhance(1.05)
        return to_numpy_rgb(pil)


def denoise_rgb(rgb: np.ndarray, strength: str, noise_level: float|None=None) -> np.ndarray:
    # Decide denoise method
    if not _HAS_CV2:
        # Median filter only if strong noise
        if noise_level is not None and noise_level < 2.0 and strength != "high":
            return rgb
        pil = from_numpy_rgb(rgb)
        pil = pil.filter(ImageFilter.MedianFilter(size=3))
        return to_numpy_rgb(pil)

    # OpenCV path
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if noise_level is None:
        noise_level = estimate_noise_gray(gray)
    # Choose bilateral if edges should be preserved
    if noise_level < 1.0 and strength == "low":
        return rgb
    if strength == "high":
        return cv2.fastNlMeansDenoisingColored(rgb, None, 10, 10, 7, 21)
    else:
        # medium/default: mild bilateral to keep details
        return cv2.bilateralFilter(rgb, d=5, sigmaColor=60, sigmaSpace=60)


def unsharp_mask(pil: Image.Image, strength: str) -> Image.Image:
    # Gentle sharpen; avoid halos
    if strength == "low":
        radius, percent = 1.2, 80
    elif strength == "high":
        radius, percent = 2.0, 140
    else:
        radius, percent = 1.5, 110
    return pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=2))


def needs_brightness_boost(rgb: np.ndarray) -> bool:
    lum = 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]
    p10, p90 = np.percentile(lum, [10, 90])
    return (p90 < 210) or (p10 > 20)  # narrow/dim dynamic range


def enhance_image(pil_img: Image.Image, strength: str="medium") -> Image.Image:
    pil = exif_transpose(pil_img.convert("RGB"))
    rgb = to_numpy_rgb(pil)

    # White balance
    rgb = grayworld_white_balance(rgb)

    # Optional gamma to normalize midtones
    rgb = auto_gamma(rgb)

    # Contrast (CLAHE or fallback)
    rgb = clahe_contrast(rgb)

    # Denoise if needed
    noise_level = None
    if _HAS_CV2:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        noise_level = estimate_noise_gray(gray)
    rgb = denoise_rgb(rgb, strength=strength, noise_level=noise_level)

    # Gentle brightness bump if image still looks dim/flat
    if needs_brightness_boost(rgb):
        pil_tmp = from_numpy_rgb(rgb)
        factor = {"low": 1.03, "medium": 1.07, "high": 1.12}[strength]
        pil_tmp = ImageEnhance.Brightness(pil_tmp).enhance(factor)
        rgb = to_numpy_rgb(pil_tmp)

    # Subtle sharpen
    out = unsharp_mask(from_numpy_rgb(rgb), strength)

    return out


def build_out_path(in_path: Path, out_dir: Path|None, suffix: str="_enhanced") -> Path:
    out_dir = out_dir or in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem, ext = in_path.stem, in_path.suffix or ".jpg"
    return out_dir / f"{stem}{suffix}{ext}"


def process_one(path: Path, out_dir: Path|None, quality: int, strength: str, keep_exif: bool):
    with Image.open(path) as im:
        exif = im.info.get("exif") if keep_exif else None
        enhanced = enhance_image(im, strength=strength)
        out_path = build_out_path(path, out_dir)
        save_args = {}
        if out_path.suffix.lower() in [".jpg", ".jpeg"]:
            save_args.update(dict(quality=quality, subsampling=1, optimize=True))
        if exif:
            save_args["exif"] = exif
        enhanced.save(out_path, **save_args)
        print(f"✓ {path} → {out_path}")


def valid_strength(s: str) -> str:
    s = s.lower()
    if s not in {"low","medium","high"}:
        raise argparse.ArgumentTypeError("strength must be one of: low, medium, high")
    return s


def main():
    ap = argparse.ArgumentParser(description="Enhance images (clarity, brightness, contrast, color).")
    ap.add_argument("inputs", nargs="+", help="Input image(s), supports globs (quote them in shells).")
    ap.add_argument("-o","--out-dir", type=str, default=None, help="Output directory (default: alongside inputs).")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality (default: 92).")
    ap.add_argument("--strength", type=valid_strength, default="medium", help="Enhancement aggressiveness.")
    ap.add_argument("--no-exif", action="store_true", help="Do not preserve EXIF metadata in outputs.")
    args = ap.parse_args()

    # Expand globs safely
    paths = []
    for p in args.inputs:
        matches = list(map(Path, sorted(Path().glob(p) if any(ch in p for ch in "*?[]") else [p])))
        paths.extend(matches)
    if not paths:
        print("No matching input files.")
        return

    out_dir = Path(args.out_dir) if args.out_dir else None
    for p in paths:
        try:
            process_one(Path(p), out_dir, args.quality, args.strength, keep_exif=not args.no_exif)
        except Exception as e:
            print(f"✗ Failed {p}: {e}")

if __name__ == "__main__":
    main()
