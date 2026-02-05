import asyncio
import io
import os
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

DOTENV_PATH = Path(__file__).with_name(".env")
DOTENV_LOADED = False

if load_dotenv is not None and DOTENV_PATH.exists():
    # Allows using a local .env file (ignored by git) for convenience.
    DOTENV_LOADED = bool(load_dotenv(dotenv_path=DOTENV_PATH, override=False))

import requests
from PIL import Image, ImageEnhance

from idotmatrix import ConnectionManager
from idotmatrix import Image as IDotMatrixImage
from bleak.exc import BleakError

# ================== CONFIG ==================
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")
LASTFM_USER = os.environ.get("LASTFM_USER")

# Use "auto" by default to avoid committing a personal device MAC address.
IDOTMATRIX_ADDRESS = os.environ.get("IDOTMATRIX_ADDRESS", "auto")

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))  # seconds
# ===========================================


def validate_config() -> None:
    missing = []
    if not LASTFM_API_KEY:
        missing.append("LASTFM_API_KEY")
    if not LASTFM_USER:
        missing.append("LASTFM_USER")
    if missing:
        hint = "Set them in your shell"
        if DOTENV_PATH.exists():
            hint = f"Fill them in {DOTENV_PATH.name}"
        raise RuntimeError(
            "Missing config: set "
            + " and ".join(missing)
            + " environment variable(s). "
            + hint
            + ". See README.md for PowerShell examples."
        )

LASTFM_URL = "https://ws.audioscrobbler.com/2.0/"
LASTFM_HEADERS = {
    # Last.fm API can be picky; a UA helps avoid 403s.
    "User-Agent": "lastfm-idotmatrix/1.0",
}
from PIL import Image

def dither_after_enhance(img: Image.Image, colors: int = 128) -> Image.Image:
    """
    Apply Floyd–Steinberg dithering AFTER contrast & sharpness.
    No RGB565, stays fully RGB for idotmatrix.
    """
    # Convert to palette with dithering
    img = img.convert(
        "P",
        dither=Image.FLOYDSTEINBERG,
        palette=Image.ADAPTIVE,
        colors=colors
    )

    # Convert back to RGB (idotmatrix expects RGB PNG)
    img = img.convert("RGB")

    return img

def rgb_to_rgb565_pil(img: Image.Image) -> Image.Image:
    """
    Convert a PIL RGB image to RGB565 color space
    while keeping it as a PIL RGB image.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    pixels = img.load()
    w, h = img.size

    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]

            # Quantize to RGB565
            r5 = r >> 3
            g6 = g >> 2
            b5 = b >> 3

            # Expand back to 8-bit (for display / PIL)
            r8 = (r5 << 3) | (r5 >> 2)
            g8 = (g6 << 2) | (g6 >> 4)
            b8 = (b5 << 3) | (b5 >> 2)

            pixels[x, y] = (r8, g8, b8)

    return img

from PIL import Image
import numpy as np

def ordered_dither(img: Image.Image, levels: int = 8) -> Image.Image:
    """
    Ordered (Bayer) dithering for RGB images.
    
    levels = number of brightness levels per channel
    Typical values:
      4  -> very strong pattern
      6  -> balanced
      8  -> subtle (recommended)
    """
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    # 4x4 Bayer matrix (values 0..15)
    bayer = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
    ], dtype=np.float32)

    h, w, _ = arr.shape
    threshold = (bayer / 16.0 - 0.5) / levels

    for y in range(h):
        for x in range(w):
            t = threshold[y % 4, x % 4]
            arr[y, x] = arr[y, x] / 255.0 + t

    arr = np.clip(arr, 0, 1)
    arr = np.round(arr * (levels - 1)) / (levels - 1)
    arr = (arr * 255).astype(np.uint8)

    return Image.fromarray(arr, "RGB")


def get_now_playing():
    params = {
        "method": "user.getrecenttracks",
        "user": LASTFM_USER,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "limit": 1
    }
    r = requests.get(LASTFM_URL, params=params, headers=LASTFM_HEADERS, timeout=10)
    r.raise_for_status()
    tracks = r.json()["recenttracks"]["track"]
    if not tracks:
        return None

    track = tracks[0]
    if "@attr" not in track or track["@attr"].get("nowplaying") != "true":
        return None

    image_url = None
    for img in reversed(track.get("image", [])):
        url = (img or {}).get("#text")
        if url:
            image_url = url
            break

    return image_url, track["name"], track["artist"]["#text"]

import numpy as np
from PIL import Image

def directional_dither_horizontal(img, levels=96):
    """
    Horizontal-only error diffusion dithering.
    Designed to hide vertical banding on LED matrices.
    
    levels = effective color levels per channel (64–128 recommended)
    """
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    h, w, _ = arr.shape
    step = 255 / (levels - 1)

    for y in range(h):
        for x in range(w):
            old_pixel = arr[y, x].copy()

            # Quantize
            new_pixel = np.round(old_pixel / step) * step
            arr[y, x] = new_pixel

            # Error
            error = old_pixel - new_pixel

            # Push error mostly to the right
            if x + 1 < w:
                arr[y, x + 1] += error * 0.9

            # Small vertical diffusion to avoid streaks
            if y + 1 < h:
                arr[y + 1, x] += error * 0.1

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def download_and_prepare_png(url: str, pixel_size: int = 32) -> str:
    if not url:
        raise ValueError("No image URL available for this track")

    r = requests.get(url, headers=LASTFM_HEADERS, timeout=10)
    r.raise_for_status()

    img = Image.open(io.BytesIO(r.content)).convert("RGB")

    # Resize
    img = img.resize((32, 32), Image.Resampling.BILINEAR)

    # Gentle enhancement (do NOT overdo contrast)
    img = ImageEnhance.Contrast(img).enhance(1.4)

    # Floyd–Steinberg dithering
    # img = img.convert(
    #     "P",
    #     dither=Image.FLOYDSTEINBERG,
    #     palette=Image.ADAPTIVE,
    #     colors=96
    # ).convert("RGB")

    img = directional_dither_horizontal(img, levels=96)


    # img = img.resize((pixel_size, pixel_size), Image.Resampling.BILINEAR)

    # sharpen the image a bit
    # sharpener = ImageEnhance.Sharpness(img)
    # img = sharpener.enhance(1.3)

    # contraster = ImageEnhance.Contrast(img)
    # img = contraster.enhance(1.5)

    # img = dither_after_enhance(img, colors=128)

    # THEN ordered dither
    # img = ordered_dither(img, levels=8)

    # brightener = ImageEnhance.Brightness(img)
    # img = brightener.enhance(1.2)

    # set dark colors to black
    # def threshold(c):
    #     return 0 if c < 10 else c
    
    # img = img.point(threshold)

    # img = rgb_to_rgb565_pil(img)


    # The idotmatrix library's upload functions work with file paths.
    # Use a real temporary file (NamedTemporaryFile(delete=False)) for Windows compatibility.
    tmp = tempfile.NamedTemporaryFile(prefix="lastfm_", suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()

    img.save(tmp_path, format="PNG", optimize=True)
    return tmp_path

async def main_async():
    try:
        validate_config()
    except RuntimeError as e:
        print("Error:", e)
        return

    conn = ConnectionManager()
    try:
        if str(IDOTMATRIX_ADDRESS).lower() == "auto":
            await conn.connectBySearch()
        else:
            await conn.connectByAddress(IDOTMATRIX_ADDRESS)
    except BleakError as e:
        print("Bluetooth error:", e)
        print(
            "If scanning fails on Windows, pair the device in Windows settings, "
            "extract its MAC address, then set IDOTMATRIX_ADDRESS (e.g. AA:BB:CC:DD:EE:FF)."
        )
        return

    device_image = IDotMatrixImage()
    await device_image.setMode(1)

    last_track = None

    print("Listening for now playing tracks...")

    while True:
        try:
            data = get_now_playing()
            if not data:
                await asyncio.sleep(POLL_INTERVAL)
                continue

            image_url, title, artist = data
            track_id = f"{artist}-{title}"

            if track_id != last_track:
                if not image_url:
                    print(f"No album art for: {artist} - {title}")
                    last_track = track_id
                    continue

                tmp_path = None
                try:
                    tmp_path = download_and_prepare_png(image_url, pixel_size=32)
                    await device_image.uploadUnprocessed(tmp_path)
                    print(f"Updated: {artist} - {title}")
                    last_track = track_id
                finally:
                    if tmp_path:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

        except Exception as e:
            print("Error:", e)

        await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main_async())
