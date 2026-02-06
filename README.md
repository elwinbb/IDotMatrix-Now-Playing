# IDot Matrix (Now Playing for Music Services using Last.fm)

## 🎵 IDotMatrix – Now Playing Display

**IDotMatrix – Now Playing Display** is a Python-based project that transforms an **IDotMatrix Bluetooth LED display** into a real-time **“Now Playing” music visualizer**.

The application continuously fetches the **currently playing track from Last.fm**, which can be linked to **Spotify, Apple Music, or any other music service supported by Last.fm scrobbling**. It retrieves track metadata and album artwork, processes the album art to fit the LED matrix resolution, and wirelessly transmits it to the display via **Bluetooth**.

An optional **clock overlay** can be rendered on top of the album artwork, allowing the display to function both as a **music display and a smart clock**. The display updates automatically whenever the track changes, creating a smooth and dynamic visual experience.

This project is well suited for **desk setups, ambient displays, and IoT experiments**, combining **Python, Bluetooth communication, image processing, and real-time music APIs**.

---

### ✨ Features

- 🎶 Fetches the **currently playing song** from Last.fm  
- 🔗 Supports **Spotify, Apple Music, and other scrobble-enabled services**  
- 🖼️ Displays **album artwork** on an IDotMatrix LED matrix  
- 🕒 Optional **time overlay** on top of the album art  
- 📡 Wireless **Bluetooth communication**  
- 🔄 Automatically updates when the song changes  

## Repository layout

- `now_playing.py`
  - Polls Last.fm for the current track and uploads the album art to the device.
  - Optional features (via env vars): overlay a small clock on the album art, and/or switch the device to its built-in clock when idle.
- `requirements.txt`
  - Dependencies for the root scripts.

## Requirements

- Windows 10/11 (recommended for this repo’s current setup)
- Python 3.10+ recommended
- A Bluetooth LE adapter
- An iDotMatrix-compatible device (often advertises as `IDM-*`)

Notes on Windows/BLE:
- If BLE scanning fails, pair the device in Windows Bluetooth settings first.
- If `IDOTMATRIX_ADDRESS=auto` doesn’t work, use the device MAC address instead.

## Setup (Windows / PowerShell)

From the repo root (this folder):

```powershell
# create & activate a venv
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Setup (Raspberry Pi / Raspberry Pi OS)

Tested target: Raspberry Pi 3 Model B (Raspberry Pi OS, Bluetooth via BlueZ).

### 1) System packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip bluetooth bluez
sudo systemctl enable --now bluetooth
```

### 2) Python venv + deps

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Configure env vars

Same variables as on Windows:

```bash
export LASTFM_API_KEY="<your key>"
export LASTFM_USER="<your username>"

# optional
export IDOTMATRIX_ADDRESS="auto"  # or a fixed MAC like AA:BB:CC:DD:EE:FF
```

You can also create a local `.env` file in the repo root.

### 4) Run

Use the Pi-friendly entrypoint:

```bash
python3 raspi_now_playing.py
```

### BLE permission notes (Linux / BlueZ)

If BLE scanning fails with a permission/DBus error, try one of:

- Run with sudo (quickest):

```bash
sudo -E python3 raspi_now_playing.py
```

- Set a fixed MAC instead of scanning:

```bash
export IDOTMATRIX_ADDRESS="AA:BB:CC:DD:EE:FF"
python3 raspi_now_playing.py
```

- Advanced: grant network caps to `python3` so scanning can work without sudo:

```bash
sudo setcap cap_net_raw,cap_net_admin+eip $(readlink -f $(which python3))
```

## Configuration

### iDotMatrix address

All root scripts support an environment variable:

- `IDOTMATRIX_ADDRESS`
  - `auto` (default in some scripts): scan for the first `IDM-*` device
  - or a MAC address like `AA:BB:CC:DD:EE:FF`

Example:

```powershell
$env:IDOTMATRIX_ADDRESS = "auto"
# or
$env:IDOTMATRIX_ADDRESS = "AA:BB:CC:DD:EE:FF"
```

### Last.fm

Set these environment variables:

- `LASTFM_API_KEY`
- `LASTFM_USER`

Example (PowerShell):

```powershell
$env:LASTFM_API_KEY = "<your key>"
$env:LASTFM_USER = "<your username>"
```

Alternatively, create a local `.env` file (this repo ignores it via `.gitignore`) based on `.env.example`.

### Clock overlay (now_playing.py)

`now_playing.py` can optionally draw a small clock on top of the album art before uploading.

- `SHOW_CLOCK`
  - `true` / `false` (default: `false`)

Optional tuning (only used when `SHOW_CLOCK=true`):

- `CLOCK_FORMAT` (default: `%I:%M`)
- `CLOCK_STRIP_LEADING_ZERO` (default: `true`)
- `CLOCK_POSITION` one of `top-left`, `top-right`, `bottom-left`, `bottom-right` (default: `bottom-right`)
- `CLOCK_RENDER` one of `tiny`, `default` (default: `tiny`)
- `CLOCK_FG` RGB like `255,255,255` (default: white)
- `CLOCK_BG` RGB like `0,0,0` (default: black)
- `CLOCK_PADDING` (default: `1`)
- `CLOCK_MARGIN` (default: `1`)

Example (PowerShell):

```powershell
$env:SHOW_CLOCK = "true"

# optional
$env:CLOCK_FORMAT = "%I:%M"   # or "%H:%M" for 24-hour
$env:CLOCK_POSITION = "bottom-right"
$env:CLOCK_RENDER = "tiny"
```

### Built-in device clock when idle (now_playing.py)

If no track is currently marked as “now playing”, `now_playing.py` can switch the iDotMatrix into its built-in clock mode.

- `USE_DEVICE_CLOCK_WHEN_IDLE`
  - `true` / `false` (default: `false`)
- `IDLE_TO_CLOCK_AFTER_SECONDS` (default: `30`)
  - How long to wait after the last seen now-playing track before switching to the built-in clock.
- `PLAYING_HOLD_SECONDS` (default: `180`)
  - Holds the last album art briefly to avoid flicker if Last.fm temporarily returns no now-playing track.

Clock appearance / behavior:

- `DEVICE_CLOCK_STYLE` (default: `3`)
- `DEVICE_CLOCK_WITH_DATE` (default: `false`)
- `DEVICE_CLOCK_24H` (default: `false`)
- `DEVICE_CLOCK_COLOR` RGB like `255,80,0` (default: orange)

Time sync (recommended when using built-in clock):

- `SYNC_DEVICE_TIME_ON_START` (default: `true`)
- `SYNC_DEVICE_TIME_ON_ENTER_CLOCK` (default: `true`)

## Usage

Activate your venv first:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Show now-playing album art

```powershell
python .\now_playing.py
```


## Troubleshooting

- **“Bluetooth error / BleakError”**
  - Try using a fixed MAC address instead of `auto`.
  - Make sure Bluetooth is enabled and you have a BLE-capable adapter.
 - **Album art looks banded / vertical stripes**
  - The dithering settings in the Last.fm script are meant to reduce visible banding.

## Credits

- Upstream client: `python3-idotmatrix-client/` (see its README for authorship/license).
- Root scripts in this repo are custom glue/scripts on top of the `idotmatrix` Python library.
