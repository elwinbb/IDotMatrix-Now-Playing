# IDot Matrix (Last.fm + iDotMatrix scripts)

This repo is a small “toolbox” for driving an iDotMatrix 16×16 / 32×32 pixel display over Bluetooth LE, with a focus on showing **Last.fm “now playing” album art**.

## Repository layout

- `lastfm_idotmatrix.py`
  - Polls Last.fm for the current track and uploads the album art to the device.
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

## Usage

Activate your venv first:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Show now-playing album art

```powershell
python .\lastfm_idotmatrix.py
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
