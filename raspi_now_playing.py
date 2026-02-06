#!/usr/bin/env python3
"""Raspberry Pi entrypoint for this repo.

This script is intentionally thin: it reuses the existing logic in `now_playing.py`
but adds Raspberry Pi / Linux Bluetooth (BlueZ) preflight hints.

Usage:
  python3 raspi_now_playing.py

Required env vars (same as now_playing.py):
  - LASTFM_API_KEY
  - LASTFM_USER

Optional:
  - IDOTMATRIX_ADDRESS=auto (default) or a MAC like AA:BB:CC:DD:EE:FF
"""

from __future__ import annotations

import asyncio
import os
import platform
import sys
from pathlib import Path


def _maybe_load_dotenv() -> None:
    """Load `.env` from the repo root if python-dotenv is installed."""

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    dotenv_path = Path(__file__).with_name(".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _print_pi_hints() -> None:
    if platform.system().lower() != "linux":
        return

    machine = (platform.machine() or "").lower()
    if not ("arm" in machine or "aarch" in machine):
        # Still could be Linux on another device, so just keep it quiet.
        return

    addr = os.environ.get("IDOTMATRIX_ADDRESS", "auto")
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0  # type: ignore[attr-defined]

    if str(addr).lower() == "auto" and not is_root:
        print(
            "Note (Raspberry Pi / BlueZ): BLE scanning may require extra permissions. "
            "If you get a Bleak/DBus permission error, try one of:\n"
            "  - run with sudo: sudo python3 raspi_now_playing.py\n"
            "  - or set a fixed MAC: export IDOTMATRIX_ADDRESS=AA:BB:CC:DD:EE:FF\n"
            "  - or grant caps to python3 (advanced):\n"
            "      sudo setcap cap_net_raw,cap_net_admin+eip $(readlink -f $(which python3))\n"
        )


def main() -> int:
    _maybe_load_dotenv()
    _print_pi_hints()

    # Import after dotenv so `now_playing.py` sees env vars at import time.
    import now_playing  # noqa: F401

    try:
        asyncio.run(now_playing.main_async())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
