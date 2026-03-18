# IDotMatrix Now Playing — Home Assistant Add-on

A Home Assistant add-on that displays your currently playing track's album art on an [iDotMatrix](https://www.idotmatrix.com/) Bluetooth LED matrix, sourced from [Last.fm](https://www.last.fm/) scrobbling.

> This add-on is based on [elwinbb/IDotMatrix-Now-Playing](https://github.com/elwinbb/IDotMatrix-Now-Playing). All credit for the core Python logic goes to the original author.

---

## Prerequisites

- An **iDotMatrix** device (advertises over Bluetooth as `IDM-*`)
- A **Last.fm account** with scrobbling enabled for your music service (Spotify, Apple Music, etc.)
- A **Last.fm API key** — [create one here](https://www.last.fm/api/account/create)
- A Home Assistant host with a **Bluetooth adapter**

---

## Installation

1. In Home Assistant, go to **Settings → Add-ons → Add-on Store**
2. Click the menu (⋮) and choose **Repositories**
3. Add this repository URL and click **Add**
4. Find **IDotMatrix Now Playing** in the store and click **Install**

---

## Configuration

Open the add-on's **Configuration** tab. All options have descriptions in the UI.

### Required

| Option | Description |
|--------|-------------|
| `lastfm_api_key` | Your Last.fm API key |
| `lastfm_user` | Your Last.fm username |

### Device Connection

| Option | Default | Description |
|--------|---------|-------------|
| `idotmatrix_address` | `auto` | Device MAC address (e.g. `AA:BB:CC:DD:EE:FF`) or `auto` to scan |

Using a fixed MAC address is more reliable than `auto` and avoids the need for elevated Bluetooth scanning privileges. To find your device's address, scan from the host:

```bash
sudo bluetoothctl scan on
# Look for a device named IDM-* and note its address
```

### Clock Overlay

Optionally overlays the current time on top of album art.

| Option | Default | Description |
|--------|---------|-------------|
| `show_clock` | `false` | Enable clock overlay |
| `clock_format` | `%I:%M` | strftime format string |
| `clock_strip_leading_zero` | `true` | Show `9:05` instead of `09:05` |
| `clock_position` | `bottom-right` | `top-left`, `top-right`, `bottom-left`, `bottom-right` |
| `clock_render` | `tiny` | `tiny` (pixel-art) or `default` (PIL font) |
| `clock_fg` | `255,255,255` | Text colour (R,G,B) |
| `clock_bg` | `0,0,0` | Background colour (R,G,B) |
| `clock_padding` | `1` | Pixels of padding inside the clock box |
| `clock_margin` | `1` | Pixels between the clock box and the display edge |

### Idle / Device Clock

When no music is playing, the add-on can switch the device to its built-in clock.

| Option | Default | Description |
|--------|---------|-------------|
| `use_device_clock_when_idle` | `false` | Enable idle clock mode |
| `idle_to_clock_after_seconds` | `30` | Seconds of silence before switching |
| `device_clock_style` | `3` | Clock face index (0–9, device-dependent) |
| `device_clock_with_date` | `false` | Show date alongside time |
| `device_clock_24h` | `false` | 24-hour format |
| `device_clock_color` | `255,80,0` | Clock colour (R,G,B) |
| `sync_device_time_on_start` | `true` | Sync system time to device at startup |
| `sync_device_time_on_enter_clock` | `true` | Re-sync time each time clock mode is entered |

### Polling & Stability

| Option | Default | Description |
|--------|---------|-------------|
| `poll_interval` | `5` | Seconds between Last.fm checks |
| `playing_hold_seconds` | `180` | Keep showing last art this long after Last.fm stops reporting it |

---

## Troubleshooting

**Add-on can't find the Bluetooth device**
- Ensure Bluetooth is enabled on the host: `sudo systemctl status bluetooth`
- Set `idotmatrix_address` to the device's MAC address instead of `auto`
- Make sure no other app holds an exclusive BLE connection to the device

**Album art not updating**
- Confirm Last.fm is receiving scrobbles from your music service
- Double-check your Last.fm username (case-insensitive but must be exact)
- Lower `playing_hold_seconds` if stale art is staying on screen too long

**Clock overlay looks wrong**
- Use `clock_render: tiny` for a crisp look on the 32×32 display
- Adjust `clock_position`, `clock_padding`, and `clock_margin` to fine-tune placement

---

## Credits

Core application by [elwinbb](https://github.com/elwinbb/IDotMatrix-Now-Playing). This repository packages it as a Home Assistant add-on.
