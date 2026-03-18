# IDotMatrix Now Playing

Displays the currently playing track's album art on an [iDotMatrix](https://www.idotmatrix.com/) Bluetooth LED matrix, sourced from [Last.fm](https://www.last.fm/) scrobbling.

## Prerequisites

- An **iDotMatrix** device (broadcasts over Bluetooth LE as `IDM-*`)
- A **Last.fm account** with scrobbling enabled for your music service (Spotify, Apple Music, etc.)
- A **Last.fm API key** – [create one here](https://www.last.fm/api/account/create)
- The Home Assistant host must have a **Bluetooth adapter** accessible to the add-on

## Configuration

### Required

| Option | Description |
|--------|-------------|
| `lastfm_api_key` | Your Last.fm API key |
| `lastfm_user` | Your Last.fm username |

### Device Connection

| Option | Default | Description |
|--------|---------|-------------|
| `idotmatrix_address` | `auto` | Bluetooth MAC address (`AA:BB:CC:DD:EE:FF`) or `auto` to scan |

Using `auto` causes the add-on to scan for nearby iDotMatrix devices on each start. This requires additional Bluetooth privileges. Setting a fixed MAC address is more reliable and avoids permission issues.

To find your device's MAC address:
```bash
sudo bluetoothctl scan on
# Look for a device named IDM-* and note its address
```

### Clock Overlay

Optionally overlay the current time on top of album art.

| Option | Default | Description |
|--------|---------|-------------|
| `show_clock` | `false` | Enable clock overlay |
| `clock_format` | `%I:%M` | strftime format (`%H:%M` for 24-hour) |
| `clock_strip_leading_zero` | `true` | Remove leading zero from hour |
| `clock_position` | `bottom-right` | `top-left`, `top-right`, `bottom-left`, or `bottom-right` |
| `clock_render` | `tiny` | `tiny` (pixel-art) or `default` (PIL font) |
| `clock_fg` | `255,255,255` | Text color (R,G,B) |
| `clock_bg` | `0,0,0` | Background color (R,G,B) |
| `clock_padding` | `1` | Pixels of padding around the text |
| `clock_margin` | `1` | Pixels between the clock box and display edge |

### Idle / Device Clock

When no music is playing, the add-on can switch the device into its built-in clock mode.

| Option | Default | Description |
|--------|---------|-------------|
| `use_device_clock_when_idle` | `false` | Enable idle clock mode |
| `idle_to_clock_after_seconds` | `30` | Seconds of silence before switching |
| `device_clock_style` | `3` | Clock face index (device-dependent, 0–9) |
| `device_clock_with_date` | `false` | Show date alongside time |
| `device_clock_24h` | `false` | 24-hour format on device clock |
| `device_clock_color` | `255,80,0` | Clock color (R,G,B) |
| `sync_device_time_on_start` | `true` | Sync system time to device on startup |
| `sync_device_time_on_enter_clock` | `true` | Sync time each time clock mode is entered |

### Polling & Stability

| Option | Default | Description |
|--------|---------|-------------|
| `poll_interval` | `5` | Seconds between Last.fm checks |
| `playing_hold_seconds` | `180` | Keep showing last art this long after Last.fm stops reporting it |

## Troubleshooting

### Add-on can't find Bluetooth device
- Ensure Bluetooth is enabled on the host: `sudo systemctl status bluetooth`
- Try setting `idotmatrix_address` to the device's MAC address instead of `auto`
- Make sure no other app has an exclusive connection to the iDotMatrix device

### Album art not updating
- Verify Last.fm is receiving scrobbles from your music service
- Check that your Last.fm username is spelled correctly (case-insensitive)
- The `playing_hold_seconds` setting may be keeping old art displayed; reduce it if needed

### Clock overlay looks wrong
- Use `clock_render: tiny` for a crisp pixel-art look on the 32×32 display
- Adjust `clock_position`, `clock_padding`, and `clock_margin` to fine-tune placement
