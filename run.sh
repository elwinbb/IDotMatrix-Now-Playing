#!/usr/bin/with-contenv bashio

bashio::log.info "Starting IDotMatrix Now Playing..."

# Read required options
LASTFM_API_KEY=$(bashio::config 'lastfm_api_key')
LASTFM_USER=$(bashio::config 'lastfm_user')

if bashio::var.is_empty "${LASTFM_API_KEY}"; then
    bashio::log.fatal "lastfm_api_key is required. Set it in the add-on configuration."
    exit 1
fi

if bashio::var.is_empty "${LASTFM_USER}"; then
    bashio::log.fatal "lastfm_user is required. Set it in the add-on configuration."
    exit 1
fi

# Export all options as environment variables for now_playing.py
export LASTFM_API_KEY="${LASTFM_API_KEY}"
export LASTFM_USER="${LASTFM_USER}"
export IDOTMATRIX_ADDRESS=$(bashio::config 'idotmatrix_address')
export POLL_INTERVAL=$(bashio::config 'poll_interval')
export SHOW_CLOCK=$(bashio::config 'show_clock')
export CLOCK_FORMAT=$(bashio::config 'clock_format')
export CLOCK_STRIP_LEADING_ZERO=$(bashio::config 'clock_strip_leading_zero')
export CLOCK_POSITION=$(bashio::config 'clock_position')
export CLOCK_RENDER=$(bashio::config 'clock_render')
export CLOCK_FG=$(bashio::config 'clock_fg')
export CLOCK_BG=$(bashio::config 'clock_bg')
export CLOCK_PADDING=$(bashio::config 'clock_padding')
export CLOCK_MARGIN=$(bashio::config 'clock_margin')
export USE_DEVICE_CLOCK_WHEN_IDLE=$(bashio::config 'use_device_clock_when_idle')
export DEVICE_CLOCK_STYLE=$(bashio::config 'device_clock_style')
export DEVICE_CLOCK_WITH_DATE=$(bashio::config 'device_clock_with_date')
export DEVICE_CLOCK_24H=$(bashio::config 'device_clock_24h')
export DEVICE_CLOCK_COLOR=$(bashio::config 'device_clock_color')
export SYNC_DEVICE_TIME_ON_START=$(bashio::config 'sync_device_time_on_start')
export SYNC_DEVICE_TIME_ON_ENTER_CLOCK=$(bashio::config 'sync_device_time_on_enter_clock')
export IDLE_TO_CLOCK_AFTER_SECONDS=$(bashio::config 'idle_to_clock_after_seconds')
export PLAYING_HOLD_SECONDS=$(bashio::config 'playing_hold_seconds')

bashio::log.info "Connecting to iDotMatrix device at: ${IDOTMATRIX_ADDRESS}"
bashio::log.info "Polling Last.fm for user: ${LASTFM_USER} every ${POLL_INTERVAL}s"

exec python3 /app/now_playing.py
