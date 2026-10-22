#!/bin/bash
# L16 Web Lab – nativní služba (GPU přes PyTorch MPS, Apple Vision). Spouští launchd:
# ~/Library/LaunchAgents/com.budhalabs.l16-native.plist
cd "$(dirname "$0")"
export L16_DATA_DIR="${L16_DATA_DIR:-$HOME/damocles_l16/data}"
exec /opt/homebrew/bin/uv run --frozen uvicorn l16native.app:app --host 127.0.0.1 --port 5310 --log-level warning
