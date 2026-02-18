#!/bin/bash
# Generate PNG icons from SVG for browser extension
# Requires: ImageMagick (convert) or rsvg-convert

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ICONS_DIR="$SCRIPT_DIR/icons"

echo "Generating PNG icons..."

# Check for available tools
if command -v convert &> /dev/null; then
    # ImageMagick
    echo "Using ImageMagick..."
    convert -background none -resize 16x16 "$ICONS_DIR/icon16.svg" "$ICONS_DIR/icon16.png"
    convert -background none -resize 32x32 "$ICONS_DIR/icon16.svg" "$ICONS_DIR/icon32.png"
    convert -background none -resize 48x48 "$ICONS_DIR/icon16.svg" "$ICONS_DIR/icon48.png"
    convert -background none -resize 128x128 "$ICONS_DIR/icon16.svg" "$ICONS_DIR/icon128.png"
elif command -v rsvg-convert &> /dev/null; then
    # librsvg
    echo "Using rsvg-convert..."
    rsvg-convert -w 16 -h 16 -o "$ICONS_DIR/icon16.png" "$ICONS_DIR/icon16.svg"
    rsvg-convert -w 32 -h 32 -o "$ICONS_DIR/icon32.png" "$ICONS_DIR/icon16.svg"
    rsvg-convert -w 48 -h 48 -o "$ICONS_DIR/icon48.png" "$ICONS_DIR/icon16.svg"
    rsvg-convert -w 128 -h 128 -o "$ICONS_DIR/icon128.png" "$ICONS_DIR/icon16.svg"
else
    echo "Error: Neither ImageMagick nor librsvg found."
    echo "Install one of:"
    echo "  macOS:   brew install imagemagick"
    echo "  Ubuntu:  sudo apt install imagemagick"
    echo "  Or:      brew install librsvg"
    exit 1
fi

echo "Icons generated successfully:"
ls -la "$ICONS_DIR"/*.png