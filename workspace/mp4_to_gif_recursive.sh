#!/usr/bin/env bash

set -euo pipefail

# Recursively convert all MP4 files under a directory to high-quality GIFs.
#
# Usage:
#   ./mp4_to_gif_recursive.sh /path/to/folder
#
# Optional environment variables:
#   FPS=20 WIDTH=960 OVERWRITE=1 ./mp4_to_gif_recursive.sh /path/to/folder
#
# Defaults:
#   FPS=20
#   WIDTH=960
#   OVERWRITE=0

ROOT_DIR="${1:-.}"
FPS="${FPS:-20}"
WIDTH="${WIDTH:-960}"
OVERWRITE="${OVERWRITE:-0}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg is not installed or not available in PATH."
    echo "Install it first, for example:"
    echo "  sudo apt update && sudo apt install ffmpeg"
    exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "Error: '$ROOT_DIR' is not a directory."
    exit 1
fi

echo "Searching for MP4 files under: $ROOT_DIR"
echo "FPS: $FPS"
echo "WIDTH: $WIDTH"
echo "OVERWRITE: $OVERWRITE"
echo

find "$ROOT_DIR" -type f -iname "*.mp4" -print0 | while IFS= read -r -d '' MP4_FILE; do
    DIRNAME="$(dirname "$MP4_FILE")"
    BASENAME="$(basename "$MP4_FILE")"
    STEM="${BASENAME%.*}"
    GIF_FILE="$DIRNAME/$STEM.gif"

    if [[ -f "$GIF_FILE" && "$OVERWRITE" != "1" ]]; then
        echo "Skipping existing GIF:"
        echo "  $GIF_FILE"
        echo
        continue
    fi

    echo "Converting:"
    echo "  Input : $MP4_FILE"
    echo "  Output: $GIF_FILE"

    PALETTE_FILE="$(mktemp --suffix=.png)"

    # High-quality GIF conversion:
    # 1. Generate an optimized color palette from the video.
    # 2. Use that palette to produce a cleaner GIF with better colors.
    #
    # lanczos scaling gives good visual quality.
    # dither=sierra2_4a usually gives good GIF results.
    ffmpeg -y \
        -nostdin \
        -i "$MP4_FILE" \
        -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,palettegen=stats_mode=full" \
        "$PALETTE_FILE"

    ffmpeg -y \
        -nostdin \
        -i "$MP4_FILE" \
        -i "$PALETTE_FILE" \
        -filter_complex "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=sierra2_4a" \
        -loop 0 \
        "$GIF_FILE"

    rm -f "$PALETTE_FILE"

    echo "Done."
    echo
done

echo "All conversions finished."
