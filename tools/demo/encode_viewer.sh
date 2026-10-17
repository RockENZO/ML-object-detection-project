#!/usr/bin/env bash
# Usage: tools/demo/encode_viewer.sh recording.webm docs/viewer-demo
# Produces <prefix>.mp4 (H.264) and <prefix>.gif (palette-optimised, README-friendly).
set -euo pipefail
input="$1"; prefix="$2"
ffmpeg -y -loglevel error -i "$input" -vf "scale=1100:-2,fps=20" -c:v libx264 -crf 29 -preset slow \
  -pix_fmt yuv420p -movflags +faststart "$prefix.mp4"
# The GIF is a 24 s excerpt (skipping the page load) to keep it README-sized.
ffmpeg -y -loglevel error -ss 3 -t 24 -i "$input" \
  -vf "fps=5,scale=700:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" \
  "$prefix.gif"
ls -lh "$prefix.mp4" "$prefix.gif"
