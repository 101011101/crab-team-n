#!/bin/zsh
# The whole chain, video to map. ~30 min, all of it CPU.
set -e
cd "$(dirname "$0")"
V=../central/.venv/bin/python
WORK=${WORK:-work}
RESULTS=${RESULTS:-../results/world3}
NAMES=${NAMES:-IMG_7373,IMG_7374,IMG_7375}
mkdir -p "$RESULTS"

[ $# -ge 1 ] || { echo "usage: WORK=work $0 <video> [<video> ...]"; exit 1; }

echo "== stage 1  frames"
$V stage1_frames.py "$@" --out "$WORK/images"
echo "== stage 2  COLMAP  (the long one)"
$V stage2_sfm.py --images "$WORK/images" --out "$WORK"
echo "== stage 3  did they fuse?"
$V stage3_analyse.py "$WORK" --names "$NAMES" --out "$RESULTS/report.json"
echo "== stage 4  2D map"
$V stage4_map2d.py "$WORK/sparse" --out "$RESULTS/map2d.json" --png "$RESULTS/map2d.png"
echo "== stage 5  viewer blob"
$V stage5_pack.py "$WORK/sparse" --out "$WORK/world3_data.json" \
   --report "$RESULTS/report.json" --names "$NAMES"
echo
echo "done. results in $RESULTS"
