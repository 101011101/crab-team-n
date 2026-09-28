# video3d — T4

Three tools and a library. Run them in this order.

- `feasibility.py` — no model, no download. Reports blur, texture, trackability and parallax, then a verdict. Runs in `central/.venv`.
- `calibrate_fov.py` — measures focal length from the footage with vanishing points. No model. Runs in `central/.venv`.
- `reconstruct.py` — Depth Anything V2 Small on MPS. Writes a `.ply` and a preview PNG. Needs its own venv.
- `corridor.py` — library. Scores a cloud against a corridor's flat, parallel, square surfaces. numpy only.
- `synth_corridor.py` — renders a corridor with a KNOWN focal length and checks `calibrate_fov.py` recovers it. Runs in `central/.venv`.

## Install

    # feasibility only: nothing to install
    cd central && ./.venv/bin/python ../tools/video3d/feasibility.py <video>

    # reconstruct
    cd tools/video3d
    python3 -m venv .venv
    ./.venv/bin/pip install -r requirements.txt

Do not put `requirements.txt` into `central/`. It unpacks to 1.0 GB.

## Run

    ./.venv/bin/python feasibility.py ../../../IMG_0300.MOV --degrade
    ./.venv/bin/python reconstruct.py ../../../IMG_0300.MOV --out out/

Useful flags:

- `--degrade` — both tools. Pushes frames through the ESP32-CAM losses (320x240, real JPEG round trip at quality 12) before measuring. This answers "does it survive OUR camera", not "does the method work".
- `--single` — reconstruct one frame only. Skips all multi-frame alignment.
- `--baseline-m` — asserted camera travel between sampled frames. Default 0.
- `--focal-px` — focal length in pixels, from `calibrate_fov.py`. Overrides `--hfov-deg`.
- `--solve-fov` — measure the focal length in-process instead of assuming it.
- `--beta-frac` — affine offset of the depth model. Default 0, the old hard-coded assumption.
- `--keep-depth-pct` — drop points past this percentile of depth. Default 100, off.
- `--denoise N` — drop points whose voxel holds fewer than N. Default 0, off.
- `--report-geometry` — score the cloud against a corridor's known flat, parallel, square surfaces.
- `--device cpu` — compare against MPS.
- `make_test_video.py` writes `translate.mp4`, `rotate.mp4` and `static.mp4`: same scene, known camera path, only one reconstructable.

## Measured

Apple Silicon, macOS 27, Python 3.13, torch 2.14, transformers 5.17. Measured 2026-09-19.

| Item | Measured |
| --- | --- |
| Model weights | 95 MB, one download to `~/.cache/huggingface` |
| Parameters | 24.8 M |
| `.venv` on disk | 1.0 GB |
| `pip install` | 31 s from a warm cache |
| Model load | 3.0 s |
| First inference | 1.1 s. MPS graph build. Excluded from all numbers below |

Depth inference, median per frame, after warm-up:

| Source frame (WxH) | Network input (WxH) | MPS | CPU |
| --- | --- | --- | --- |
| 320x240 (robot) | 518x392 | **92 ms (10.9 fps)** | 262 ms (3.8 fps) |
| 640x480 | 686x518 | 209 ms (4.8 fps) | not measured |
| 1080x1920 (phone) | 518x924 | 270 ms (3.7 fps) | not measured |

- At QVGA the split is 5 ms preprocess, 96 ms forward, 0.7 ms resample.
- Cost tracks the NETWORK input, not the source. The processor rescales to 518 on one side and keeps the aspect, so a 320x240 frame is UPSCALED to 518x392. Feeding the model smaller frames buys almost nothing below QVGA.
- MPS is 2.8x CPU. Both are single-stream; no batching was measured.

End to end, 8 frames of `IMG_0300.MOV --degrade`, point stride 1: 566,254 points, 8.5 MB `.ply`, 22 s wall including model load.

**Decision this supports:** 11 fps depth at QVGA on one laptop, for one camera. Two robots at 5 fps each is viable. Depth on every frame of every robot at full rate is not.

## What was verified

- `feasibility.py` runs on the real phone video (`IMG_0300.MOV`, 1080x1920, 12.4 s) and on all three synthetic clips. 7.5 s for a source plus degraded run (was 6.5 s before the reference-grid pass was added).
- Depth output is qualitatively correct on a real 320x240 corridor frame: floor gradient, two side walls, far end dark. See the preview PNG.
- `.ply` writes, reloads and renders. The corridor is recognisable as a corridor from two angles.
- MPS and CPU timings above, warm-up excluded and `torch.mps.synchronize()` called before every stop.

## What is NOT verified

- **Multi-frame geometry.** `IMG_0300.MOV` translates forward, not sideways, and its baseline is ASSUMED by `--baseline-m`, never solved. The stitch has never run against ground truth. Treat any multi-frame cloud as a depth shell from roughly one viewpoint, not as solved geometry.
- **Scale.** Nothing here is metric. Depth Anything V2 returns affine-invariant inverse depth. `to_depth` pins the nearest surface to `--near-m` and ASSUMES the affine offset is zero. Now measurable, and measured below: the assumption is close to correct.
- **Field of view.** `--hfov-deg` defaults to 60 and is a guess. Now measurable with `calibrate_fov.py`. Two independent methods disagree. See below; unresolved.
- **Real ESP32-CAM footage.** Never tested. `--degrade` is an approximation, and it squashes the source aspect into 4:3 instead of cropping, so per-pixel distances are distorted between the two runs.
- **Tag size and range.** Nothing here measures how far a marker stays detectable. The 10 to 20 cm figure in the docs is a prior, not a measurement.

## Blur was measured wrong (fixed)

Symptom: `--degrade` on the phone video reported blur 19.2 -> 103.8, "+440%". A degraded copy cannot be sharper than its source.

Cause: variance of the Laplacian is resolution dependent. Downscaling averages pixels away and concentrates the surviving edge energy, so the small copy always scores sharper. Same class of error on ORB count (rises with pixel count), and on flow and parallax (reported in pixels, which shrink with the image).

Fix:

- Blur and ORB are measured a second time on one common grid, `REF_DIAG` = the QVGA diagonal, 400 px, aspect preserved. Those measurements are what the comparison differences and what the absolute thresholds are applied to.
- Native values are still printed, labelled, and never differenced. The comparison table shows them below a rule with `--` in the delta columns.
- Flow and parallax are differenced as per mille of the diagonal.
- Any metric that IMPROVES under damage is marked `(*)` in place and explained under the table.

After the fix, on the same video: blur 187.0 -> 103.8, -44%.

The `(*)` marks caught a second real effect: on one grid, JPEG quality 12 raises the ORB count from 149 to 362 while sharpness falls. Block edges are being detected as corners. Those keypoints do not match — trackability falls 73% and RANSAC survival falls 8 points over the same step. Keypoint count is not a texture measure on compressed video. Trackability is.

## The parallax test was wrong, and so was the first verdict (fixed)

The first run on `IMG_0300.MOV` returned DO NOT ATTEMPT, on the claim that the camera never translated. That was false. The clip walks forward along corridors, confirmed by looking at its frames and by recovering the motion with `cv2.recoverPose`: t is roughly [0, +0.6, -0.8], forward dominant.

Cause: the test measured parallax as the residual a single homography cannot explain. A homography models a rotation exactly, but it also absorbs nearly all of a corridor walk — walls at similar depth, a distant vanishing point — so the residual stayed small while the translation was perfectly real. The same bug rejected `translate.mp4`, whose sideways camera path is known by construction.

Fix: compare essential-matrix inliers against homography inliers. A homography cannot model translation unless the scene is one plane, so it only loses ground when translation is present. The ratio is a comparison between two fits of the same points, so a wrong assumed focal length shifts both sides and largely cancels.

| clip | ground truth | E/H | verdict |
| --- | --- | --- | --- |
| `translate.mp4` | sideways | 1.51 | WORTH ATTEMPTING |
| `rotate.mp4` | rotation only | 1.01 | DO NOT ATTEMPT |
| `static.mp4` | still | 0.00 | DO NOT ATTEMPT |
| `IMG_0300.MOV` | forward walk | 1.18 | MARGINAL |

## Verdict on `IMG_0300.MOV`

MARGINAL: 6 pairs sideways, 25 forward, 0 rotation only, 22 weak. Forward motion is real translation and does reconstruct, but the baseline points along the view direction, so whatever is straight ahead barely shifts between frames and triangulates poorly. The side walls come out far better than the corridor ahead. A strafing clip costs the same walking and gives depth everywhere.

`reconstruct.py` produces a plausible cloud regardless, because a monocular depth model does not need parallax — it guesses from learned priors. That is exactly why its output must not be read as a measurement.

## Focal length: measured, and disputed

`calibrate_fov.py` measures focal length from the footage. It finds vanishing points with RANSAC over LSD line segments, then uses `f^2 = -(v1-c).(v2-c)` on every orthogonal pair. Runs in `central/.venv`.

    ../../central/.venv/bin/python calibrate_fov.py ../../../IMG_0300.MOV

Measured on `IMG_0300.MOV`, 40 frames:

| Quantity | Value |
| --- | --- |
| focal | **1641 px**, 95% bootstrap CI [1598, 1663], MAD 47 |
| from | 51 orthogonal pairs over 37 of 40 frames |
| FOV across 1080 px (image WIDTH) | 36.4 deg |
| FOV across 1920 px (image HEIGHT) | 60.7 deg |
| 35 mm-equivalent, long axis | 31 mm |

60 frames at a finer stride give 1655 px, CI [1642, 1671]. The two runs agree.

**The estimator was validated against ground truth.** `synth_corridor.py` renders a corridor with a known `f` and known camera pitch, then asks the estimator to recover it. Reproduce with:

    ../../central/.venv/bin/python synth_corridor.py


| true f | recovered, pitch 10 deg | pitch 18 deg |
| --- | --- | --- |
| 935 | 937 (+0.2%) | 936 (+0.1%) |
| 1200 | 1199 (-0.1%) | 1201 (+0.1%) |
| 1655 | 1656 (+0.1%) | 1687 (+1.9%) |
| 2000 | 1999 (-0.1%) | 2070 (+3.5%) |

The estimator is not biased toward large values: given a true 935 px it returns 936, not 1650.

### The disagreement

COLMAP, run separately on 124 frames of the same clip, bundle-adjusted to **982.8 px** (HFOV 57.6 deg across 1080), which would make the shipped 60 deg default nearly right. The vanishing-point method says 1641. The two differ by 1.67x. This is NOT resolved. Evidence on each side:

For ~1650 px:
- Validated against synthetic ground truth to within 2% (table above), including at 935 px.
- An independent focal sweep over `cv2.findFundamentalMat`, scoring how close `K^T F K` is to a true essential matrix, put the minimum at **1690 px**.
- Floor-to-wall squareness, measured on a fixed wall pixel set, falls from 2.99 deg at f=935 to 1.01 deg at f=1655 and flattens at 0.83 deg by f=1900. It rejects 935 and prefers anything above ~1400.
- 1650 px is 31 mm-equivalent, 67 deg diagonal. That is right for an iPhone shooting stabilised 1080p, which crops. 982.8 px is 18 mm-equivalent, 97 deg diagonal, which is neither the main camera (26 mm) nor the ultra-wide (13 mm).

For ~983 px:
- Bundle adjustment over 124 registered frames at 1.245 px mean reprojection error.

Two reasons to distrust the COLMAP number despite that:
- **Forward motion is a degenerate configuration for focal self-calibration.** With the baseline along the optical axis, focal length trades off against depth and the error surface is nearly flat in `f`. A low reprojection error does not imply a constrained focal length.
- **That COLMAP run's input directory was contaminated, by this work.** 124 frames extracted here at every 3rd frame were left in the shared directory and collided by filename with a second, differently spaced extraction. All 207 surviving images are portrait 1080x1920, so orientation is consistent, but the mean absolute difference between filename-consecutive frames is **16.3, against 8.1 for true temporal order** — larger despite there being more frames. Filename order no longer follows time, and `sequential_matcher` matches on filename adjacency. This plausibly also explains that run's collapse in structure (21, then 9, then 0 points per frame).

**Do not settle this from the numbers above.** Re-run COLMAP on a clean, uniformly sampled directory. If it still returns ~983 px with good structure, the vanishing-point estimate is wrong and should be removed.

### What the 60 deg default actually got wrong

If ~1650 px is right, the default was not a bad guess about the lens. It was an axis mix-up. The measured FOV across the LONG axis is 60.7 deg. `intrinsics()` applied 60 deg to `width`, which on this portrait video is the SHORT axis, giving f=935 instead of 1655. The number 60 was very nearly correct; it was applied to the wrong edge of a rotated frame. `cv2.VideoCapture` auto-applies the container's `rotation=-90`, so a 1920x1080 stream arrives as 1080x1920 and any FOV default written for landscape is silently wrong.

## Affine offset: measured, small, not worth changing

`--beta-frac` exposes the offset `to_depth` used to hard-code. `d = alpha/Z + beta`, so `Z = alpha/(d - beta)`; `beta-frac 0` puts beta at the 2nd percentile of `d`, which is the shipped assumption.

Scored by wall bow, on a wall pixel set chosen ONCE and then re-measured under each offset, so that the comparison is over the same piece of wall:

| reference set chosen at | best beta-frac | bow there | bow at 0 |
| --- | --- | --- | --- |
| 0.0 | 0.00 | 1.049% | 1.049% |
| 0.1 | 0.05 | 0.561% | 0.846% |
| 0.3 | 0.05 | 0.477% | 0.870% |

The winner moves with the reference set, so this measurement carries selection bias: pixels chosen as planar under one offset look planar under it. Two of the three references nevertheless agree on ~0.05, which is weak evidence for a small POSITIVE offset.

**The default is unchanged at 0.** The effect is small, its sign is only weakly established, and the honest reading is that the zero-offset assumption is not what is bowing these walls. This is a smaller error than the focal length by a wide margin.

## Wall flatness as a metric, and its limits

`corridor.py` scores a cloud against three things a corridor is known to have: flat walls, parallel walls, floors square to walls. All three survive an unknown scale. `reconstruct.py --report-geometry` prints them.

**Two of the three cannot select a focal length.** On a fixed wall pixel set, bow and wall-parallelism both fall monotonically as `f` rises:

| f | 700 | 935 | 1200 | 1400 | 1655 | 1900 | 2300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bow % | 2.18 | 1.83 | 1.44 | 1.24 | 1.05 | 0.92 | 0.76 |
| wall-wall deg | 22.8 | 17.1 | 13.4 | 11.5 | 10.0 | 8.9 | 7.4 |
| floor-wall deg | 4.50 | 2.99 | 1.80 | 1.34 | 1.01 | 0.84 | 0.83 |

Raising `f` scales x and y by 1/f and leaves z alone, so the whole cloud flattens toward a sliver and every flatness and parallelism number improves for free. Only floor-to-wall squareness has a real interior minimum, and even that one is shallow above 1900. An earlier version of this measurement re-segmented the walls per configuration and appeared to show an interior minimum in bow at f=1655; that was an artefact of the point set changing, and it is withdrawn.

## Outlier and tail filtering

Depth distribution over 12 frames at f=1655, 2.5 M points:

| band | share |
| --- | --- |
| 0-1 m | 55.8% |
| 1-2 m | 25.8% |
| 2-3 m | 6.9% |
| 3-4 m | 4.0% |
| 4-6 m | 4.2% |
| 6-8 m | 1.6% |
| 8-12 m | 1.8% |

81.6% of points sit inside 2 m. The 3.4% beyond 6 m double the extent of the cloud. That tail is the model's far field, where very few disparity values are stretched over many metres, and it is the least trustworthy part of the output.

- `--keep-depth-pct 97` cuts at the 97th percentile of depth, 6.37 m. Drops 75,155 of 2,505,144 points (3.0%) and nearly halves the extent.
- `--denoise 3` drops points whose voxel holds fewer than 3. Removed a further 105,751 points (4.3%). Targets the speckle thrown off depth discontinuities at doorways and light fittings. Occupancy of a voxel grid, not a k-NN test, because scipy is deliberately absent from both venvs.
- Points with z <= 0 are now dropped. Previously only the far clamp was.

This is a readability change, not an accuracy change. No claim is made that the removed points were wrong, only that they were sparse and far.

## Before and after

Both runs use `--baseline-m 0`, the default, so the multi-frame stitch is identical between them and the comparison isolates the changes above.

    ./.venv/bin/python reconstruct.py ../../../IMG_0300.MOV --out out/baseline --hfov-deg 60 --report-geometry
    ./.venv/bin/python reconstruct.py ../../../IMG_0300.MOV --out out/improved \
        --focal-px 1655 --keep-depth-pct 97 --denoise 3 --report-geometry

| | baseline (f=935) | improved (f=1655, filtered) |
| --- | --- | --- |
| floor-wall off square | 2.6 deg | **1.3 deg** |
| wall-wall off parallel | 19.5 deg | 10.2 deg (*) |
| wall bow | 1.83% | 2.19% (*) |
| depth extent | 12.0 m | 6.4 m |
| points | 1.5 M (capped) | 1.5 M (capped) |

(*) These two move with `f` for free, as the table further up shows. Only the floor-to-wall figure is evidence.

Files: `out/baseline/IMG_0300_cloud.ply`, `out/improved/IMG_0300_cloud.ply`, and a `_preview.png` beside each.

Visually the improved cloud loses the speckle halo and the long sparse tail, and the oblique view resolves a clean wall-floor junction. It is also NARROWER, which is what a longer focal length does and is the point.

## What is still wrong

- **The focal length is disputed.** Everything in the improved run is conditional on ~1650 px being right. Resolve it before trusting the cloud.
- **The multi-frame stitch is still `--baseline-m`, still asserted.** Not addressed here. COLMAP's own structure on this clip collapsed to 14 points over 28 frames, so the footage may not support a solved trajectory at all, contaminated input or not.
- **Forward motion.** The clip walks down the corridor instead of strafing across it, so the baseline lies along the view direction and what is straight ahead barely moves between frames. This is the root problem and no amount of post-processing fixes it. A strafing clip costs the same to shoot.
- **The footage is feature-poor.** SIFT at default contrast threshold finds a median of 350 keypoints on a 2 MP frame, ranging down to 128. var(Laplacian) has a median of 17 native. Lowering the threshold to 0.01 raises the median to 2925, which suggests the extractor settings, not only the footage, limit what any SfM gets.
- **Nothing here is metric**, and no monocular method makes it so.
