# Video capture guide

> The first test video failed. This states why, and how to shoot one that works.

## Result of the first attempt

`IMG_0300.MOV`, 1080x1920, 12.4 s, 54 frames sampled.

```
parallax     6 sideways, 25 forward, 0 rotate-only, 22 weak
epipolar     median E/H 1.18, forwardness 0.93
VERDICT      MARGINAL -- usable, but the camera walks forward
```

- The camera walked forward along corridors. That is real translation, and the clip can reconstruct.
- The baseline points along the view direction, which is the worst geometry for triangulating whatever is straight ahead. Strafing gives far more depth for the same walking.

### Correction

- The first version of this guide said the clip was rotation-only and could not reconstruct. That was wrong, and the tool was wrong with it.
- The parallax test used a homography residual. A homography absorbs nearly all of the motion in a corridor walk -- walls at similar depth, a distant vanishing point -- so the residual stayed small while the translation was real.
- The test now compares essential-matrix inliers against homography inliers, which a rotation cannot fake. Verified against three synthetic clips of known motion, and against this clip frame by frame.

## Why rotation gives nothing

- Depth comes from **parallax**: the same point seen from two *different positions* shifts by an amount set by its distance. Near things shift more. That shift is the only depth signal a single camera has.
- Rotate in place and every point shifts by the same amount, near or far. The shift encodes the turn, not the distance.
- A GPU cannot fix this. The information is absent from the file.

## Forward motion is weak, not useless

- Walking forward is real translation and does reconstruct.
- But the baseline lies along the view direction, so a point straight ahead barely shifts between frames. Depth there is poorly conditioned.
- Points off to the sides shift properly. A forward walk therefore reconstructs the walls beside it far better than the corridor in front of it.
- Strafing costs the same effort and gives depth everywhere. Prefer it.

## How to shoot a video that reconstructs

- **Walk sideways past the scene.** Do not stand still and sweep.
- Point the camera at one region and **strafe** across it, keeping that region in frame.
- Travel at least 1 to 2 m while the same wall or object stays in view.
- Move slowly. About one third of walking pace.
- 30 to 60 s is enough. More frames of the same mistake do not help.
- Hold the phone at robot height, about 10 cm, if you want the honest test.
- Bright, even light. Shade the scene from a single harsh lamp.

### Sanity check

- Pick a near object and a far object in the same shot.
- Move. If the near one slides across the far one, you have parallax.
- If they move together, you are rotating.

## Verify before you hand it over

```
./central/.venv/bin/python tools/video3d/feasibility.py <video> --degrade --no-per-frame
```

- `sideways` should outnumber `forward`.
- `rotate-only` should be near zero.
- `median E/H` above 1.15 means the motion contains translation a rotation cannot explain.
- The check costs about a minute. Run it before handing a clip over.

## What to shoot

| Clip | Purpose |
| --- | --- |
| Phone, strafing, arena walls and a block | Best-case test. If this fails, the approach fails. |
| Phone, robot height, same path | Tests whether a low viewpoint is the problem. |
| Robot camera, same path | The real test. Compare with the phone clip to separate a camera problem from a code problem. |

- Shoot the phone clips first. The robot clip is only worth the effort once a phone clip passes.

## Reading the degraded comparison

- `--degrade` reruns every metric on a 320x240 JPEG-q12 copy, approximating the robot camera.
- Compare texture, trackability and parallax across the two columns. The drop is the margin lost going from phone to robot camera.
- Blur is not comparable across resolutions. Variance of the Laplacian rises when an image is downscaled, because edge energy concentrates. Read blur within one column only.
