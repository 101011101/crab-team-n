# How big to print the tags

> Measured 2026-09-19 with `central/tag_probe.py`'s solver, 1280x720 at 60 deg horizontal, 200 trials per row at 0.3 px corner noise, off-axis angle uniform over +/-30 deg.

## The rule

**Print about 10 cm of tag for every metre of range you need.**

Accuracy tracks the tag's APPARENT SIZE IN PIXELS, not its printed size. Below roughly 100 px across, planar pose becomes ill-conditioned and the error grows fast.

```
apparent_px  =  focal_px * tag_metres / range_metres
```

## Measured

| tag | range | apparent | median error | p90 | refused |
| --- | --- | --- | --- | --- | --- |
| 100 mm | 1 m | 111 px | 1.7 cm | 7.0 cm | 1/200 |
| 100 mm | 2 m | 48 px | 8.0 cm | 45.7 cm | 23/200 |
| 100 mm | 3 m | 38 px | 18.0 cm | 132.4 cm | 45/200 |
| 150 mm | 1 m | 163 px | 1.2 cm | 3.5 cm | 0/200 |
| 150 mm | 2 m | 82 px | 4.7 cm | 18.8 cm | 4/200 |
| 150 mm | 3 m | 55 px | 11.8 cm | 57.0 cm | 22/200 |
| 200 mm | 1 m | 217 px | 0.8 cm | 2.1 cm | 0/200 |
| 200 mm | 2 m | 110 px | 3.2 cm | 11.3 cm | 2/200 |
| 200 mm | 3 m | 70 px | 7.8 cm | 29.7 cm | 10/200 |

- With EXACT corners every row solves to under 1 mm, so the solver is not the limit. Corner noise is.
- The p90 column matters more than the median. A 100 mm tag at 2 m is 8 cm typical but 46 cm at the tail, and a robot driving on the tail is a robot in a wall.
- "Refused" is the ambiguity gate firing. Those are not failures: a refused pose is the system saying it cannot tell, which is the correct answer and far better than a confident wrong one.

## What to do

- **Arena up to 2 m:** 150 mm tags, comfortably.
- **Arena up to 3 m:** 200 mm tags, or accept ~8 cm.
- **ESP32-CAM at 320x240** has roughly a quarter of the focal length in pixels of the test camera, so the same accuracy needs roughly **four times the tag size** at the same range. A 100 mm tag at 1 m on the robot camera behaves like a 100 mm tag at 4 m here. Print big, or keep the tags close.
- More tags in view beats bigger tags: `markers.py` fuses jointly and measured a 4 to 6 times gain from multi-tag over single-tag. Cover the walls rather than perfecting one tag.

## Caveats

- 0.3 px corner noise is a reasonable figure for sub-pixel refinement on a sharp image. Motion blur makes it far worse, and blur is the thing a vibrating robot chassis produces.
- This is a synthetic sweep with a perfect camera model. It bounds the best case; a real uncalibrated camera measured 16.5 cm against 1.80 cm calibrated.
