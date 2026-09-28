# COLMAP on the first phone clip

> Run 2026-09-19 on `IMG_0300.MOV`. COLMAP 4.2.0, Homebrew bottle, CPU only (no CUDA on this Mac).

## Setup

- 124 frames, every 3rd frame of 372, 1080x1920, JPEG q95.
- `--ImageReader.single_camera 1`, SIMPLE_RADIAL, `--FeatureExtraction.max_image_size 1600`.
- `sequential_matcher`, overlap 10. Then `mapper`.

## Timing, CPU only

| Stage | Time |
| --- | --- |
| Feature extraction | 918 s |
| Sequential matching | 31 s |
| Mapper | 47 s |
| **Total** | **996 s, about 17 min** |

- Feature extraction is 92% of the cost and is the part a GPU would remove.

## Result: pose recovery succeeded completely

```
Registered images        124 / 124
Mean reprojection error  1.245 px
Points                   4462
Observations             24479
Mean track length        5.49
Solved focal             982.8 px  ->  HFOV 57.6 deg
```

- Every frame registered into one model. No fragmentation, despite thin texture.
- 1.245 px mean reprojection error is a good solve.

## But the structure is nearly empty where the camera walked straight

Points reconstructed per frame, by position along the corridor:

| Segment | Frames | Points | Points per frame |
| --- | --- | --- | --- |
| First third | 73 | 1552 | 21 |
| Middle third | 22 | 203 | 9 |
| Final third | 28 | 14 | **0** |

- 28 frames of walking produced 14 points. That stretch of corridor is geometrically invisible.
- The points cluster where the camera turned. The straight walk contributed almost nothing.
- See `out/colmap_trajectory.png`: the camera path is a long straight run, and the reconstructed points sit in a clump at one end of it.

## Why: tracking and triangulating are different problems

- Matching only needs a point to be recognisable in two frames. 555 of 606 pairs verified, median 87 inliers. Matching was never the bottleneck.
- Triangulating needs the two views to see that point from meaningfully different positions. Walking toward a scene gives rays that are nearly parallel, so where they cross is barely constrained.
- So the solver is confident about where the camera was and nearly blind about what it was looking at.

## Corrections to earlier claims in this repo

- **Field of view was not the problem.** The solved HFOV is 57.6 deg against the 60 deg `reconstruct.py` assumed. Within 2.4 deg. An earlier note treated COLMAP's 2304 px starting prior as evidence the assumed FOV was wrong; that prior is just an initialisation heuristic and means nothing.
- **COLMAP did not fail on low texture.** It was expected to fragment. It did not.
- SIFT counts were thin but sufficient: median 1185 per frame, 10% of frames under 500.

## What this settles

- The bottleneck is the capture, not the software and not the hardware.
- A GPU would make this run in about a minute instead of 17. It would not add a single point to the empty third.
- The fix is a capture that moves the camera **across** the scene rather than into it. See [video-capture-guide.md](video-capture-guide.md).

## Reproduce

```
brew install colmap
colmap feature_extractor --database_path db.db --image_path images \
  --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.use_gpu 0 --FeatureExtraction.max_image_size 1600
colmap sequential_matcher --database_path db.db \
  --FeatureMatching.use_gpu 0 --SequentialMatching.overlap 10
colmap mapper --database_path db.db --image_path images --output_path sparse
colmap model_analyzer --path sparse/0        # writes to stderr
```

- COLMAP 4.x renamed `SiftExtraction`/`SiftMatching` to `FeatureExtraction`/`FeatureMatching`. The old names are rejected outright, the database stays empty, and the mapper then reports "no images with matches" — a failure that looks like a fast success.
- The Homebrew bottle has no GPU support, so `use_gpu` must be set to 0 explicitly. The default is 1.

## Two fixes found in practitioner threads, both free

### COLMAP was discarding the structure, not failing to find it

- Default `Mapper.tri_min_angle` is 1.5 degrees and `tri_ignore_two_view_tracks` is true.
- Forward motion produces triangulation angles near zero for everything ahead of the camera, and surviving tracks are mostly two-view. So COLMAP triangulated the points and then filtered them out as unstable.
- That is the exact signature: excellent poses, vanishing structure.

```
colmap mapper --database_path db.db --image_path images --output_path sparse \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.tri_min_angle 0.1 \
  --Mapper.tri_ignore_two_view_tracks 0 \
  --Mapper.filter_min_tri_angle 0.1
```

| | before | after |
| --- | --- | --- |
| points | 3537 | **11780** |
| observations | — | 46019 |
| mean reprojection | 1.092 px | **0.815 px** |
| registered | 124/124 | 124/124 |

- Points per frame along the walk: 3.4 / 25.5 / 34.8 becomes **16.0 / 73.2 / 97.5**. No segment is empty.
- Reprojection error fell while point count tripled, so these are real observations, not noise.
- Anchors per frame for the depth fit roughly doubled, 189 to 354 median.
- Source: [colmap#3278](https://github.com/colmap/colmap/issues/3278).

### Poisson meshing crashed for a reason that had nothing to do with our data

- `create_from_point_cloud_poisson` aborted with SIGABRT, "Failed to close loop".
- It is a race in PoissonRecon's parallel isosurface extraction, confirmed by its author and reproduced specifically on ARM. Passing **`n_threads=1`** makes it deterministic.
- With that one argument the same cloud meshes in 19 s: 180k vertices, 359k triangles.
- Density trimming must be light. Poisson density is lowest at surface borders, so an aggressive quantile erodes every panel inward until the corridor stops being one object. 12 percent works; 42 percent cut it into disconnected patches.
- Sources: [PoissonRecon#139](https://github.com/mkazhdan/PoissonRecon/issues/139), [#136](https://github.com/mkazhdan/PoissonRecon/issues/136).
