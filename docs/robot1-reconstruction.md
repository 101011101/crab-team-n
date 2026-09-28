# Robot 1, first real dataset

> Run 2026-09-20 on `out/robot1/2026-09-19_190600_896778`. COLMAP 4.2.0 Homebrew, CPU only. Apple Silicon, no CUDA.

## Verdict

- SfM **succeeded**. 70 of 125 dataset frames registered at 0.790 px. The 70 are exactly the frames captured while the robot moved.
- The robot drove **straight, forward and back**. It never turned. One 0.1 s steering blip is the only asymmetric command in 49.3 s.
- SfM camera path and wheel dead reckoning agree to **2.1 % along track** (Pearson r = -0.9974).
- Focal length is constrained only to about **160 to 300 px**. Nine runs on one camera returned nine focals. Do not quote a solved focal.
- Two splats of one room **can** be merged, but only with a shared calibration. Measured: **0.5 %** scale error with one, **65 %** without.
- Densification: COLMAP dense stereo is **CUDA-blocked**. Monocular depth anchored to the sparse points gives **307k colour points**, accurate to **10-19 % in depth**, reaching only **1.5 m**. TSDF meshing fails. See [section 4](#4-densification).

## No Gaussian splat was trained

- `gsplat`, `nerfstudio`, `diff_gaussian_rasterization`, `simple_knn`: all absent. No trainer of any kind is installed.
- `torch.cuda.is_available()` is False. MPS only. `gsplat` needs CUDA for its rasteriser.
- 3DGS training is **not possible on this machine**. This is a hard blocker, not a tuning problem.
- Delivered instead: COLMAP **sparse point clouds**, 2225 points from the dataset frames and 5714 from the AVI. These are point clouds, not splats.
- The poses are the expensive part and they exist. If a CUDA machine appears, feed it the COLMAP folder and `gsplat` per [splat-plan.md](splat-plan.md). Fix the focal first.
- **Now ready to run**: [`tools/video3d/train_splat_colab.ipynb`](../tools/video3d/train_splat_colab.ipynb) plus `out/robot1/robot1_splat_input.zip`. It pins gsplat 1.5.3, rewrites `cameras.bin` to one shared fixed focal before training, and exports a `.ply` splat.

## Environment corrections

- COLMAP 4.2 splits the old names. **Both prefixes exist**: `FeatureExtraction.*` holds type/threads/gpu/max_image_size; `SiftExtraction.*` holds max_num_features/peak_threshold/edge_threshold/first_octave. Matching splits the same way into `FeatureMatching.*` and `SiftMatching.*`.
- [sfm-colmap-result.md](sfm-colmap-result.md) says `SiftExtraction` was renamed to `FeatureExtraction`. That is now wrong and cost one silent failure: a bad option name aborts the extractor, the database stays empty, and the mapper reports "No images with matches".
- Always check the exit code of each COLMAP stage. The failure looks like a fast success.

## Data as measured, not as described

| Item | Value |
| --- | --- |
| Dataset frames | 125, 320x240, 3-6 KB JPEG |
| Mean frame interval | 0.398 s (2.51 fps) |
| Session length | 49.3 s |
| Telemetry rows | 492 at 10 Hz |
| AVI | 360 frames, 12 fps, 30 s |
| AVI duplicate frames | **84 of 359** consecutive pairs are byte-identical; 276 unique |
| AVI time span | session t = 4.87 to 34.87 s |
| AVI frame rate, fitted | 11.991 fps, `capture_ms = 83.394 x frame + 11656624` |

- `dataset.json`: `calibrated: false`, `intrinsics: null`, `odometry_source: motor_targets_not_encoders`.
- Raw JPEGs are `raw_sensor_stream`. Apply a **vertical flip** (rotate 180 then flip horizontal reduces to flip vertical).
- The **AVI is already in display orientation**. Applying the flip again mirrors it. Found by cross-matching: 110 of 125 dataset frames match an AVI frame at correlation 1.000 only when the extra flip is removed.
- A mirrored image set still reconstructs, so this error is silent. It breaks any later comparison to odometry, because a proper rotation cannot align a mirrored model.

## 1. Reconstruction

### Dataset frames, 2.5 fps

```
colmap feature_extractor --database_path db.db --image_path images \
  --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.use_gpu 0 --FeatureExtraction.max_image_size 3200 \
  --SiftExtraction.max_num_features 16384 --SiftExtraction.peak_threshold 0.0033 \
  --SiftExtraction.edge_threshold 16 --SiftExtraction.first_octave -1
colmap sequential_matcher --database_path db.db --FeatureMatching.use_gpu 0 \
  --SequentialMatching.overlap 15 --SequentialMatching.quadratic_overlap 1 \
  --SiftMatching.max_ratio 0.9
colmap mapper --database_path db.db --image_path images --output_path sparse \
  --Mapper.tri_min_angle 0.1 --Mapper.tri_ignore_two_view_tracks 0 \
  --Mapper.filter_min_tri_angle 0.1 --Mapper.abs_pose_min_num_inliers 12 \
  --Mapper.min_num_matches 12
```

| Metric | Value |
| --- | --- |
| Registered | **70 / 125** |
| Mean reprojection error | **0.790 px** |
| Points | 2341 |
| Observations | 15709 |
| Mean track length | 6.71 |
| Solved focal | 142.4 px — **do not trust, see below** |

- The 70 registered frames are indices **13 to 82, contiguous, with no gaps**.
- Commanded motion ran t = 2.20 to 33.20 s. Frame 82 is t = 33.54 s. The registered block **ends at the last motion command**.
- So the correct reading is not "70 of 125 failed partly". It is "**every frame taken while the robot moved registered; the 55-frame idle tail did not**".
- Idle-tail frames show a blank wall. Median 99 SIFT keypoints against 382 for registered frames.

### The tri_min_angle fix still applies

- `tri_min_angle 0.1` and `tri_ignore_two_view_tracks 0` were used throughout, per [sfm-colmap-result.md](sfm-colmap-result.md).

### Exhaustive matching is much worse

- Exhaustive matcher on the same 125 frames: **2 of 125 registered**.
- Repetitive blank walls produce false long-range matches. Sequential matching is required here.

### Two failure modes to filter

- 1 of 70 cameras diverged to `|C|` about 1e14. 11 of 2341 points likewise. Drop them before any use.
- Median consecutive baseline / median scene depth = **0.063**. That is just above the level where triangulation is stable.

### AVI frames, 12 fps

| Metric | Value |
| --- | --- |
| Registered | **276 / 276** unique frames |
| Mean reprojection error | 0.883 px |
| Points | 6073 |
| Mean track length | 12.04 |
| Solved focal | 62.6 px — again not to be trusted |

- Use the **unflipped** extraction. 84 of 359 consecutive AVI frames are byte-identical and must be dropped first; zero-baseline pairs are degenerate.
- 276 of 276 is the best registration rate of any run here, and it still produced a worse trajectory than 70 of 125. **Registration rate is not a quality metric.**

## Focal length is not identifiable

- This is the second time this project has been bitten here. The answer is that the data does not contain the answer.

Every independent run solved a different focal on the same physical camera:

| Run | Images | Solved f (px) |
| --- | --- | --- |
| halfB fragment | 2 | 25.0 |
| ds joint | 70 | 142.4 |
| halfA | 37 | 157.0 |
| halfA_nolap | 34 | 195.8 |
| halfB_nolap | 34 | 508.7 |
| avid fragment | 2 | 1146.9 |
| halfB degenerate | 19 | 302626.2 |

- Among models with 34 or more images: **142.4, 157.0, 195.8, 508.7 px**. A 3.6x spread.

Focal held **fixed**, 125 dataset frames, focal refinement off:

| f (px) | Registered | Points | Mean reproj (px) |
| --- | --- | --- | --- |
| 120 | 70 | 2552 | 0.788 |
| 160 | 70 | 2465 | 0.755 |
| 200 | 70 | 2456 | 0.749 |
| 236 | 70 | 2424 | 0.744 |
| 280 | 70 | 2465 | **0.742** |
| 340 | 70 | 2334 | 0.776 |
| 420 | 70 | 2313 | 0.834 |
| 550 | 70 | 2331 | 0.830 |

- Every focal from 120 to 550 px registers **70 of 70**. Registration count carries no information about focal.
- Reprojection error moves only **0.092 px across a 4.6x focal range**.
- The sweep does give a **weak bracket**. Taking minimum + 0.02 px, the admissible band is about **160 to 300 px**, that is HFOV 56 to 90 deg. It rejects 120 px and everything at or above 340 px.
- OV2640 with the common 2.65 mm lens predicts f about 236 px, HFOV about 68 deg. That is a **datasheet expectation, not a measurement**. It sits inside the bracket, which is mild corroboration and nothing more.
- Free bundle adjustment returned 142.4 px, HFOV 96.6 deg, with only mild distortion (k1 = -0.072). A 97 deg lens needs strong distortion. That value falls **outside** the sweep bracket and is an artefact.
- Honest summary: the data constrains the focal to roughly a factor of two and no better. **Any single solved focal quoted from this dataset is noise.**
- **Action: calibrate the camera.** A checkerboard or a known-size AprilTag. Nothing in this dataset can substitute.

## Why the geometry is hard here

- SIFT keypoints per image: median **116** over all 125 frames. The earlier phone clip had median 1185.
- Motion is along the optical axis. The **epipole of F falls inside the 320x240 frame** for about half of the moving pairs, median 115 to 175 px from the image centre. That is forward or backward driving, not lateral.
- F is near-degenerate: second-to-first singular value ratio **0.046 to 0.055**. Epipole scatter is correspondingly large.
- Fundamental-matrix inlier ratio median **0.47**. Homography inlier ratio median **0.18**. Ratio `H_inliers / F_inliers` median **0.40**, and **0 of 44** moving pairs exceed 0.9.
- So the motion is **not pure rotation and the scene is not a plane**. This corrects the earlier "rotation only" verdict recorded in this repo.
- The real limit is parallax: forward motion, 0.063 baseline-to-depth, and 116 features per frame.

## 2. Two traces of the same drive

### What the robot actually did

- Distinct motor commands in the whole session: `(0.25, 0.25)`, `(-0.25, -0.25)`, and `(0.45, 0.05)` for **one 0.1 s sample**.
- **No turn command was ever issued.** `turn_speed` 0.2 never appears.

| Quantity | Telemetry echo | Laptop CSV (independent) |
| --- | --- | --- |
| Forward time | 9.00 s | 9.02 s |
| Reverse time | 10.50 s | 10.45 s |
| Stationary | 29.80 s of 49.30 s | — |

- Two independent command records agree to **0.05 s**.
- The robot was stationary for **60 %** of the session.

### Dead reckoning

- `v = k (l + r) / 2`, `omega = k (r - l) / track`.
- **Assumption**: `k = 0.50 m/s` at motor target 1.0. **Assumed, not measured.**
- **Assumption**: `track = 0.12 m`. **Assumed, not measured.**
- **Scale is unidentifiable from this data.** Motor targets are not encoder counts. No wheel ever reported its true speed. Every metre below is proportional to `k`.
- Track width is almost irrelevant here, because no differential command was issued except the single blip.

| Quantity | Telemetry echo | Laptop CSV |
| --- | --- | --- |
| Path length | 2.438 m | 2.443 m |
| Net displacement | 0.238 m | 0.245 m |
| Final heading | -9.55 deg | -10.41 deg |

- The two dead-reckoning routes disagree by **1.2 cm, 0.5 % of path length**.
- All 9.55 deg of heading change comes from the single 0.1 s blip. Under the assumed `k` and `track` that blip is a real 9.5 deg turn, but its size scales with both assumptions.

### Telemetry validated against the images

- Image flow, frames where the robot was commanded to stop: median **0.01 px/s**.
- Image flow, frames where the robot was commanded to move: median **1.85 px/s**.
- Mann-Whitney p = 2e-13.
- So the motors really did follow the targets, and the stopped periods were genuinely stopped. The 0.25 target is above the deadband.

### SfM path against dead reckoning

- SfM agrees independently that the motion was straight:
  - **99.1 %** of camera-centre variance lies on one axis.
  - Angle between the motion axis and the mean optical axis: **7.5 deg**.
  - Optical-axis deviation: median 2.8 deg, max 8.0 deg.

| Comparison | Value |
| --- | --- |
| Along-track Pearson r | **-0.9974** |
| Along-track residual RMS | **2.29 cm**, 2.1 % of 1.114 m travelled |
| 2D similarity residual RMS | 5.19 cm, **2.4 %** of the 2.17 m path |
| Max residual | 15.2 cm |
| Fitted scale | 0.0900 m per model unit |
| SfM path length / DR path length | 2.121 / 2.170 m = **0.978** |

- Where they disagree, and why:
  - **Lateral wander, about 3 cm.** Dead reckoning uses motor *targets*, so it draws a perfectly straight line. The real robot veers. SfM sees the veer. This disagreement is dead reckoning being wrong, not SfM.
  - **Start of the run**, 15 cm. The first registered frames have the weakest structure.
  - **Scale** is shared, not independent. The 0.0900 m per unit comes from fitting SfM to dead reckoning, so it inherits the assumed `k`.

### More frames bought nothing

| Source | Registered | RMS vs DR | r | SfM path / DR path |
| --- | --- | --- | --- | --- |
| 125 dataset frames, 2.5 fps | 70 / 125 | **5.2 cm**, 2.4 % | -0.9974 | **0.978** |
| 276 AVI frames, 12 fps | **276 / 276** | 8.4 cm, 3.9 % | -0.9954 | **1.384** |

- Both recover the shape of the drive. The 12 fps run registers 4x as many frames and is still **1.6x less accurate** in position.
- The clear loss is path length. The 12 fps run integrates a path **38 % too long**; the 2.5 fps run is within 2 %.
- At 12 fps each inter-frame step is near the pose noise floor, so noise adds to path length instead of cancelling. At 2.5 fps the step is about 5x larger against the same noise.
- **A correction to an earlier draft of this note.** The first 12 fps figures were RMS 26.7 cm and r = -0.64. Both were wrong. Two bugs, now fixed:
  - the AVI had been flipped a second time, so the model was **mirrored** and no proper rotation could align it;
  - the AVI-to-telemetry time map was built by holding duplicate frames, giving errors up to **1249 ms**.
  - Fixing both moved r from -0.64 to -0.9954. Only the path-length finding survived unchanged.
- Lesson worth keeping: a low correlation against odometry is evidence of a **bookkeeping** bug at least as often as a reconstruction failure. Check orientation and time alignment first.

### Scene extent

- 99 % of points fit a **5.05 x 2.48 x 2.96 m** box, under the assumed scale.
- Camera path extent: **1.05 m**, against 1.114 m of dead-reckoned travel. Consistent.

## 3. Can two splats of one room be merged?

**Yes, but only if the two runs share one calibration. Measured below.**

### Why they start in different frames

- Monocular SfM fixes the scene only up to a **7-DoF similarity**: 3 rotation, 3 translation, 1 scale.
- Nothing in a set of images defines a world origin, a world orientation, or a metre. The solver picks them arbitrarily.
- So two independent runs of the same room produce two clouds that are correct and incompatible at the same time.
- **Worse in practice**: if each run also solves its own focal, the two reconstructions are related by a **projective**, not a similarity, transform. No 7-DoF fit can then align them. This dataset showed exactly that, twice: 195.8 px against 508.7 px for the same camera.

### The demonstration

- Split the registered frames into two "robots" with **no shared images**: A = frames 13-47, B = frames 48-82.
- The robot reversed then drove forward over the same ground, so the two halves see overlapping scene.
- Reconstruct each independently. Both give 34 of 35 registered.
- Link them only through **cross-half feature matches**, then fit a 7-DoF similarity with RANSAC.
- Ground truth: a **joint** COLMAP run over all 70 frames at once.

| | Independent intrinsics | Shared fixed focal (160 px) |
| --- | --- | --- |
| Solved focals | 195.8 / 508.7 px | 160 / 160 px |
| 3D-3D correspondences | 599 | 563 |
| RANSAC inliers | 35 (**6 %**) | 66 (**12 %**) |
| Scale error vs truth | **64.8 %** | **0.5 %** |
| Rotation error | 24.6 deg | 15.8 deg |
| Camera displacement | **106 %** of path radius | **5.8 %** of path radius |

- Camera displacement is RMS distance between the post-hoc placement and the true placement, normalised by the half-B camera-path radius.

- Over 6 RANSAC seeds: scale error 64.8-72.1 % independent, **0.1-3.0 %** shared. The result is stable, not a lucky draw.
- Reading: with independent intrinsics the merge is **worse than not merging**; the cloud lands a full scene-radius away. With one shared calibration it lands within 6 % and is usable.

### The margin is thin

- Held out half the correspondences, fitted on the rest:

| | Independent | Shared f=160 |
| --- | --- | --- |
| Held-out residual, median | 106 % of scene radius | 27 % of scene radius |
| Held-out pairs within 5 % radius | 2 % | **9 %** |
| Cloud-to-cloud nearest-neighbour, median | 11.1 % of radius | **7.6 %** of radius |

- So **about 88 % of the cross-robot "correspondences" are false matches**, made on repetitive wall texture.
- The merge works only because RANSAC isolates the 12 % that are real. That margin is thin and it will not survive a worse scene or fewer frames.
- The transform itself is right: checked against the joint-run ground truth, not against this outlier-heavy set.
- This is the strongest argument for the tag. A fiducial gives **4 exact correspondences of known metric size** instead of 563 correspondences that are 88 % wrong.

### The rotation error that survives

- 15.8 deg of rotation error remains even with a shared focal.
- Its axis is **0.91 to 0.95 aligned with the drive axis**. It is almost pure **roll about the direction of travel**.
- A straight-line camera path with a thin point cloud cannot observe roll. No amount of feature matching fixes it.
- This is the precise gap a fiducial closes.

### Why an AprilTag of known printed size fixes all 7 DoF at once

- **Scale**: 4 corners at a known metre spacing. Scale stops being free.
- **Rotation, all 3**: the tag plane gives 2 axes; the tag's own in-plane orientation gives the third. That third one is the roll a straight drive cannot see.
- **Translation, all 3**: the tag centre is a shared world origin.
- One tag seen by both robots collapses the whole 7-DoF search to a lookup. That is why this project uses tags.
- Tag size must be known and printed accurately. See [tag-size-rule.md](tag-size-rule.md).

### The order to prefer

1. **One joint COLMAP run over all images from all robots.** Usually correct, and it was correct here: each half agreed with the joint run to **5.5 to 10.3 %** of the joint camera-path radius, against 106 % for the failed post-hoc merge. Costs nothing extra at this scale.
2. **Shared fiducial.** Needed when robots cannot pool images, or when a metric scale is required.
3. **Shared fixed calibration plus post-hoc similarity.** Works, 0.5 % scale error here, but leaves unobservable rotations.
4. **Post-hoc registration with independent intrinsics.** Failed here. Do not use.
- ICP was not tried and is not recommended for this data: 2341 points over a 5 m room is too sparse, and ICP cannot recover scale.

### What specifically breaks at 320x240

- **116 median SIFT features per frame.** Cross-robot correspondences fall to 6-12 % RANSAC inlier rate.
- **Focal unidentifiable**, so independent runs are not even similarity-related.
- **F near-degenerate**, ratio 0.046, so relative pose between two robots is poorly conditioned.
- **Blank walls**: 55 of 125 frames here carried almost no features at all.
- Mitigations, in order of effect: bolt down one shared calibration; put a tag in view; drive **across** the scene, not into it; sample near 2.5 fps, not 12.

## 4. Densification

> Run 2026-09-20, same machine. Inputs: `out/robot1/work/ds`, model `sparse/1`, 70 registered frames.

### Verdict first

- COLMAP dense stereo is **blocked, not slow**. It needs CUDA and refuses in under a second.
- The monocular-depth route runs and gives **307,364 colour points** against 2,225 sparse ones.
- Those points are **predicted, not triangulated**, and accurate to only **10 to 19 % in depth**.
- The dense cloud reaches **1.5 m**. The sparse cloud reaches **5 m**. The far half of the room is absent.
- TSDF meshing **fails** at every voxel size tried. Best result 2,131 vertices over 0.16 m2.
- Read this as: more points, not more information. See "What the extra points are worth".

### COLMAP dense stereo will not run

```
colmap image_undistorter --image_path .../images --input_path .../sparse/1 \
  --output_path .../dense_ds --output_type COLMAP --max_image_size 640     -> exit 0, 70 images
colmap patch_match_stereo --workspace_path .../dense_ds                    -> SIGABRT
```

- Exact error: `Dense stereo reconstruction requires CUDA, which is not available on your system.`
  Source `mvs.cc:261`. The binary reports `COLMAP 4.2.0 ... without GPU support`.
- Undistortion succeeds. Only PatchMatch fails. There is no CPU fallback in COLMAP.
- No other MVS engine is installed: OpenMVS, PMVS, CMVS all absent. `brew list` shows `colmap` alone.
- Rectified two-view stereo is **not** a substitute for this footage. The epipole falls inside the
  320x240 frame for about half the moving pairs, and stereo rectification is undefined when the
  epipole lies inside the image. Plane-sweep or PatchMatch is required, and both need the GPU.
- Time-boxed to 5 minutes. It aborted in under 1 second. No tuning changes this.

### One silent COLMAP trap, found and fixed

| Model | Registered images | Points |
| --- | --- | --- |
| `ds/sparse/0` | 3 | 92 |
| `ds/sparse/1` | **70** | **2341** |

- COLMAP numbers sub-models by **discovery order, not size**. On this dataset `sparse/0` is a stub.
- `fuse_colmap_depth.py` and `tsdf_mesh.py` both read `sparse/0` unconditionally. The first dense run
  therefore used 3 images and nobody would have seen it: the pipeline succeeds and returns almost nothing.
- Both tools now pick the model with the most registered images and print the choice. Check any COLMAP
  workspace with more than one `sparse/N` before use.

### Monocular depth, anchored to the sparse points

- Method unchanged from the corridor work: Depth Anything V2 Small predicts affine-invariant **inverse**
  depth; a 2-parameter fit `inv = a*d + b` per frame ties it to the true depths of the sparse points that
  frame observed. Reject `a <= 0` (flipped) or median residual above 25 %.

| Metric | Value |
| --- | --- |
| Posed views | 70 |
| Median anchors per view | 218 |
| Fused | **57** |
| Rejected, residual > 25 % | 11 |
| Rejected, flipped depth `a <= 0` | 1 (`000078.png`) |
| Skipped, fewer than 12 anchors | 1 |
| Anchor fit residual, kept frames | median **14.6 %** of depth, p90 23.3 % |
| Anchor fit residual, all 69 evaluated | median 16.0 %, worst 110.7 % |
| Raw points, 2 px sample step | 966,574 |
| Deliverable, 1 cm voxel + outlier removal | **307,364** |

- Rejected frames cluster: `000044`-`000055`, `000067`-`000069`, `000075`, `000078`. All near the reversal.
- Colour is sampled from the source frame at each kept pixel.

### The fit residual is a training number

- 14.6 % is measured at the very pixels the 2-parameter fit was optimised on. It is not an error bar.
  Three independent checks follow. Only the third one is load-bearing.

**1. Hold out half the anchors.** Fit on half, test on the other half, compare depths in metres.

| | Median | p90 |
| --- | --- | --- |
| Train residual | 16.5 % | 26.3 % |
| **Held-out depth error** | **18.9 %** | 31.9 % |

- So the fit does not overfit. It is simply only this good. Depth is right to about a fifth.

**2. Nearest neighbour between two disjoint halves.** Fuse even and odd frames separately, measure the
distance from each point to the nearest point of the other half.

| | Value |
| --- | --- |
| Cross-group nearest neighbour, median | 8.7 mm |
| **Within-group nearest neighbour, same measure** | **7.9 mm** |
| Ratio | 1.10 |

- **This test proves nothing.** 240,000 points inside a 0.51 m radius means nearest-neighbour distance
  measures sampling density, not agreement. The within-group control is the density floor and the
  cross-group number sits 10 % above it. Quoting the 8.7 mm alone would have been a false precision claim.

**3. View-to-view depth consistency.** Project frame i's depth cloud into frame j and compare against
frame j's own depth at the same pixel. Scale-free and density-free.

| Frame gap | Pairs | Median disagreement |
| --- | --- | --- |
| 3 | 52 | **10.7 %** |
| 6 | 49 | 13.5 % |
| 10 | 45 | **16.0 %** |

- At the 0.66 m median depth this is **7 to 11 cm**. That is the honest error bar on the dense cloud.
- It agrees with the held-out anchor number, 18.9 %, by a second and independent route.

**4. Sparse points to nearest dense point.** Median 16 mm, p90 133 mm. Partly circular, because the
sparse points are the anchors, so read it only as "the dense cloud did not drift off its anchors".

### Why the dense cloud stops at 1.5 m

- `Z = 1 / (a*d + b)`, so the deepest point any frame can produce is **`1/b`**.
- Measured over the 57 kept frames: `b` median +0.068, so the ceiling is **1.32 m** (p10 0.47, p90 2.09).
- The anchors are mostly near surfaces, so `b` is fitted from near points and caps the reachable depth.

| Cloud | 99 % extent |
| --- | --- |
| Sparse, triangulated | 2.48 x 1.28 x 1.79 m |
| Dense, predicted | **1.52 x 1.06 x 1.22 m** |

- This is **not** caused by the trimming. Loosening the per-frame far cut from the 92nd to the 99th
  percentile and the radial trim from 96 % to 99 % moved the extent only 1.42 m to 1.59 m.
- The ceiling is the mechanism. Nothing beyond `1/b` exists to keep.

### What the extra points are worth

| Voxel size | Occupied voxels |
| --- | --- |
| 5 mm | 659,509 |
| 10 mm | 321,483 |
| 20 mm | 76,711 |
| 50 mm | 7,591 |

- 966,574 raw points are not 966,574 pieces of geometry: 57 depth maps of one wall stack 57 near-copies.
- The measured accuracy is 7 to 11 cm. At **50 mm**, still finer than that accuracy, only **7,591** voxels
  are occupied.
- So the honest gain over the 2,225 sparse points is roughly **3x in independent surface samples**, not 138x.
- What the dense cloud does add that the sparse cloud cannot: **continuous surfaces with colour**. It
  reads as walls rather than dots, and it is enough to render. It is not enough to measure.

### TSDF meshing fails

Voxel sweep, 57 frames integrated each time, largest-component filter applied:

| Voxel (model units) | Voxel (m) | `sdf_trunc` (m) | Vertices kept |
| --- | --- | --- | --- |
| 0.06 | 0.005 | 0.022 | 248 |
| **0.12** | **0.011** | **0.043** | **2,131** |
| 0.30 | 0.027 | 0.108 | 1,491 |
| 0.60 | 0.054 | 0.216 | 1,067 |
| 1.00 | 0.090 | 0.360 | 569 |

- Best result: 2,131 vertices, 3,502 triangles, extent 0.51 x 0.10 x 1.11 m, surface area **0.16 m2**.
- A 0.10 m Y extent over a 1.5 m scene is a **sliver, not a room**. Rendered top-down it is a set of
  disconnected horizontal slabs.
- Cause, and it is measured: views must agree within `sdf_trunc` to reinforce. View-to-view disagreement is
  7 to 11 cm. Any `sdf_trunc` large enough to absorb that is also large enough to erase room structure.
  The sweep shows exactly that shape, a single weak peak with falloff on both sides.
- **The mesh is not usable.** It is kept as evidence, not as a deliverable.
- Poisson was not attempted. It is the wrong tool for stacked depth shells, and the ARM threading race
  makes it unreliable at `n_threads > 1`.

### The 12 fps AVI densifies worse, again

| | 70 dataset frames, 2.5 fps | 276 AVI frames, 12 fps |
| --- | --- | --- |
| Fused | 57 of 70 (81 %) | 121 of 276 (**44 %**) |
| Rejected | 12 | **155** |
| Anchor residual, median | 14.6 % | **18.7 %** |
| Degenerate fits (`a` below 1e-30) | 0 | present, `a` down to 1.5e-37 |

- The AVI cloud is wider, 3.9 x 2.9 m, but that is rejected-frame noise spreading, not recovered room.
- **Not shipped.** Evidence kept in `dense_fit_avi.json` and `dense_density_avi.json`.
- Third independent confirmation that registration rate is not a quality metric on this dataset.

### Honest summary

- The densification **worked as designed and the design is limited by the capture**.
- Nothing here is a measurement of the room. It is a prediction, anchored to measurements, with a 7 to 11 cm
  error bar and a 1.3 m horizon.
- It is good enough to look at and to publish. It is not good enough to plan on, and the plan view in the
  viewer therefore uses the traces, not the cloud.
- The fix is capture, not compute. See "Next capture".

## 5. Deliverables added

| File | Contents |
| --- | --- |
| `crab-team-n/viewers/robot1world.html` | Self-contained viewer. 155k dense points, both traces, plan view, gap-vs-time, honest annotations. 1.94 MB, no external data fetch. three.js r186 ES module from cdnjs. |
| `crab-team-n/tools/video3d/train_splat_colab.ipynb` | Ready-to-run 3DGS training notebook for a rented CUDA GPU. Forces one shared fixed focal. Verified valid; every pinned version checked live on PyPI. |
| `out/robot1/dense_ds.ply` | 307,364-point colour cloud, metres, 1 cm voxel, outliers removed |
| `out/robot1/dense_ds_raw.ply` | 966,574-point cloud before voxel reduction, model units |
| `out/robot1/dense_mesh_ds.ply` | TSDF mesh, 2,131 vertices. **Evidence of failure, not a deliverable.** |
| `out/robot1/dense_fit_ds.json` | Per-frame anchor fit: `a`, `b`, anchor count, residual, keep or reject and why |
| `out/robot1/dense_consistency_ds.json` | Density control and view-to-view consistency |
| `out/robot1/dense_validation_ds.json` | Held-out anchor test, nearest-neighbour tests |
| `out/robot1/dense_density.json` | Voxel occupancy against voxel size |
| `out/robot1/tsdf_sweep.json` | The TSDF voxel sweep and the best mesh's extent |
| `out/robot1/dense_fit_avi.json`, `dense_density_avi.json` | The 12 fps comparison |
| `out/robot1/splat_input/`, `robot1_splat_input.zip` | Exactly what the notebook expects: 70 registered images plus `sparse/0` copied from `sparse/1` |

### Viewer notes

- **three.js r186 has no UMD build.** `cdnjs.../three.js/r186/three.min.js` returns 404. The real path is
  `cdnjs.../three.js/0.186.0/three.module.min.js`, an ES module, loaded with `<script type="module">`.
  Verified: HTTP 200, `REVISION="186"`. The older `world.html` uses r128, which still has a UMD build.
- Point data is quantised to 3 x uint16 plus 3 x uint8 and embedded as base64. No external fetch.
- The page contains **no non-ASCII bytes**, so it cannot depend on a charset declaration.
- The SfM-to-dead-reckoning similarity was **re-derived independently** for the viewer, by Umeyama fit.
  It reproduces the stored `sfmfit` to 2.2e-16 m and returns scale 0.090050 m per unit, rotation 175.382
  deg, RMS 5.19 cm, max 15.20 cm. All four match section 2. The traces in the viewer are therefore the
  same fit, not a second one.

## Outputs

| File | Contents |
| --- | --- |
| `out/robot1/sfm_points_ds.ply` | 2225-point sparse cloud, outliers trimmed, assumed metric scale |
| `out/robot1/sfm_cameras_ds.ply` | 69 camera centres |
| `out/robot1/sfm_camera_trace.json` | camera centres with frame names |
| `out/robot1/deadreckon_trace.json` | both dead-reckoning routes, assumptions recorded inline |
| `out/robot1/trace_comparison.json` | SfM against dead reckoning |
| `out/robot1/merge_experiment.json` | both merge conditions |
| `out/robot1/focal_sweep.json` | the fixed-focal sweep |
| `out/robot1/sfm_points_avi.ply` | 5714-point cloud from the 12 fps AVI, correct orientation |
| `out/robot1/trace_comparison_avi.json` | AVI trace against dead reckoning |
| `out/robot1/trace_overlay.png` | three traces, overlay, residual |
| `out/robot1/deadreckon_trace.png` | trace, motor targets, flow validation |
| `out/robot1/merge_demo.png` | merge demonstration |
| `out/robot1/focal_identifiability.png` | sweep and solved-focal spread |

## Next capture

- **Calibrate the camera once.** Checkerboard, 30 frames, or a known-size tag. Everything downstream is blocked on this.
- **Put an AprilTag in the scene** and keep it in view at the start and end of every run.
- **Drive across the room**, not down its axis. Forward driving is the worst case for triangulation and this run was almost pure forward driving.
- **Turn.** This run contains 0.1 s of steering in 49.3 s. Rotation is what makes structure observable.
- **Do not idle with the camera running.** 60 % of this session, and 55 of 125 frames, were a stationary view of a blank wall.
- Sample near **2.5 fps**. 12 fps added frames and lost accuracy.
