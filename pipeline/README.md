# pipeline — video in, a drivable 2D map out

Five stages. Each writes a file the next one reads. Run them in order, or run `run_all.sh`.

    video  ->  frames  ->  poses + 3D points  ->  metrics  ->  2D map  ->  viewer blob
              stage1        stage2               stage3      stage4     stage5

Everything is in **model units**. Monocular SfM fixes a scene only up to a 7-DoF
similarity, so there is no metre anywhere in this pipeline. `R`, the median camera-path
radius, is the unit every threshold is expressed in, so the numbers carry across rooms.

## The stages

| File | Does | Writes | Cost |
| --- | --- | --- | --- |
| `stage1_frames.py` | Decodes at 12 fps, keeps the sharpest frame per bucket | `images/a_0000.jpg` … | < 1 min |
| `stage2_sfm.py` | COLMAP: features, matching, mapper | `db.db`, `sparse/N/` | 30 min, CPU |
| `stage3_analyse.py` | Did the clips fuse? Was it worth it? | `report.json` | seconds |
| `stage4_map2d.py` | Flattens the cloud to an occupancy grid | `map2d.json`, `.png` | seconds |
| `stage5_pack.py` | Packs the cloud for `viewers/world3.html` | `world3_data.json` | seconds |

`world_io.py` and `colmap_io.py` are libraries, not stages. `world_io.load()` is the one
place that drops diverged cameras and puts the floor on the floor; every stage after 2
goes through it, so they cannot disagree about what the world is.

## Run it

    cd pipeline
    V=../central/.venv/bin/python
    $V stage1_frames.py ../../IMG_7373.MOV ../../IMG_7374.MOV ../../IMG_7375.MOV --out work/images
    $V stage2_sfm.py --images work/images --out work
    $V stage3_analyse.py work --names IMG_7373,IMG_7374,IMG_7375 --out ../results/world3/report.json
    $V stage4_map2d.py work/sparse --out ../results/world3/map2d.json --png ../results/world3/map2d.png
    $V stage5_pack.py work/sparse --out work/world3_data.json --report ../results/world3/report.json

Needs `colmap` (`brew install colmap`), `ffmpeg`, and `central/.venv` for numpy + OpenCV.
No torch, no CUDA, no GPU.

## The three decisions that make this work

- **Sharpest-per-bucket, not fixed stride.** A blurred frame yields no SIFT features and
  costs a full extraction anyway. Measured: 10th-percentile sharpness up 1.4x to 3.0x.
- **Every clip matched against every other clip.** Sequential matching links frame N to
  N+1 inside one clip and never links clip A to clip C. Cross-clip matching is the only
  reason the three clips land in ONE coordinate frame. It cost 11 of the 30 minutes.
- **`tri_min_angle 0.1`, two-view tracks kept.** COLMAP's defaults discard structure
  triangulated at a shallow angle, which is most of it under forward motion. The
  defaults give excellent poses and a nearly empty cloud.

## Traps, all of which have bitten this repo

- **`sparse/0` is not the model.** COLMAP numbers sub-models by discovery order, not size.
  Our run left 2 images in `sparse/0` and the real 179-image model in `sparse/1`. Reading
  index 0 returns almost nothing and looks like a success. `colmap_io.largest_model()`
  picks by size and prints its choice.
- **COLMAP 4.x renamed only some options.** `FeatureExtraction.*` and `SiftExtraction.*`
  both exist and are not interchangeable. A wrong name is a hard parse error, an empty
  database, and then a mapper that says "no images with matches" — which reads like a
  fast success.
- **The Homebrew bottle has no GPU.** `use_gpu` must be set to `0` explicitly; the
  default is 1.
- **A registered camera can still be garbage.** One pose in an earlier run solved to
  1e14 model units. `world_io` drops any camera past 10 units from the camera median and
  lists what it dropped.

## stage4: how a pushable box is told from a wall

Two tests, both needed.

- **Colour.** Above-floor points that are bright and unsaturated. On the joint cloud 34.7%
  of above-floor points pass, so this is a real population, not a sliver.
- **Size.** A white wall is also white. The pushable region is cut into tiles
  `TILE_CELLS` on a side; each tile is a unit the simulator can push.

A tile is **a unit of pushable area, not a detected carton.** The cartons in this room
are stacked shoulder to shoulder, so the pushable region is one connected run 58 x 48
cells across. Connected components cannot split it, and neither can a human eye from a
sparse cloud.

**What would break this:** the colour rule is tuned to this room — white cartons on green
lino under yellow walls. A white wall reads as pushable. So does the stainless fridge
door. A colour threshold cannot tell a carton from an appliance; the fix is a size prior
or a fiducial, not a better threshold.

Also: a sparse cloud cannot support a fine grid. At R/30 most cells hold no point and the
map reads as noise. R/20 is the finest that stays connected.
