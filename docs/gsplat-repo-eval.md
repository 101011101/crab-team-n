# Evaluation: Nannigalaxy/video-3d-reconstruction-gsplat

> Evaluated 2026-09-20. Clone at `out/eval-gsplat` (gitignored). Repo HEAD `9fae019`, 2026-09-05.

## Verdict

- **Do not adopt.** It does not build here and it does not beat our pipeline.
- Training is CUDA-only. Two C++ extensions fail on Apple Silicon.
- Its COLMAP stage also fails here. It uses option names that COLMAP 4.2.0 rejects.
- Its mapper settings are the exact defaults our footage is known to break on.
- It hardcodes `sparse/0`. Our good models land in `sparse/1`. It would train on 3 frames and call it success.
- Keep 3 ideas from it: `image_undistorter`, the `sparse/0` move, the PLY export. All 3 are single COLMAP commands.

## What it actually is

- 3 bash scripts, about 250 lines. **Zero Python.** GitHub reports language `Shell`.
- `git ls-files` returns 9 entries: 3 scripts, LICENSE, README, 2 doc images, `.gitmodules`, 1 submodule pointer.
- The name is a misnomer. **No `gsplat` and no `nerfstudio` anywhere.** It trains `speedy-splat`, a fork of INRIA `graphdeco` 3D-GS.
- It implements nothing. Every step shells out to `ffmpeg`, `colmap`, or `python speedy-splat/train.py`.
- The value it adds is argument defaults and directory plumbing.

Real pipeline:

| Step | Command | Owner |
| --- | --- | --- |
| 1 | `ffmpeg -vf fps=N frame_%04d.png` | ffmpeg |
| 2 | `colmap feature_extractor` | COLMAP |
| 3 | `colmap sequential_matcher` or `exhaustive_matcher` | COLMAP |
| 4 | `colmap mapper` (all defaults) | COLMAP |
| 5 | `colmap image_undistorter` | COLMAP |
| 6 | move `sparse/*.bin` into `sparse/0/` | script |
| 7 | `colmap model_converter` to PLY + TXT (debug only) | COLMAP |
| 8 | `python speedy-splat/train.py` | speedy-splat |

- No viewer. README points at SIBR, which you must build yourself.

## Maintenance and licence

| Fact | Value |
| --- | --- |
| Stars / forks | 52 / 9 |
| Open issues | 0 (1 issue ever, closed) |
| Created | 2025-04-27 |
| Commits | 14 total; 13 in Apr-May 2025, 1 on 2026-09-05 |
| Last push | 2026-09-05 |
| Script licence | MIT |
| Effective licence | **INRIA non-commercial, research only** |

- The submodule drags in the Gaussian-Splatting licence. The MIT header on the scripts does not cover what they run.
- The single 2026 commit added `--disable_gpu` and `--exhaustive`. See below: `--disable_gpu` does not work.

## Dependencies

- speedy-splat `environment.yml`: python 3.7.13, pytorch 1.12.1, torchvision 0.13.1, torchaudio 0.12.1, **cudatoolkit 11.6**, plyfile 0.8.1, tqdm.
- Two pip submodules, both `CUDAExtension` with `.cu` sources:
  - `submodules/simple-knn` (gitlab.inria.fr/bkerbl/simple-knn)
  - `submodules/diff-gaussian-rasterization` (j-alex-hanson/speedy-splat-rasterizer)
- The rasterizer is hand-written CUDA C++. There is no Metal path and no CPU path.

## Build attempt on this machine

Environment: macOS 27, Apple M1, COLMAP 4.2.0 Homebrew, ffmpeg 9.0.2. Fresh venv, Python 3.13.7, torch 2.14.0 `macosx_14_0_arm64`, `torch.cuda.is_available() == False`.

- torch, plyfile, tqdm, pillow, opencv-python: **installed OK.**
- `pip install ./speedy-splat/submodules/simple-knn`, with build isolation:

```
ModuleNotFoundError: No module named 'torch'
```

- Same, `--no-build-isolation`:

```
File ".../torch/utils/cpp_extension.py", line 1573, in CUDAExtension
    library_dirs += library_paths(device_type="cuda")
File ".../torch/utils/cpp_extension.py", line 3315, in _join_cuda_home
    raise OSError('CUDA_HOME environment variable is not set. '
OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
```

- `diff-gaussian-rasterization`: **identical** `OSError`.
- `python -c "import train"`:

```
File ".../gaussian_renderer/__init__.py", line 14, in <module>
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
ModuleNotFoundError: No module named 'diff_gaussian_rasterization'
```

- **Training cannot run here.** This is not a flag or a version pin. The kernels are CUDA.

### Its COLMAP stage fails here too

- The script passes `--SiftExtraction.use_gpu` and `--SiftMatching.use_gpu`. COLMAP 4.2.0 rejects both:

```
E20260920 02:08:29 base_option_manager.cc:265] Failed to parse options -
unrecognised option '--SiftExtraction.use_gpu=0'.
E20260920 02:08:30 base_option_manager.cc:265] Failed to parse options -
unrecognised option '--SiftMatching.use_gpu=0'.
```

- COLMAP 4.x renamed these to `FeatureExtraction` / `FeatureMatching`. Our `--FeatureExtraction.use_gpu 0` runs.
- The script dies at step 1 with or without `--disable_gpu`. The flag sets the value, not the name.
- So **no part of this repo runs on this machine.** Not the training, not the SfM.

## Against our pipeline

Ours is `out/robot1/work/run_colmap.sh`. Theirs is `colmap_undistorted_sfm_export.sh`.

| Setting | Ours | Theirs |
| --- | --- | --- |
| Option names | `FeatureExtraction` (valid) | `SiftExtraction` (**rejected**) |
| `single_camera` | 1 | unset |
| Camera model | SIMPLE_RADIAL | unset |
| `max_image_size` | 3200 | unset |
| SIFT tuning | 16384 features, peak 0.0033, edge 16, first_octave -1 | none |
| Matcher | sequential, overlap 15, quadratic | sequential, defaults (overlap 10) |
| `tri_min_angle` | **0.1** | 1.5 (default) |
| `tri_ignore_two_view_tracks` | **0** | true (default) |
| `filter_min_tri_angle` | 0.1 | default |
| Model chosen | largest | **`sparse/0` always** |
| Undistort + export | not run | run |

Three concrete failures on our data:

- **Mapper defaults destroy structure.** `sfm-colmap-result.md` measured it: defaults give 3537 points at 1.092 px; our overrides give 11780 at 0.815 px. Forward motion makes triangulation angles near zero, so the 1.5 degree default filters real points away.
- **`--exhaustive` is their fix for bad results. It is backwards here.** On robot 1: sequential registered 70/125, exhaustive registered **2/125**.
- **`sparse/0` is the wrong model.** Measured across every workspace we have:

| Workspace | `sparse/0` | `sparse/1` |
| --- | --- | --- |
| `avid2` | 3 images, 78 pts | **276 images, 6073 pts, 0.883 px** |
| `avid` | 2 images, 62 pts | 276 images, 5967 pts |
| `ds` | 3 images, 92 pts | 70 images, 2341 pts |
| `ds_ex` | 2 images, 51 pts | 125 images, 2888 pts |

- Their script prints a warning, then uses `sparse/0` anyway. On our best run that is **3 frames of 276**. Training succeeds and the output is garbage.

Answers:

- **Does it do anything ours does not?** Yes, 3 things: `image_undistorter`, the `sparse/*.bin` to `sparse/0/` move, and PLY/TXT export. All 3 are COLMAP one-liners. It adds no algorithm.
- **Is its COLMAP call better or worse?** Worse. It is invalid on COLMAP 4.x, it uses the defaults we proved destructive, and it picks the wrong model.
- **Net?** **Net loss.** Adopting it replaces tuned, measured settings with defaults that fail on this footage.

## One-line recommendation

- Do not adopt the repo. Copy its 3 COLMAP commands into our own script, and rent a GPU only if we decide a splat is worth it.

## If we rent a GPU

Requirements:

- NVIDIA, compute capability >= 7.0. T4, L4, A10, A100, RTX 4090 all qualify.
- `nvcc` present, CUDA 11.8 recommended over the pinned 11.6. Pair with torch 2.0-2.1 cu118, not torch 2.14.
- Build both extensions on the box. Budget 5-10 minutes.
- Our job is tiny: 276 images at 219x164, 10k iterations. Expect minutes, and a few GB of VRAM.

Approximate cost, **not verified in this session** (web search budget was exhausted; confirm before spending):

| Host | Class | Approx / hr |
| --- | --- | --- |
| Colab | free T4 | 0 |
| Colab Pro | T4 / L4 | ~$10 / month |
| RunPod | RTX 4090 | ~$0.35-0.70 |
| Lambda | A10 / A100 | ~$0.75-1.50 |

- One hour is more than enough. Total cost is under $1 or free.

### Can our workspace feed it directly?

- Not as-is. Verified end to end today, minus the CUDA training step.
- 3D-GS `scene/dataset_readers.py:95` refuses our camera:

```
assert False, "Colmap camera model not handled: only undistorted datasets
(PINHOLE or SIMPLE_PINHOLE cameras) supported!"
```

- Our model is SIMPLE_RADIAL. Two commands fix it, both CPU, both under 1 second:

```
colmap image_undistorter --image_path out/robot1/work/avid2/images \
  --input_path out/robot1/work/avid2/sparse/1 \
  --output_path <dest> --output_type COLMAP
mkdir -p <dest>/sparse/0 && mv <dest>/sparse/*.bin <dest>/sparse/0/
```

- Note `sparse/1`, not `sparse/0`. Their script cannot do this.
- Result parses with the 3D-GS loader: **276 images, 6073 points, PINHOLE 219x164.**
- COLMAP 4.x also writes `rigs.bin` and `frames.bin`. The 2023 loader ignores them. No problem.

Two quality risks, both ours and not the repo's:

- **Resolution.** 320x240 undistorts to **219x164**. The solved distortion `k = 11.76` is extreme, so the undistorter crops hard. 3D-GS at 219x164 will look poor.
- **Initialisation.** 6073 points for 276 views is thin. 3D-GS densifies from the sparse cloud, so a thin start limits the result.
- `robot1-reconstruction.md` already records that focal length is not identifiable on this data. The `k = 11.76` fit is a symptom.

## Reproduce

```
git clone https://github.com/Nannigalaxy/video-3d-reconstruction-gsplat out/eval-gsplat
cd out/eval-gsplat
git submodule update --init speedy-splat
cd speedy-splat && git submodule update --init \
  submodules/simple-knn submodules/diff-gaussian-rasterization
cd .. && python3.13 -m venv .venv && .venv/bin/pip install torch plyfile tqdm
.venv/bin/pip install --no-build-isolation ./speedy-splat/submodules/simple-knn
```

- Init the submodules by name. A plain `--recursive` also pulls `SIBR_viewers`, which is large and not needed.
- Clone plus venv is 962 MB.
