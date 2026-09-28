# Gaussian splatting: the plan

> Researched 2026-09-19. The browser half was tested in headless Chrome under our actual CSP, not inferred.

## Why this is now possible

- 3DGS needs solved camera poses. We have them: 124 of 124 frames, 1.092 px, focal pinned at 1641. See [sfm-colmap-result.md](sfm-colmap-result.md).
- That was the blocker. It is gone.
- The remaining limit is the capture, not the tooling. See "Expectation" below.

## Train

- Use **`gsplat`** (Apache-2.0, actively maintained), NOT the INRIA original.
- The INRIA licence is non-commercial, and nearly every "gaussian splatting Colab" wraps it.
- Feed it the COLMAP folder we already have.

```
pip install gsplat
python simple_trainer.py mcmc \
  --data_dir <colmap parent> --data_factor 1 \
  --strategy.cap-max 350000 \
  --save_ply --ply_steps 30000 \
  --disable_viewer --result_dir out
```

- MCMC gives a hard splat budget, which is what makes the file fit the page.
- Expect roughly 15 to 25 minutes on an A100, 2 to 4 GB VRAM.

## Budget

- The page allows about 12 MiB of binary, since base64 inflates by exactly 4/3 into a 16 MiB limit.
- `.splat` is 32 bytes per gaussian, so **about 390,000 fit**.
- Published indoor scenes land at 0.5M to 0.8M gaussians after reduction (room 0.56M, counter 0.50M, bonsai 0.52M). A corridor is simpler than any of those.
- So 350k is a sensible cap and 390k is the ceiling. Tight, but the right order.
- Spherical harmonics are most of the bytes. Truncating SH degree is the first lever if it does not fit.

## View in the browser

- **three.js r186** added native Gaussian splat support. It is the only option that works under our CSP.
- Confirmed by test, both backends: `backend=WebGPUBackend` SUCCESS, `backend=WebGLBackend` SUCCESS, rendering from an inline base64 script with **zero network requests for the data**.
- A grep of every splat file and the whole `three.webgpu.js` build for `new Worker`, `createObjectURL`, `Blob(`, `WebAssembly`, `.wasm`, `eval(` and `new Function` returns **nothing**.

Two rules that are not optional:

- **Use `parse()`, never `.load(url)`.** `.load` routes through `FileLoader` to `fetch()` and dies on our `connect-src`, even for a same-origin relative path.
- The class is **`GaussianSplat`**. `GaussianSplatMesh` was the pre-release name; blog posts using it are stale.
- Add `WebGPURenderer({ forceWebGL: true })` as the fallback where `navigator.gpu` is absent.

### Sorting is not a problem

Measured on Apple Silicon, `CountingSort.computeCPU`:

| gaussians | sort | share of a 60 fps frame |
| --- | --- | --- |
| 100k | 0.6 ms | 4% |
| 250k | 1.8 ms | 11% |
| 500k | 2.9 ms | 17% |
| 1M | 5.7 ms | 34% |

- Linear, about 5.7 ms per million. At 390k that is under 3 ms.
- The widely quoted 150 ms per million from antimatter15's README predates three sort rewrites and is obsolete.
- Re-sort triggers at about 1.8 degrees of view rotation, so a slow camera almost never pays it.

## Every other viewer is blocked

| Library | Blocker |
| --- | --- |
| mkkellogg/GaussianSplats3D | blob worker plus four wasm sorters, no JS fallback |
| antimatter15/splat | blob worker; not on npm |
| @lumaai/luma-web | blob worker plus Emscripten wasm; archived 2024 |
| PlayCanvas engine | near miss. Its sorter builds a worker by stringifying a function, which needs `unsafe-eval` |
| @sparkjsdev/spark | `WebAssembly.compile` at module top level, so importing it alone trips the policy |

- No WebGL2-only GPU splat sorter exists. Every one of them gates the GPU sort behind WebGPU.

## If we mesh instead

- Export the GLB **without** embedded textures and hand three.js the atlas separately.
- Tested: a texture inside a glTF `bufferView` **fails**, because three.js turns it into a blob URL and then fetches it. So does a `.gltf` with `data:` URI buffers.
- What works: `GLTFLoader.parse(glb, '', onLoad)` for geometry, then build the texture from a base64 `<img>` and assign it. That needs only `img-src data:`.
- Budget: about 9 MiB of texture (two 4096 JPEGs at q85) plus about 2 MiB of geometry. 30k to 80k triangles is ample for a corridor.

## Expectation

- This footage walks forward, so the baseline points along the view direction and 3DGS will fit elongated gaussians that look right from the original path and fall apart beside it.
- Compute does not fix that. A capture that moves **across** the scene does. See [video-capture-guide.md](video-capture-guide.md).
- Worth one cheap A100 hour on the current data anyway, to prove the toolchain before demo day rather than during it.

## Unverified

- The `.spz` byte figure is inferred from a vendor claim, not measured.
- The sort benchmark is one Apple Silicon machine. Low-end mobile will be several times slower, though still inside budget at 500k.
- Whether `EXT_texture_webp` shares the blob-URL failure. Assume it does.
