# Viewers

Browser views of the system. Published as Artifacts; the HTML is kept here so it survives the session.

| File | Live | Shows |
| --- | --- | --- |
| `world.html` | [Corridor Point Cloud](https://claude.ai/artifact/JyTg5svgpqYsDdYLnzbabx) | The reconstructed 3D world, three sources compared |
| `mission.html` | [Swarm Mission Replay](https://claude.ai/artifact/5emLuEyby8BTXtS8K7FGnM) | The simulated mission and the waypoint traces |
| `birdseye.html` | not yet published | The same corridor as a 2D occupancy + semantic map |
| `robot1world.html` | not yet published | Robot 1's real session: dense cloud, solved trace against wheel dead reckoning, plan view |

## Data

Each page loads sibling `.js` files that assign one global. They are generated, not committed — regenerate with the exporters noted below.

| File | Global | From |
| --- | --- | --- |
| `cloud_fixed.js` | `CLOUD_FIXED` | `tools/video3d/fuse_colmap_depth.py` output, COLMAP-posed |
| `cloud_qvga.js` | `CLOUD_QVGA` | `reconstruct.py --degrade` |
| `cloud_hires.js` | `CLOUD_HIRES` | `reconstruct.py --single` |
| `mission.js` | `MISSION` | replay recorder over `central/mission_sim.py` |
| `waypoints.js` | `WAYPOINTS` | `central/demo_waypoints.py` routes |
| `birdseye.js` | `BIRDSEYE` | `tools/video3d/birdseye.py` over `out/world_tri.ply` |

Cloud format: `{n, pos, col, path?, npath?}` where `pos` is base64 of a Float32Array of `n*3` metres and `col` base64 of a Uint8Array of `n*3`. Mean-centred.

Bird's-eye format: `{meta, plane, counts, grid, layers, path, walls, openings, regions, widths, metrics}`. `grid` carries `w`, `h`, `cell` metres and the world corner `x0, y0`; `layers` are base64 row-major byte arrays over that grid -- `occ` one byte per cell (0 unknown, 1 free, 2 occupied), `height` one byte scaled by `grid.height_scale` metres, `colour` three bytes. Row 0 is `y0`, so the page flips it to draw +y up. 0.45 MB at 4 cm cells.

## Notes

- Point clouds are loaded as script tags, not fetched. The artifact CSP blocks external fetch, and a same-origin script tag avoids the question entirely.
- Keep the eager payload under about 3 MB. At 4.66 MB the page took over ten seconds to first paint and looked broken.
- `cloud_hires.js` loads lazily on demand for that reason.
- The bird's-eye grid is in its own plane frame: +x along the corridor, +y across it, origin at the fitted floor. It is not the cloud's frame; `plane.normal` and `plane.offset` relate the two.
- The solved cloud is exported already y-up, so it hangs off an unrotated group. The two depth-shell clouds are in camera convention (y down, z forward) and get a pi-about-x flip.

### `robot1world.html` differs from the pages above

- **Self-contained.** No sibling `.js`. All geometry is one embedded JSON blob: positions quantised to 3 x uint16 over a shared box, colours 3 x uint8, base64. 1.95 MB total.
- **three.js r186 is ES-module only.** `cdnjs.../three.js/r186/three.min.js` is a 404; there is no UMD build at that revision. Use `cdnjs.../three.js/0.186.0/three.module.min.js` with `<script type="module">`. `world.html` stays on r128, which still ships UMD.
- Set `THREE.ColorManagement.enabled = false` and `renderer.outputColorSpace = LinearSRGBColorSpace`, or sRGB bytes from the JPEGs are converted twice and wash out.
- Traces and their markers use `depthTest:false` with a high `renderOrder`. Inside a dense cloud an occluded trajectory is an invisible one.
- The page holds **no non-ASCII bytes**, so it does not depend on a charset declaration. Write `\u00b7`, not the character.
- Verified at a real 400 px viewport (`docSW == 400`, no overflow) in both themes, by loading it in a 400 px iframe. Headless Chrome clamps its own window to 500 px, so `--window-size=400` silently tests the wrong width.

## The index is the front door — keep it current

**[index.html](index.html) is published at https://claude.ai/artifact/2epjqDAnMfrgoTDXrS4ppL and pinned.**

Whenever a new page is published, add a row to `index.html` and republish it to that same URL. A link that exists only in a chat transcript is a link the team cannot find.

Each row carries: the name, a one-line description of what it proves, a status chip, and the measured number that makes it worth opening. If a page has no measured number it is probably not finished.

Status chips: `live` published and working, `building` in progress, `webcam`/`sim`/`printer` run it locally.
