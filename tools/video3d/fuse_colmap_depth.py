"""Place monocular depth maps in COLMAP's solved world frame.

The published cloud stacks depth maps at a baseline the operator GUESSED,
which is why it smears. COLMAP solved the real poses, so each frame can be
placed where the camera actually was.

The remaining problem is scale: Depth Anything returns affine-invariant
inverse depth, so every frame has its own arbitrary scale and offset. But
COLMAP also hands us, for each image, the sparse 3D points it observed and
the pixels they landed on. Those are true depths at known pixels. Fitting
the model's prediction to them, per frame, pins each depth map to the one
shared metric frame -- the sparse points become the anchors.
"""
import json, os, struct, sys, time
import numpy as np, cv2, torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def pick_model(root):
    """COLMAP writes every sub-model it found, numbered by discovery order, not
    by size. On this dataset `sparse/0` holds a 3-image stub and `sparse/1` holds
    the real 70-image reconstruction. Reading `sparse/0` blindly is a silent
    failure: everything downstream runs and returns almost nothing. Pick the
    model with the most registered images."""
    import glob as _g
    cands = _g.glob(f"{root}/sparse/*/images.bin") + _g.glob(f"{root}/sparse/images.bin")
    if not cands:
        raise SystemExit(f"no COLMAP model under {root}/sparse")
    def n_images(f):
        with open(f, "rb") as fh: return struct.unpack("<Q", fh.read(8))[0]
    best = max(cands, key=n_images)
    others = sorted((n_images(c), c) for c in cands)
    if len(cands) > 1:
        print(f"  models: {', '.join(f'{os.path.dirname(c)}={n}' for n, c in others)}")
    return os.path.dirname(best)

SFM = pick_model(sys.argv[1])
IMGS = f"{sys.argv[1]}/images"
OUT = sys.argv[2]
MAX_FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 40
REPORT = sys.argv[4] if len(sys.argv) > 4 else None
STEP = int(os.environ.get("FUSE_STEP", "6"))
# Far field is where an affine-fitted monocular depth map is least trustworthy,
# so both cuts are deliberate. Exposed because the right value depends on how
# deep the scene is relative to the baseline.
ZPCT = float(os.environ.get("FUSE_ZPCT", "92"))     # per-frame far-depth cut
TRIMPCT = float(os.environ.get("FUSE_TRIMPCT", "96"))  # global radial trim
print(f"  using model {SFM}")

def qR(q):
    w,x,y,z = q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])

# --- camera ---
with open(f"{SFM}/cameras.bin","rb") as f:
    struct.unpack('<Q',f.read(8))
    cid,model,W,H = struct.unpack('<iiQQ',f.read(24))
    fpx,cx,cy,k = struct.unpack('<4d',f.read(32))
print(f"  camera {W}x{H} f={fpx:.1f} k={k:+.4f}")

# --- 3D points ---
pts3d = {}
with open(f"{SFM}/points3D.bin","rb") as f:
    n = struct.unpack('<Q',f.read(8))[0]
    for _ in range(n):
        pid = struct.unpack('<Q',f.read(8))[0]
        xyz = np.array(struct.unpack('<3d',f.read(24)))
        f.read(3); struct.unpack('<d',f.read(8))
        tl = struct.unpack('<Q',f.read(8))[0]; f.read(tl*8)
        pts3d[pid] = xyz

# --- images: pose + which 3D points landed on which pixels ---
views = []
with open(f"{SFM}/images.bin","rb") as f:
    n = struct.unpack('<Q',f.read(8))[0]
    for _ in range(n):
        struct.unpack('<i',f.read(4))
        q = np.array(struct.unpack('<4d',f.read(32)))
        t = np.array(struct.unpack('<3d',f.read(24)))
        struct.unpack('<i',f.read(4))
        name=b""
        while True:
            c=f.read(1)
            if c==b'\x00': break
            name+=c
        m = struct.unpack('<Q',f.read(8))[0]
        obs=[]
        for _ in range(m):
            px,py,pid = struct.unpack('<ddq',f.read(24))
            if pid != -1 and pid in pts3d: obs.append((px,py,pid))
        views.append(dict(name=name.decode(), R=qR(q), t=t, obs=obs))
views.sort(key=lambda v: v["name"])
print(f"  {len(views)} posed views, median {int(np.median([len(v['obs']) for v in views]))} anchors each")

# --- depth model ---
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
net = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(dev).eval()

sel = views[::max(1,len(views)//MAX_FRAMES)][:MAX_FRAMES]
clouds=[]; fits=[]; rejected=[]; per_frame=[]; skipped_few=[]
t0=time.time()
for i,v in enumerate(sel):
    img = cv2.imread(f"{IMGS}/{v['name']}")
    if img is None: continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        out = net(**proc(images=rgb, return_tensors="pt").to(dev))
    d = torch.nn.functional.interpolate(out.predicted_depth[None], size=(H,W),
            mode="bicubic", align_corners=False)[0,0].cpu().numpy()   # inverse depth

    # True depth of each anchor, in the camera's own frame.
    P = np.array([pts3d[pid] for _,_,pid in v["obs"]])
    uv = np.array([(px,py) for px,py,_ in v["obs"]])
    zc = (v["R"] @ P.T).T[:,2] + v["t"][2]
    ok = zc > 1e-6
    if ok.sum() < 12:
        skipped_few.append(v["name"]); continue
    u = np.clip(uv[ok,0].astype(int),0,W-1); vv = np.clip(uv[ok,1].astype(int),0,H-1)
    pred = d[vv,u]; true_inv = 1.0/zc[ok]

    # Depth Anything is affine in INVERSE depth, so the fit is linear there.
    A = np.stack([pred, np.ones_like(pred)],1)
    (a,b), *_ = np.linalg.lstsq(A, true_inv, rcond=None)
    resid = np.abs(A@[a,b] - true_inv)
    good = resid < 3*np.median(resid)+1e-9
    (a,b), *_ = np.linalg.lstsq(A[good], true_inv[good], rcond=None)
    rel = float(np.median(np.abs(A[good]@[a,b]-true_inv[good])/np.maximum(true_inv[good],1e-9)))

    # A negative slope means the fit decided the model's near surfaces are
    # far and vice versa. Depth Anything's inverse depth rises toward the
    # camera, so a<=0 is not a scale, it is a flipped frame -- and a flipped
    # frame back-projects into the splayed fans visible in the first render.
    # Refuse it rather than place it.
    if a <= 0 or rel > 0.25:
        rejected.append((v["name"], a, rel))
        per_frame.append(dict(name=v["name"], a=float(a), b=float(b), anchors=int(good.sum()),
                              resid=rel, kept=False,
                              why="flipped_depth_a<=0" if a <= 0 else "resid>25%"))
        continue

    fits.append((a,b,int(good.sum()),rel))
    per_frame.append(dict(name=v["name"], a=float(a), b=float(b), anchors=int(good.sum()), resid=rel, kept=True))

    inv = a*d + b
    Z = 1.0/np.maximum(inv, 1e-6)
    Z[(inv <= 1e-6)] = np.nan

    step=STEP
    ys,xs = np.mgrid[0:H:step, 0:W:step]
    z = Z[ys,xs]
    m = np.isfinite(z) & (z>0.05) & (z<np.nanpercentile(z,ZPCT))
    xs,ys,z = xs[m],ys[m],z[m]
    Xc = np.stack([(xs-cx)*z/fpx, (ys-cy)*z/fpx, z],1)
    Xw = (v["R"].T @ (Xc - v["t"]).T).T
    col = img[ys,xs][:,::-1]
    clouds.append((Xw, col))
    if i%10==0: print(f"    {i+1}/{len(sel)}  scale a={a:.3f} b={b:+.4f}  anchors={good.sum()}")

print(f"  {time.time()-t0:.0f}s, {len(clouds)} frames fused, {len(rejected)} rejected")
for nm,a,r in rejected[:6]:
    why = "flipped depth (a<=0)" if a<=0 else f"poor fit ({100*r:.0f}%)"
    print(f"    rejected {nm}: {why}")
rel = np.array([f[3] for f in fits])
print(f"  anchor fit residual: median {100*np.median(rel):.1f}% of depth")

allX = np.vstack([c[0] for c in clouds]); allC = np.vstack([c[1] for c in clouds])
c = np.median(allX,0); dist = np.linalg.norm(allX-c,axis=1)
keep = dist < np.percentile(dist,TRIMPCT)
allX, allC = allX[keep], allC[keep]
print(f"  {len(allX)} points after trimming")

with open(OUT,'wb') as f:
    f.write(b"ply\nformat binary_little_endian 1.0\n")
    f.write(f"element vertex {len(allX)}\n".encode())
    f.write(b"property float x\nproperty float y\nproperty float z\n")
    f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p,cc in zip(allX.astype('<f4'), allC.astype(np.uint8)):
        f.write(struct.pack('<3f3B',*p,*cc))
print(f"  wrote {OUT}")

if REPORT:
    rq = np.array([f[3] for f in fits])
    json.dump(dict(
        model=SFM, images=IMGS, out=OUT, step=STEP,
        posed_views=len(views), selected=len(sel),
        fused=len(clouds), rejected=len(rejected), skipped_too_few_anchors=len(skipped_few),
        resid_pct_of_depth=dict(median=float(100*np.median(rq)), p90=float(100*np.percentile(rq,90)),
                                min=float(100*rq.min()), max=float(100*rq.max())),
        scale_a=dict(median=float(np.median([f[0] for f in fits])),
                     min=float(min(f[0] for f in fits)), max=float(max(f[0] for f in fits))),
        points_written=int(len(allX)),
        reject_threshold="a<=0 (flipped) or median anchor residual > 25% of depth",
        per_frame=per_frame,
    ), open(REPORT,"w"), indent=1)
    print(f"  wrote {REPORT}")
