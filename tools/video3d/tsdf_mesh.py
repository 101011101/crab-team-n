"""Fuse RGB + depth + solved poses into a watertight surface (TSDF).

Poisson reconstruction on the merged point cloud kept aborting, and it was
the wrong tool anyway: it takes an unstructured cloud and guesses a surface,
throwing away the fact that we know which camera saw which pixel.

We have exactly what volumetric integration wants -- colour images, depth
maps, and a pose per frame. Each frame carves a truncated signed-distance
field, so surfaces seen from several views reinforce and space nobody saw
stays empty. That is why the result reads as walls rather than dots.

    ./.venv/bin/python tsdf_mesh.py <sfm_dir> <out.ply> [n_frames]
"""
import glob, os, struct, sys, time
import numpy as np, cv2, torch, open3d as o3d
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT, OUT = sys.argv[1], sys.argv[2]
# COLMAP numbers sub-models by discovery order, not size: on robot 1 `sparse/0`
# is a 3-image stub and `sparse/1` is the real 70-image model. Reading `sparse/0`
# blindly succeeds and returns almost nothing.
_c = glob.glob(f"{ROOT}/sparse/*/images.bin") + glob.glob(f"{ROOT}/sparse/images.bin")
if not _c: raise SystemExit(f"no COLMAP model under {ROOT}/sparse")
MODEL = os.path.dirname(max(_c, key=lambda f: struct.unpack("<Q", open(f, "rb").read(8))[0]))
print(f"  using model {MODEL}")
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def qR(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])

with open(f"{MODEL}/cameras.bin","rb") as f:
    struct.unpack('<Q',f.read(8)); _,_,W,H = struct.unpack('<iiQQ',f.read(24))
    fpx,cx,cy,_ = struct.unpack('<4d',f.read(32))
print(f"  camera {W}x{H} f={fpx:.1f}")

pts={}
with open(f"{MODEL}/points3D.bin","rb") as f:
    n=struct.unpack('<Q',f.read(8))[0]
    for _ in range(n):
        pid=struct.unpack('<Q',f.read(8))[0]
        pts[pid]=np.array(struct.unpack('<3d',f.read(24)))
        f.read(3); struct.unpack('<d',f.read(8))
        tl=struct.unpack('<Q',f.read(8))[0]; f.read(tl*8)

views=[]
with open(f"{MODEL}/images.bin","rb") as f:
    n=struct.unpack('<Q',f.read(8))[0]
    for _ in range(n):
        struct.unpack('<i',f.read(4))
        q=np.array(struct.unpack('<4d',f.read(32))); t=np.array(struct.unpack('<3d',f.read(24)))
        struct.unpack('<i',f.read(4))
        nm=b""
        while True:
            c=f.read(1)
            if c==b'\x00': break
            nm+=c
        m=struct.unpack('<Q',f.read(8))[0]
        obs=[]
        for _ in range(m):
            px,py,pid=struct.unpack('<ddq',f.read(24))
            if pid!=-1 and pid in pts: obs.append((px,py,pid))
        views.append(dict(name=nm.decode(), R=qR(q), t=t, obs=obs))
views.sort(key=lambda v:v["name"])

dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
net = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(dev).eval()

# Scene is ~14 units long; 3 cm voxels resolve a doorframe without exploding.
VOX = float(os.environ.get("TSDF_VOX", "0.03"))
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOX, sdf_trunc=4*VOX,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
intr = o3d.camera.PinholeCameraIntrinsic(W, H, fpx, fpx, cx, cy)
ZMAX = float(os.environ.get("TSDF_ZMAX", "12"))

sel = views[::max(1,len(views)//N)][:N]
used=0; t0=time.time()
for v in sel:
    img = cv2.imread(f"{ROOT}/images/{v['name']}")
    if img is None: continue
    with torch.no_grad():
        out = net(**proc(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_tensors="pt").to(dev))
    d = torch.nn.functional.interpolate(out.predicted_depth[None], size=(H,W),
            mode="bicubic", align_corners=False)[0,0].cpu().numpy()

    P=np.array([pts[p] for _,_,p in v["obs"]]); uv=np.array([(a,b) for a,b,_ in v["obs"]])
    zc=(v["R"]@P.T).T[:,2]+v["t"][2]
    ok=zc>1e-6
    if ok.sum()<12: continue
    u=np.clip(uv[ok,0].astype(int),0,W-1); vv=np.clip(uv[ok,1].astype(int),0,H-1)
    A=np.stack([d[vv,u], np.ones(ok.sum())],1); y=1.0/zc[ok]
    (a,b),*_ = np.linalg.lstsq(A,y,rcond=None)
    r=np.abs(A@[a,b]-y); g=r<3*np.median(r)+1e-9
    (a,b),*_ = np.linalg.lstsq(A[g],y[g],rcond=None)
    rel=float(np.median(np.abs(A[g]@[a,b]-y[g])/np.maximum(y[g],1e-9)))
    if a<=0 or rel>0.25: continue          # flipped or poor fit, see fuse_colmap_depth.py

    Z=1.0/np.maximum(a*d+b,1e-6)
    Z[(a*d+b)<=1e-6]=0
    Z[Z>ZMAX]=0                             # far field is noise at this baseline

    rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB).copy()),
        o3d.geometry.Image(Z.astype(np.float32)),
        depth_scale=1.0, depth_trunc=ZMAX, convert_rgb_to_intensity=False)
    E=np.eye(4); E[:3,:3]=v["R"]; E[:3,3]=v["t"]      # world -> camera
    vol.integrate(rgbd, intr, E)
    used+=1

print(f"  integrated {used}/{len(sel)} frames in {time.time()-t0:.0f}s")
mesh = vol.extract_triangle_mesh()
mesh.compute_vertex_normals()
print(f"  mesh {len(mesh.vertices)} verts / {len(mesh.triangles)} tris")

# Keep only the largest connected component: stray blobs float where a single
# frame's depth was wrong and nothing contradicted it.
tri_id, cnt, _ = mesh.cluster_connected_triangles()
tri_id=np.asarray(tri_id); cnt=np.asarray(cnt)
mesh.remove_triangles_by_mask(cnt[tri_id] < max(cnt)*0.02)
mesh.remove_unreferenced_vertices()
print(f"  {len(mesh.vertices)} verts after dropping small components")
o3d.io.write_triangle_mesh(OUT, mesh)
print(f"  wrote {OUT}")
