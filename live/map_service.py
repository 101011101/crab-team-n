"""Frames in, a 2D map out. Runs in its own thread because it is slow.

    robot frames  ->  COLMAP image_registrator  ->  updated 3D model
                                                          |
                                                   stage4_map2d
                                                          |
                                                    map2d.json  (atomic swap)

HOW SLOW: feature extraction is 5-9 s per frame on this CPU, and registration needs
matching against the prior model on top. So the 3D world updates on the order of
TENS OF SECONDS, not at frame rate. COLMAP cannot run in a control loop and nothing
here pretends otherwise.

That is why this is a separate thread with an atomic handoff: the control loop reads
the newest COMPLETE map every tick at 10 Hz and never waits for a reconstruction. A
map that is 30 seconds old is worth far more than a control loop that stalls.

WHAT IS PROVEN AND WHAT IS NOT
  proven      the offline chain. 179/183 frames, 0.52 px, a map, a working loop.
  NOT proven  registering 320x240 ESP32 frames against a model built from 1080p
              phone frames. Different camera, 20x fewer features. This is the weak
              link. `stats()` reports the registration success rate; watch it, and
              if it sits at zero the system is running on dead reckoning alone.
"""
import json, os, queue, shutil, subprocess, tempfile, threading, time
import numpy as np
import cv2

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
import world_io                      # noqa: E402
from colmap_io import read_images     # noqa: E402
import stage4_map2d as MAP           # noqa: E402


class MapService(threading.Thread):
    def __init__(self, work, prior_sparse, out_map, every=25.0, batch=6,
                 max_image_size=1600, prior_stride=3):
        super().__init__(daemon=True)
        self.work, self.prior, self.out = work, prior_sparse, out_map
        self.every, self.batch, self.mis = every, batch, max_image_size
        self.prior_stride = prior_stride
        self.last_pairs = 0
        self.inbox = queue.Queue(maxsize=64)
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.map_path = out_map if os.path.exists(out_map) else None
        self.poses = {}              # robot name -> (x, y) in map units
        self.n_submitted = self.n_registered = self.n_rounds = 0
        self.last_error = ""
        self.last_round_s = 0.0
        os.makedirs(os.path.join(work, "live_images"), exist_ok=True)

    def submit(self, name, frame):
        if frame is None:
            return False
        """Non-blocking. A full inbox means reconstruction is behind; drop the
        frame rather than stall the thread that is flying the robots."""
        try:
            self.inbox.put_nowait((name, frame, time.time()))
            return True
        except queue.Full:
            return False

    # --- the slow part -------------------------------------------------------
    def run(self):
        while not self.stop_flag.is_set():
            batch = []
            deadline = time.time() + self.every
            while time.time() < deadline and len(batch) < self.batch:
                try:
                    batch.append(self.inbox.get(timeout=0.5))
                except queue.Empty:
                    pass
                if self.stop_flag.is_set():
                    return
            if not batch:
                continue
            t0 = time.time()
            try:
                self._round(batch)
            except Exception as e:                      # never kill the thread
                self.last_error = f"{type(e).__name__}: {e}"
            self.last_round_s = time.time() - t0
            self.n_rounds += 1

    def _round(self, batch):
        img_dir = os.path.join(self.work, "live_images")
        names = []
        for name, frame, ts in batch:
            fn = f"{name}_{int(ts*1000)}.jpg"
            cv2.imwrite(os.path.join(img_dir, fn), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            names.append((name, fn))
        self.n_submitted += len(names)

        db = os.path.join(self.work, "db.db")
        if not os.path.exists(db):
            raise RuntimeError(f"no prior database at {db}; run the offline pipeline first")

        # The robot camera is NOT the phone camera. Give it its own camera model,
        # or COLMAP will force one intrinsic onto two different lenses.
        self._colmap("feature_extractor", "--database_path", db,
                     "--image_path", img_dir,
                     "--ImageReader.single_camera", "1",
                     "--ImageReader.camera_model", "SIMPLE_RADIAL",
                     "--FeatureExtraction.use_gpu", "0",
                     "--FeatureExtraction.max_image_size", str(self.mis),
                     "--image_list_path", self._list([f for _, f in names]))

        # A new frame with no MATCHES cannot be registered, however good its
        # features are. Match each new frame against the model's existing images.
        # Subsampled: 166 priors x 6 new is 996 pairs and blows the round budget,
        # while every 3rd prior still gives every new frame plenty to lock onto.
        model_in, _ = world_io.largest_model(self.prior)
        prior_imgs = sorted(read_images(model_in))[::self.prior_stride]
        pairs = os.path.join(self.work, "_live_pairs.txt")
        with open(pairs, "w") as f:
            for _, new in names:
                for old in prior_imgs:
                    f.write(f"{new} {old}\n")
        self.last_pairs = len(prior_imgs) * len(names)
        self._colmap("matches_importer", "--database_path", db,
                     "--match_list_path", pairs, "--match_type", "pairs",
                     "--FeatureMatching.use_gpu", "0",
                     "--SiftMatching.max_ratio", "0.9")

        out_model = os.path.join(self.work, "live_sparse")
        os.makedirs(out_model, exist_ok=True)
        self._colmap("image_registrator", "--database_path", db,
                     "--input_path", model_in, "--output_path", out_model)
        self._colmap("model_converter", "--input_path", out_model,
                     "--output_path", out_model, "--output_type", "TXT")

        # image_registrator writes a flat model, not sparse/N. Point world_io at a
        # directory holding it either way.
        src = out_model if os.path.exists(os.path.join(out_model, "images.txt")) \
            else self.prior
        w = self._load(src)

        # A registered camera centre IS the robot, in exactly the map's frame. No
        # extrinsic calibration to get wrong, because there is no extrinsic.
        got = 0
        for name, fn in names:
            if fn in w.cam_names:
                c = w.cam[w.cam_names.index(fn)]
                with self.lock:
                    self.poses[name] = (float(c[0]), float(c[2]))
                got += 1
        self.n_registered += got

        m, grid = MAP.build(w)
        tmp = self.out + ".tmp"
        json.dump(m, open(tmp, "w"), separators=(",", ":"))
        os.replace(tmp, self.out)        # atomic: readers never see a half-written map
        with self.lock:
            self.map_path = self.out

    @staticmethod
    def _load(model_dir):
        """world_io.load expects a parent holding sparse/N. image_registrator writes
        the model files directly, so wrap it in a temporary parent view."""
        if os.path.exists(os.path.join(model_dir, "images.txt")):
            parent = os.path.join(os.path.dirname(model_dir), "_asparent")
            os.makedirs(parent, exist_ok=True)
            link = os.path.join(parent, "0")
            if not os.path.exists(link):
                os.symlink(os.path.abspath(model_dir), link)
            return world_io.load(parent)
        return world_io.load(model_dir)

    def _list(self, files):
        p = os.path.join(self.work, "_new.txt")
        open(p, "w").write("\n".join(files) + "\n")
        return p

    def _colmap(self, *args):
        log = os.path.join(self.work, f"live_{args[0]}.txt")
        with open(log, "w") as f:
            subprocess.run(["colmap", *args], stdout=f, stderr=subprocess.STDOUT,
                           check=True, timeout=600)

    def stats(self):
        with self.lock:
            return dict(rounds=self.n_rounds, submitted=self.n_submitted,
                        pairs=self.last_pairs,
                        registered=self.n_registered,
                        rate=round(self.n_registered / max(self.n_submitted, 1), 3),
                        last_round_s=round(self.last_round_s, 1),
                        map=self.map_path, error=self.last_error,
                        poses={k: [round(v[0], 2), round(v[1], 2)]
                               for k, v in self.poses.items()})
