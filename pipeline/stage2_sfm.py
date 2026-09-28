"""Stage 2 — frames in, camera poses and a sparse point cloud out. This is COLMAP.

THE ONE DECISION THAT MATTERS: every clip is matched against every other clip.

Sequential matching alone links frame N to frame N+1 inside one clip. It never links
clip A to clip C, so three clips give three reconstructions in three unrelated
coordinate frames, and no amount of post-hoc fitting reliably merges them
(docs/robot1-reconstruction.md measured 65% scale error trying). Matching across
clips is what fuses them into ONE world.

Cost: cross pairs grow as the product of frame counts. 85x40 + 85x58 + 40x58 = 10,650
pairs here, 11 minutes on CPU. Worth it.

Non-obvious COLMAP behaviour this script works around:
  - 4.x renamed SiftExtraction/SiftMatching to FeatureExtraction/FeatureMatching for
    SOME options and not others. Wrong name = hard parse error, empty database, and a
    mapper that reports "no images with matches" like a fast success.
  - The Homebrew bottle has no GPU support, so use_gpu must be 0 explicitly.
  - tri_min_angle defaults to 1.5 deg and two-view tracks are discarded by default.
    Forward motion triangulates at near-zero angle, so COLMAP finds the structure and
    then throws it away. Lowering both is what turns excellent poses into a cloud.
"""
import argparse, itertools, os, subprocess, collections, sys


def run(cmd, log):
    print("  $", " ".join(cmd[:3]), "...")
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


def pair_list(images, path, within=15, quadratic=(24, 32, 48, 64)):
    by = collections.defaultdict(list)
    for n in sorted(os.listdir(images)):
        if n.lower().endswith((".jpg", ".png")):
            by[n[0]].append(n)
    pairs, seq = set(), 0
    for lst in by.values():
        for i in range(len(lst)):
            for d in list(range(1, within + 1)) + list(quadratic):
                if i + d < len(lst):
                    pairs.add((lst[i], lst[i + d]))
    seq = len(pairs)
    for va, vb in itertools.combinations(sorted(by), 2):
        for x in by[va]:
            for y in by[vb]:
                pairs.add((x, y))
    with open(path, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")
    return {k: len(v) for k, v in sorted(by.items())}, seq, len(pairs) - seq


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, help="working directory for db, pairs, sparse")
    ap.add_argument("--max-image-size", type=int, default=1600)
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    db = f"{a.out}/db.db"
    L = lambda n: f"{a.out}/logs_{n}.txt"

    if not a.skip_extract:
        if os.path.exists(db):
            os.remove(db)
        print("[1/3] feature extraction — one shared camera for every clip")
        run(["colmap", "feature_extractor", "--database_path", db, "--image_path", a.images,
             "--ImageReader.single_camera", "1",
             "--ImageReader.camera_model", "SIMPLE_RADIAL",
             "--FeatureExtraction.use_gpu", "0",
             "--FeatureExtraction.max_image_size", str(a.max_image_size),
             "--SiftExtraction.max_num_features", str(a.max_features)], L("extract"))

    counts, seq, cross = pair_list(a.images, f"{a.out}/pairs.txt")
    print(f"[2/3] matching — clips {counts}, {seq} within-clip + {cross} cross-clip pairs")
    run(["colmap", "matches_importer", "--database_path", db,
         "--match_list_path", f"{a.out}/pairs.txt", "--match_type", "pairs",
         "--FeatureMatching.use_gpu", "0", "--SiftMatching.max_ratio", "0.85"], L("match"))

    sparse = f"{a.out}/sparse"
    os.makedirs(sparse, exist_ok=True)
    print("[3/3] mapper — all clips at once")
    run(["colmap", "mapper", "--database_path", db, "--image_path", a.images,
         "--output_path", sparse,
         "--Mapper.tri_min_angle", "0.1",
         "--Mapper.tri_ignore_two_view_tracks", "0",
         "--Mapper.filter_min_tri_angle", "0.1"], L("mapper"))

    for d in sorted(os.listdir(sparse)):
        p = os.path.join(sparse, d)
        if os.path.isdir(p):
            subprocess.run(["colmap", "model_converter", "--input_path", p,
                            "--output_path", p, "--output_type", "TXT"], check=True)
    print(f"done. text models under {sparse}/  — read the LARGEST, not sparse/0")


if __name__ == "__main__":
    main()
