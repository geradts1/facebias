"""
Failure-to-enrol (FTE) analysis with a second, independent face detector.

The first pass (insightface SCRFD-10G, native + 2x upscale) found no face in
~15% of the probe images. This script re-runs every failed image through
OpenCV YuNet (a different architecture and training set) at 1x/2x/4x and:

  1. records per image whether YuNet finds a face (logs/fte_second_detector.csv),
     so FTE can be split into "detector problem" (YuNet finds it) and
     "image problem" (neither finds a face);
  2. aligns the rescued faces with YuNet's 5 landmarks and embeds them with
     AdaFace (and ArcFace), so the pairs can be re-scored
     (data/embeddings/<model>_det2.npz -> used by analyze_by_condition.py --rescued);
  3. writes a contact sheet of failed images per condition (logs/fte_sheet_<cond>.jpg).

    python tools/fte_second_detector.py [--models adaface arcface]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from run_open_matchers import EMB, EXPORT, load_embeddings, local_path, log  # noqa: E402

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"


def yunet():
    import cv2
    if not YUNET.exists():
        YUNET.parent.mkdir(parents=True, exist_ok=True)
        log(f"downloading YuNet -> {YUNET}")
        urllib.request.urlretrieve(YUNET_URL, YUNET)
    return cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), score_threshold=0.6,
                                     nms_threshold=0.3, top_k=50)


def detect(det, bgr):
    """Largest face over scales 1, 2, 4. Returns (score, face_w_px_original, kps5_original) or None."""
    import cv2
    best = None
    for s in (1, 2, 4):
        img = bgr if s == 1 else cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        if max(h, w) > 4000:
            break
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        if faces is not None and len(faces):
            f = max(faces, key=lambda f: f[2] * f[3])
            kps = f[4:14].reshape(5, 2) / s
            best = (float(f[14]), float(f[2] / s), kps)
            break
    return best


def to_arcface_order(kps):
    """YuNet: right eye, left eye, nose, right mouth, left mouth (subject's view).
    ArcFace template: eye with smaller x first, nose, mouth corner with smaller x first."""
    eyes = sorted(kps[[0, 1]].tolist(), key=lambda p: p[0])
    mouth = sorted(kps[[3, 4]].tolist(), key=lambda p: p[0])
    return np.array([eyes[0], eyes[1], kps[2].tolist(), mouth[0], mouth[1]], np.float32)


def main(argv=None):
    import cv2
    from insightface.utils import face_align
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["adaface", "arcface"])
    a = p.parse_args(argv)

    images = pd.read_csv(EXPORT / "images.csv")
    idx, _, ok = load_embeddings(a.models[0])
    failed_files = {f for f, i in idx.items() if not ok[i]}
    fail = images[images.file.isin(failed_files)].reset_index(drop=True)
    log(f"{len(fail)} images without a face in pass 1 (of {len(images)})")

    out_csv = ROOT / "logs" / "fte_second_detector.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["file"])
    todo = fail[~fail.file.isin(set(done.file))]
    det = yunet()
    rows, crops, crop_files = [], [], []
    crop_cache = EMB / "det2_crops.npz"
    if crop_cache.exists():
        z = np.load(crop_cache, allow_pickle=True)
        crops, crop_files = list(z["crops"]), list(z["files"])
    t0 = time.time()
    for k, r in enumerate(todo.itertuples()):
        bgr = cv2.imread(str(local_path(r.path)))
        res = detect(det, bgr) if bgr is not None else None
        rows.append({"file": r.file, "condition": r.condition, "identity_id": r.identity_id,
                     "found": res is not None, "det_score": res[0] if res else np.nan,
                     "face_w_px": res[1] if res else np.nan,
                     "img_w": bgr.shape[1] if bgr is not None else np.nan})
        if res is not None:
            crops.append(face_align.norm_crop(bgr, to_arcface_order(res[2]), image_size=112))
            crop_files.append(r.file)
        if (k + 1) % 2000 == 0 or k + 1 == len(todo):
            df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            df.to_csv(out_csv, index=False)
            np.savez(crop_cache, crops=np.array(crops, np.uint8), files=np.array(crop_files))
            log(f"YuNet {k + 1}/{len(todo)}  found {df.found.astype(bool).mean():.1%}  "
                f"{(k + 1) / (time.time() - t0):.0f} img/s")
    df = pd.read_csv(out_csv)
    df["found"] = df.found.astype(str).str.lower().eq("true")

    # embed rescued faces
    from facebias.embed import EMBEDDERS
    order = json.loads((EMB / "adaface_channel_order.json").read_text())["channel_order"]
    for m in a.models:
        f = EMB / f"{m}_det2.npz"
        if f.exists() or not crops:
            continue
        net = EMBEDDERS[m](**({"channel_order": order} if m == "adaface" else {}))
        e = np.concatenate([net(crops[i:i + 64]) for i in range(0, len(crops), 64)]).astype(np.float16)
        np.savez(f, files=np.array(crop_files), emb=e)
        log(f"{m}: embedded {len(crop_files)} rescued faces -> {f}")
        del net

    # summary: FTE pass 1 vs after YuNet, per condition and group
    meta = pd.read_csv(ROOT / "data/synth1000/metadata.csv",
                       usecols=["identity_id", "ancestry", "gender_presentation", "age_band"]
                       ).drop_duplicates("identity_id")
    img = images.merge(meta, on="identity_id", how="left")
    img["fte1"] = img.file.isin(failed_files)
    rescued = set(df.file[df.found])
    img["fte2"] = img.fte1 & ~img.file.isin(rescued)
    lines = ["# FTE met een tweede gezichtsdetector (OpenCV YuNet)\n",
             f"Pass 1 (SCRFD, 1x+2x): {img.fte1.mean():.1%} van alle beelden zonder gezicht. "
             f"Na YuNet (1x/2x/4x): {img.fte2.mean():.1%}. "
             f"YuNet vond een gezicht in {df.found.mean():.1%} van de mislukte beelden.\n",
             "Interpretatie: wat YuNet wel vindt is een *detectorprobleem* van SCRFD; wat geen van beide vindt "
             "ligt vooral aan het beeld (gezicht te klein, te donker of te sterk gecomprimeerd).\n"]
    for g in ["condition", "ancestry", "gender_presentation", "age_band"]:
        t = img.groupby(g).agg(n=("file", "size"), FTE_pass1=("fte1", "mean"), FTE_after_YuNet=("fte2", "mean"))
        t[["FTE_pass1", "FTE_after_YuNet"]] *= 100
        lines.append(f"\n## per {g} (%)\n")
        try:
            lines.append(t.round(2).to_markdown())
        except ImportError:
            lines.append("```\n" + t.round(2).to_string() + "\n```")
    # face size of rescued faces
    lines.append("\n## gezichtsbreedte (px, originele resolutie) van door YuNet gevonden gezichten\n")
    lines.append("```\n" + df[df.found].groupby("condition").face_w_px.describe()[
        ["count", "25%", "50%", "75%"]].round(1).to_string() + "\n```")
    (ROOT / "results_fte_second_detector.md").write_text("\n".join(lines), encoding="utf-8")
    log("summary -> results_fte_second_detector.md")

    # contact sheets of images that neither detector found (12 per condition)
    neither = img[img.fte2]
    for cond, g in neither.groupby("condition"):
        sel = g.sample(min(12, len(g)), random_state=0)
        tiles = []
        for pth in sel.path:
            im = cv2.imread(str(local_path(pth)))
            if im is not None:
                tiles.append(cv2.resize(im, (256, 256)))
        if tiles:
            while len(tiles) % 6:
                tiles.append(np.full((256, 256, 3), 255, np.uint8))
            sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
            cv2.imwrite(str(ROOT / "logs" / f"fte_sheet_{cond}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    log("done")


if __name__ == "__main__":
    main()
