"""
Forensic *video* quality for the synthetic face set.

For every finished identity, each edited probe image (the 'hq' scene) is turned
into short CCTV / doorbell / body-cam style clips:

  1. the face is scaled to a target width in pixels (as in real footage, where
     the face is often only 15-60 px wide) and placed in a camera frame;
  2. a few seconds of motion are simulated (subject walking / camera shake,
     slight approach) with exposure motion blur;
  3. sensor effects: IR night mode (monochrome, noise, bloom), interlacing of
     analogue cameras, rolling-shutter-like skew for body-cams, sensor noise;
  4. real codec compression with ffmpeg/libx264 at CCTV bitrates and GOP
     structure;
  5. frames are decoded again: the I-frame (best quality) and the last
     P-frame (most compression drift) are stored as stills, plus the .mp4 clip
     itself for video-to-image comparisons.

Results per identity go to video.json (separate from identity.json, so this can
run while generation is still going); `build finalize` merges them.

    python -m facebias.synth.video data/synth1000            # all profiles
    python -m facebias.synth.video data/synth1000 --profiles cctv_d1 cctv_ir

Requires ffmpeg (imageio-ffmpeg provides it: pip install imageio-ffmpeg).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# name: frame size, fps, bitrate, face width range (px), effects
PROFILES = {
    "cctv_d1":      dict(size=(704, 576),   fps=12, kbps=384,  face=(18, 34), ir=False, interlace=False, shake=0.5, walk=2.0, noise=4),
    "cctv_analog":  dict(size=(720, 576),   fps=25, kbps=1500, face=(28, 48), ir=False, interlace=True,  shake=0.3, walk=2.5, noise=6, chroma_blur=3),
    "cctv_ir":      dict(size=(1280, 720),  fps=15, kbps=900,  face=(24, 44), ir=True,  interlace=False, shake=0.3, walk=1.5, noise=9),
    "cctv_720p":    dict(size=(1280, 720),  fps=15, kbps=1200, face=(36, 64), ir=False, interlace=False, shake=0.4, walk=2.0, noise=3),
    "doorbell":     dict(size=(1536, 1152), fps=15, kbps=1500, face=(90, 160),ir=False, interlace=False, shake=0.2, walk=1.0, noise=5, approach=0.012),
    "bodycam":      dict(size=(1920, 1080), fps=30, kbps=4000, face=(60, 120),ir=False, interlace=False, shake=4.0, walk=3.0, noise=4, skew=0.04),
}
N_FRAMES = 24
SOURCE_EDITS = None  # None = every edited probe except ref2


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class FaceBox:
    """Face box detector (CPU, so it never competes with a running GPU job)."""
    def __init__(self):
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                    providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception as e:  # fallback: centred face of known width
            print("insightface not available, using centred-face fallback:", e)
            self.app = None

    def __call__(self, img: Image.Image, fallback_width: float):
        if self.app is not None:
            import cv2
            faces = self.app.get(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
            if faces:
                f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                x0, y0, x1, y1 = map(float, f.bbox)
                return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0
        w, h = img.size
        return w / 2, h * 0.42, fallback_width


def make_frames(img: Image.Image, cx, cy, fw, prof, rng):
    """Render N_FRAMES camera frames (uint8) with motion and sensor effects.

    The generated scene (1024 px) is narrower than a wide CCTV view once the face
    is scaled down to 20-60 px, so we render the region of interest (ROI) of the
    camera frame that the scene covers - which is also what an examiner crops.
    The bitrate is scaled with the ROI area, so compression per pixel equals that
    of the full camera frame. Returns (frames, face_px, roi_size, kbps)."""
    W, H = prof["size"]
    target = rng.uniform(*prof["face"])
    sc0 = target / max(fw, 1.0)
    even = lambda v: max(16, int(v) // 2 * 2)
    rw, rh = even(min(W, img.width * sc0 * 0.9)), even(min(H, img.height * sc0 * 0.9))
    sw, sh = rw / sc0, rh / sc0                       # ROI size in source pixels
    # ROI centre in source coordinates: around the face, kept inside the image
    ox = cx + rng.uniform(-0.25, 0.25) * sw
    oy = cy + rng.uniform(-0.15, 0.25) * sh
    ang = rng.uniform(0, 2 * np.pi)
    drift = prof["walk"] / sc0                        # source px per frame
    vx, vy = np.cos(ang) * drift, np.sin(ang) * drift * 0.4
    approach = prof.get("approach", 0.004)
    src = img.convert("RGB")
    frames = []
    for t in range(N_FRAMES):
        sub = []
        for s in (0.0, 0.33, 0.66):                   # exposure: 3 sub-steps -> motion blur
            tt = t + s
            sc = sc0 * (1 + approach * tt)
            hw, hh = rw / sc / 2, rh / sc / 2
            mx = np.clip(ox + vx * tt + rng.normal(0, prof["shake"]) / sc, hw, img.width - hw)
            my = np.clip(oy + vy * tt + rng.normal(0, prof["shake"]) / sc, hh, img.height - hh)
            skew = prof.get("skew", 0.0) * np.sin(tt / 3)   # rolling-shutter-like shear
            # affine: frame pixel (u,v) -> source pixel
            a, b, c = 1 / sc, skew / sc, mx - hw - skew * hh
            d, e, f = 0, 1 / sc, my - hh
            fr = src.transform((rw, rh), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)
            sub.append(np.asarray(fr, dtype=np.float32))
        fr = np.mean(sub, axis=0)
        if prof["ir"]:
            g = fr @ np.array([0.30, 0.59, 0.11], np.float32)
            g = 255 * (g / 255) ** 0.7                  # IR: skin brighter and flatter
            bloom = np.asarray(Image.fromarray(g.clip(0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(max(1, rw / 120))), np.float32)
            g = 0.8 * g + 0.25 * bloom
            fr = np.repeat(g[..., None], 3, axis=2)
        fr += rng.normal(0, prof["noise"], fr.shape)
        if prof.get("chroma_blur"):
            lum = np.array([0.299, 0.587, 0.114], np.float32)
            blur = np.asarray(Image.fromarray(fr.clip(0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(prof["chroma_blur"])), np.float32)
            fr = blur + (fr @ lum - blur @ lum)[..., None]   # keep luma detail, smear colour
        frames.append(fr.clip(0, 255).astype(np.uint8))
    if prof["interlace"]:                             # combine fields of t and t+1
        out = []
        for t in range(N_FRAMES - 1):
            f = frames[t].copy(); f[1::2] = frames[t + 1][1::2]; out.append(f)
        frames = out + [frames[-1]]
    kbps = max(24, int(prof["kbps"] * (rw * rh) / (W * H)))
    return frames, target, (rw, rh), kbps


def encode_decode(frames, prof, kbps, mp4_path: Path, tmp: Path):
    ff = ffmpeg_exe()
    for i, f in enumerate(frames):
        Image.fromarray(f).save(tmp / f"in_{i:03d}.png")
    k = kbps
    cmd = [ff, "-y", "-loglevel", "error", "-framerate", str(prof["fps"]), "-i", str(tmp / "in_%03d.png"),
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-b:v", f"{k}k", "-maxrate", f"{k}k", "-bufsize", f"{k // 2}k",
           "-g", str(N_FRAMES), "-bf", "0", str(mp4_path)]
    subprocess.run(cmd, check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4_path), str(tmp / "out_%03d.png")], check=True)
    outs = sorted(tmp.glob("out_*.png"))
    return outs[0], outs[-1]   # I-frame, last P-frame


_FB = None


def _init_worker():
    global _FB
    _FB = FaceBox()


def _one(args):
    jf, profiles, seed = args
    try:
        return _process_identity(Path(jf), profiles, seed, _FB)
    except Exception as e:  # never stop a long batch on one bad file
        return f"{jf}: ERROR {e}"


def process(root: Path, profiles, seed=0, limit=None, max_seconds=None, workers=1):
    files = sorted(str(p) for p in root.glob("identities/*/identity.json"))
    if limit:
        files = files[:limit]
    if workers <= 1:
        _init_worker()
        t0, n = time.time(), 0
        for jf in files:
            if max_seconds and time.time() - t0 > max_seconds:
                break
            msg = _one((jf, profiles, seed))
            if msg: print(msg); n += 1
        return n
    from concurrent.futures import ProcessPoolExecutor
    n = 0
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        for msg in ex.map(_one, [(jf, profiles, seed) for jf in files], chunksize=4):
            if msg: print(msg, flush=True); n += 1
    return n


def _process_identity(jf: Path, profiles, seed, fb):
    if True:
        r = json.loads(jf.read_text())
        if r["status"] != "done":
            return None
        d = jf.parent
        vf = d / "video.json"
        vrec = json.loads(vf.read_text()) if vf.exists() else {"images": [], "clips": [], "done": []}
        todo = [im for im in r["images"] if im["condition"] == "hq" and im["edit"] != "ref2"
                and (SOURCE_EDITS is None or im["edit"] in SOURCE_EDITS)]
        changed = False
        for im in todo:
            src = d / Path(str(im["path"]).replace("\\", "/")).name
            img = None
            for pname in profiles:
                key = f"{im['edit']}|{pname}"
                if key in vrec["done"]:
                    continue
                if img is None:
                    img = Image.open(src).convert("RGB")
                    cx, cy, fw = fb(img, im.get("face_width_px") or 300)
                rng = np.random.default_rng(int(hashlib.sha256(f"{r['identity_id']}|{key}|{seed}".encode()).hexdigest()[:8], 16))
                frames, target, roi, kbps = make_frames(img, cx, cy, fw, PROFILES[pname], rng)
                base = f"SYNTH_{r['identity_id']}__{im['edit']}__{pname}"
                with tempfile.TemporaryDirectory() as tmp:
                    mp4 = d / f"{base}.mp4"
                    i_png, p_png = encode_decode(frames, PROFILES[pname], kbps, mp4, Path(tmp))
                    for tag, p in (("I", i_png), ("P", p_png)):
                        dst = d / f"{base}__{tag}.png"
                        shutil.copy(p, dst)
                        vrec["images"].append({"path": str(dst), "edit": im["edit"], "condition": f"{pname}_{tag}",
                                               "video_profile": pname, "frame": tag, "face_width_px": round(target, 1),
                                               "roi_size": list(roi), "kbps_roi": kbps,
                                               "sim_to_ref": im["sim_to_ref"], "source": src.name})
                vrec["clips"].append({"path": str(mp4), "edit": im["edit"], "profile": pname,
                                      "face_width_px": round(target, 1), "roi_size": list(roi), "kbps_roi": kbps, **{k: v for k, v in PROFILES[pname].items()
                                                                            if k in ("size", "fps", "kbps")}})
                vrec["done"].append(key)
                changed = True
        if changed:
            vf.write_text(json.dumps(vrec, indent=1))
            return f"{time.strftime('%H:%M:%S')} [video] {r['identity_id']}: {len(vrec['clips'])} clips"
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--profiles", nargs="*", default=list(PROFILES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args(argv)
    n = process(Path(a.dir), a.profiles, limit=a.limit, max_seconds=a.max_seconds, workers=a.workers)
    print(f"klaar: {n} personen verwerkt")


if __name__ == "__main__":
    main()
