"""
Quality control and privacy checks for the synthetic set.

1. FaceCheck      exactly one face, minimum size, detector confidence, yaw;
                  embedding from an *independent* recogniser (default
                  insightface buffalo_l; NOT System C, which is under test).
2. identity check probe vs. reference similarity >= threshold (drops edits
                  where the editor changed the person). Rejection rates are
                  logged per group: if the filter rejects one group more often,
                  that is itself a bias in the data set and must be reported.
3. VLMCheck       local vision-language model verifies the intended
                  attributes (gender presentation, age, headwear, glasses,
                  beard, single person, photorealism). Optional.
4. dedup          removes near-duplicate *identities* (generator mode
                  collapse). Use a high threshold: real populations also contain
                  look-alikes, and removing them would make non-mated scores -
                  and thus LRs - optimistic.
5. leakage check  GDPR: compares every synthetic reference face against a
                  gallery of real faces (e.g. public datasets the generators may
                  have seen: FFHQ, CelebA, LFW, celebrity sets). Faces too close
                  to a real person are removed (possible memorisation).
6. OFIQ           ISO/IEC 29794-5 quality via BSI's open-source OFIQ binary
                  (hook; stores a unified quality score for quality-matched
                  calibration subsets).

Note: insightface's pre-trained models are licensed for non-commercial
research; check that this fits your use.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


class FaceCheck:
    def __init__(self, model="buffalo_l", min_face_px=80, min_det=0.6):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name=model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.min_face_px, self.min_det = min_face_px, min_det

    def __call__(self, img: Image.Image, require_size=True) -> dict:
        import cv2
        faces = self.app.get(cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR))
        faces = [f for f in faces if f.det_score >= self.min_det]
        if len(faces) != 1:
            return {"ok": False, "reason": f"{len(faces)} faces"}
        f = faces[0]
        w = float(f.bbox[2] - f.bbox[0])
        if require_size and w < self.min_face_px:
            return {"ok": False, "reason": f"face too small ({w:.0f}px)"}
        yaw = float(f.pose[1]) if getattr(f, "pose", None) is not None else np.nan
        return {"ok": True, "emb": f.normed_embedding.astype(np.float32), "face_width_px": w,
                "det_score": float(f.det_score), "yaw": yaw,
                "bbox": [float(v) for v in f.bbox]}


class MockFaceCheck:
    """For pipeline tests without models: random but identity-consistent embeddings."""
    def __init__(self, **_):
        self.rng = np.random.default_rng(0)

    def __call__(self, img, require_size=True):
        arr = np.asarray(img, dtype=np.float32).ravel()[:512]
        e = arr - arr.mean(); e /= np.linalg.norm(e) + 1e-9
        return {"ok": True, "emb": e, "face_width_px": 300.0, "det_score": 0.99, "yaw": 0.0,
                "bbox": [0, 0, 1, 1]}


# --------------------------------------------------------------------------
VLM_QUESTION = (
    "Describe the person in this photo. Answer ONLY with JSON with keys: "
    '"n_people" (int), "photorealistic" (true/false: indistinguishable from a real photo), '
    '"artifacts" (true/false: visible generation errors on the face, hands, ears, teeth), '
    '"gender_presentation" ("woman"|"man"|"androgynous"), "apparent_age" (int), '
    '"headwear" (short text or "none"), "glasses" (true/false), "facial_hair" (true/false), '
    '"face_occluded" (true/false), '
    '"pointed_ears" (true/false: an ear ends in a sharp elf-like point instead of a normal rounded human ear), '
    '"monk_skin_tone" (int 1-10 on the Monk Skin Tone scale, 1 = lightest, 10 = darkest).')

HEADWEAR_KEYS = {"hijab": ["hijab", "headscarf", "scarf"], "dastar (Sikh turban)": ["turban", "dastar"],
                 "kippah": ["kippah", "yarmulke", "skullcap", "kipa"], "kufi / taqiyah": ["kufi", "taqiyah", "cap", "skullcap"],
                 "tichel / headscarf": ["tichel", "headscarf", "scarf", "head wrap", "wrap", "snood", "bandana", "kerchief", "head covering"], "secular cap": ["cap", "hat"]}


EAR_Q = ("This is a close-up of the side of a person's head. Look at the ear. Answer ONLY with JSON "
         '{"ear_visible": true/false, "pointed": true/false}. "pointed" is true if the top of the ear '
         "narrows to a tip or point (elf-like, even slightly), false for a normal rounded human ear rim.")


def ear_crops(img: Image.Image, bbox, size=448):
    """Left/right crops around the ears, relative to the face box (insightface bbox
    excludes the ears; ears lie just outside it, upper 3/4 of the box height)."""
    x0, y0, x1, y1 = bbox; w, h = x1 - x0, y1 - y0
    out = []
    for cx0, cx1 in ((x0 - 0.35 * w, x0 + 0.08 * w), (x1 - 0.08 * w, x1 + 0.35 * w)):
        box = (int(max(0, cx0)), int(max(0, y0 - 0.05 * h)), int(min(img.width, cx1)), int(min(img.height, y0 + 0.75 * h)))
        c = img.crop(box)
        if c.width < 8 or c.height < 8:
            continue
        s = size / max(c.size)
        out.append(c.resize((max(1, int(c.width * s)), max(1, int(c.height * s)))))
    return out


class VLMCheck:
    def __init__(self, model="Qwen/Qwen2.5-VL-7B-Instruct", age_tol=12, skin_tol=None):
        from transformers import pipeline
        # Put the VLM on ONE explicit GPU (the last one). With device_map="auto" next to the
        # T2I model, accelerate sometimes split it over GPU0/CPU and the pipeline then tried to
        # move it all to cuda:0 -> CUDA OOM at chunk start (19-20 Oct 2026).
        import torch
        n = torch.cuda.device_count()
        try:
            if n > 1:   # explicit device: the pipeline moves the whole model to that GPU
                self.pipe = pipeline("image-text-to-text", model=model, device=n - 1, torch_dtype=torch.bfloat16)
            else:
                self.pipe = pipeline("image-text-to-text", model=model, device_map="auto", torch_dtype=torch.bfloat16)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.pipe = pipeline("image-text-to-text", model=model, device_map="auto", torch_dtype=torch.bfloat16)
        self.age_tol = age_tol
        self.skin_tol = skin_tol   # reject a reference that looks this many Monk steps LIGHTER than requested

    def ask(self, img):
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                                            {"type": "text", "text": VLM_QUESTION}]}]
        txt = self.pipe(text=msg, max_new_tokens=200, return_full_text=False)[0]["generated_text"]
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else {}

    def pointed_ears(self, img, bbox) -> tuple:
        """Ask the VLM about each ear crop separately (whole-face questions miss it)."""
        raws = []
        for c in ear_crops(img.convert("RGB"), bbox):
            msg = [{"role": "user", "content": [{"type": "image", "image": c}, {"type": "text", "text": EAR_Q}]}]
            txt = self.pipe(text=msg, max_new_tokens=40, return_full_text=False)[0]["generated_text"]
            m = re.search(r"\{.*\}", txt, re.S)
            try:
                r = json.loads(m.group(0)) if m else {}
            except ValueError:
                r = {}
            raws.append(r)
            if r.get("ear_visible", True) and r.get("pointed") is True:
                return True, raws
        return False, raws

    def __call__(self, img, a: dict, is_reference: bool, bbox=None) -> dict:
        try:
            r = self.ask(img)
        except Exception as e:  # never crash a long run on a parse error
            return {"vlm_ok": False, "vlm_reason": f"vlm error {e}"}
        # Hard checks = things the labels depend on. Photorealism/artifact judgements of
        # VLMs proved unreliable (pilot: ~all references flagged), so they are only logged.
        problems = []
        if r.get("n_people", 1) != 1: problems.append("people")
        if is_reference:
            if r.get("gender_presentation") not in (a["gender_presentation"], None) and a["gender_presentation"] != "androgynous":
                problems.append("gender")
            tol = max(self.age_tol, 0.3 * a["age"])   # VLMs underestimate the age of older people
            try:
                if abs(int(r.get("apparent_age", a["age"])) - a["age"]) > tol: problems.append("age")
            except (TypeError, ValueError):
                pass
            hw = str(r.get("headwear", "none")).lower()
            if a["headwear"] == "none" and hw not in ("none", "", "no"): problems.append("unwanted headwear")
            if a["headwear"] != "none" and not any(k in hw for k in HEADWEAR_KEYS[a["headwear"]]):
                problems.append("headwear missing")
            if bool(r.get("glasses")) != (a["glasses"] != "none"): problems.append("glasses")
            if r.get("face_occluded", False): problems.append("occluded")
            if r.get("pointed_ears", False) is True: problems.append("pointed ears")
            if self.skin_tol is not None and "monk_skin_tone" in a:
                try:
                    if a["monk_skin_tone"] - int(r.get("monk_skin_tone", a["monk_skin_tone"])) > self.skin_tol:
                        problems.append("skin too light")
                except (TypeError, ValueError):
                    pass
        if is_reference and not problems and bbox is not None:
            try:
                pe, raws = self.pointed_ears(img, bbox)
                r["ear_check"] = raws
                if pe: problems.append("pointed ears")
            except Exception as e:
                r["ear_check"] = f"error {e}"
        return {"vlm_ok": not problems, "vlm_reason": ";".join(problems), "vlm_raw": json.dumps(r)}


# --------------------------------------------------------------------------
def near_duplicate_identities(ids: list, embs: np.ndarray, threshold: float) -> set:
    """Remove as few identities as possible so that no two remaining identities
    are more similar than `threshold` (greedy maximum independent set: repeatedly
    keep the identity with the fewest look-alikes and drop its look-alikes)."""
    E = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    adj = S > threshold
    alive = np.ones(len(ids), bool)
    keep = []
    while alive.any():
        deg = np.where(alive, adj[:, alive].sum(1), np.iinfo(np.int64).max)
        i = int(np.argmin(deg))
        keep.append(i)
        alive[i] = False
        alive[adj[i]] = False
    kept = {ids[i] for i in keep}
    return {i for i in ids if i not in kept}


def build_real_gallery(folder: str, facecheck, out_npy: str, max_images=None):
    """Embed a folder of REAL face images once (processing of biometric data:
    do this under the appropriate legal basis, keep only the embeddings)."""
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    embs = []
    for p in paths[:max_images]:
        r = facecheck(Image.open(p), require_size=False)
        if r["ok"]:
            embs.append(r["emb"])
    np.save(out_npy, np.stack(embs))
    return len(embs)


def leakage_scores(embs: np.ndarray, gallery: np.ndarray, batch=4096) -> np.ndarray:
    g = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    e = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    out = np.full(len(e), -1.0)
    for i in range(0, len(g), batch):
        out = np.maximum(out, (e @ g[i:i + batch].T).max(axis=1))
    return out


def ofiq_scores(paths: list, command: str | None) -> dict:
    """Hook for BSI OFIQ (github.com/BSI-OFIQ/OFIQ-Project). `command` must accept a
    text file with image paths and print/write 'path,unified_quality' lines.
    Wrap OFIQSampleApp accordingly; returns {} when not configured."""
    if not command:
        return {}
    lst = Path("ofiq_inputs.txt"); lst.write_text("\n".join(map(str, paths)))
    out = subprocess.run(command.split() + [str(lst)], capture_output=True, text=True, check=True).stdout
    res = {}
    for line in out.splitlines():
        if "," in line:
            p, q = line.rsplit(",", 1)
            try: res[p.strip()] = float(q)
            except ValueError: pass
    return res
