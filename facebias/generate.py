"""
Generate a demographically balanced synthetic face set for bias testing.

Step 1  Identity seeds: SDXL text-to-image, one passport-style frontal photo
        per synthetic identity, for every cell of the demographic grid
        (sex x age band x ethnicity), N identities per cell.
Step 2  Same-identity variations: SDXL + IP-Adapter-FaceID, conditioned on the
        ArcFace embedding of the seed (pose, expression, lighting, glasses...).
Step 3  Forensic degradations applied in post-processing so they are exactly
        controlled and identical across groups (CCTV resolution, blur, JPEG,
        noise, low light).
Step 4  Quality control: exactly one face detected; identity consistency of
        variations vs. seed (ArcFace cosine) above a threshold; estimated
        sex / age from insightface logged to spot prompt failures.

Output: images/<identity_id>/<image>.png + metadata.csv

Caveats (also in README):
 * The 'ethnicity' label is a prompt, not ground truth. Have a sample
   annotated (or classify with e.g. FairFace) and report agreement.
 * The generator has its own demographic biases (image quality, realism,
   'average face' effects) that can masquerade as matcher bias. Compare the
   per-group image quality (e.g. OFIQ) and repeat the analysis with a real
   balanced data set before drawing conclusions about System C.
 * Identity in step 2 is defined by ArcFace; mated similarity is therefore
   inflated for ArcFace-like matchers. System C is an independent system,
   but do not use ArcFace scores from this set as an unbiased reference.
"""
from __future__ import annotations

import itertools
import io
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class GenConfig:
    out_dir: str = "data/synth"
    n_per_cell: int = 10
    sexes: list = field(default_factory=lambda: ["woman", "man"])
    age_bins: dict = field(default_factory=lambda: {
        "18-30": 25, "31-45": 38, "46-60": 53, "61+": 68})
    ethnicities: list = field(default_factory=lambda: [
        "West African", "East Asian", "South Asian",
        "Northern European", "Middle Eastern", "Latin American"])
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    ip_adapter_repo: str = "h94/IP-Adapter-FaceID"
    ip_adapter_weight: str = "ip-adapter-faceid_sdxl.bin"
    ip_adapter_scale: float = 0.8
    steps: int = 30
    guidance: float = 6.0
    size: int = 1024
    # variations generated with IP-Adapter-FaceID (prompt fragment)
    variations: dict = field(default_factory=lambda: {
        "v_smile": "smiling, frontal view, indoor lighting",
        "v_turn30": "head turned about 30 degrees to the side, neutral expression",
        "v_outdoor": "outdoors, natural daylight, slight shadows on the face",
        "v_glasses": "wearing glasses, neutral expression, frontal view",
        "v_down": "looking slightly downward, overhead lighting",
    })
    # degradations applied to variation images -> conditions used as probes
    degradations: dict = field(default_factory=lambda: {
        "cctv_low": {"face_px": 40, "blur": 1.0, "jpeg": 40, "noise": 6, "gamma": 1.0},
        "cctv_dark": {"face_px": 60, "blur": 0.8, "jpeg": 50, "noise": 10, "gamma": 0.55},
        "hq": None,  # undegraded variation
    })
    min_identity_sim: float = 0.45  # ArcFace cosine variation vs seed
    seed: int = 1234

    @staticmethod
    def from_yaml(path):
        import yaml
        with open(path) as f:
            return GenConfig(**(yaml.safe_load(f).get("generate", {})))


NEG = ("cartoon, painting, illustration, 3d render, deformed, disfigured, extra limbs, "
       "multiple people, text, watermark, makeup heavy, blurry, cropped face")


def seed_prompt(sex, age, eth):
    return (f"passport photograph of a {age}-year-old {eth} {sex}, frontal view, "
            f"neutral expression, plain light grey background, even studio lighting, "
            f"sharp focus, realistic skin texture, natural photograph, 85mm")


def variation_prompt(sex, age, eth, frag):
    return (f"photograph of a {age}-year-old {eth} {sex}, {frag}, "
            f"realistic, natural skin texture, sharp")


# --------------------------------------------------------------------------
# Degradations
# --------------------------------------------------------------------------
def degrade(img: Image.Image, face_width_px: float, face_px=None, blur=0.0, jpeg=None,
            noise=0.0, gamma=1.0, rng=None) -> Image.Image:
    rng = rng or np.random.default_rng()
    w, h = img.size
    if face_px:  # rescale so the face is ~face_px wide, then back up (lost detail)
        f = min(1.0, face_px / max(face_width_px, 1))
        small = img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BICUBIC)
        img = small.resize((w, h), Image.BICUBIC)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if gamma != 1.0:
        arr = np.asarray(img).astype(float) / 255
        img = Image.fromarray((255 * arr ** (1 / gamma)).clip(0, 255).astype(np.uint8))
        img = ImageEnhance.Contrast(img).enhance(0.8)
    if noise:
        arr = np.asarray(img).astype(float) + rng.normal(0, noise, (h, w, 3))
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if jpeg:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(jpeg))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


# --------------------------------------------------------------------------
# Face analysis helper
# --------------------------------------------------------------------------
class FaceQC:
    def __init__(self):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def analyse(self, img: Image.Image):
        import cv2
        faces = self.app.get(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        if len(faces) != 1:
            return None
        f = faces[0]
        return {"emb": f.normed_embedding, "width": float(f.bbox[2] - f.bbox[0]),
                "est_age": float(getattr(f, "age", np.nan)),
                "est_sex": {0: "woman", 1: "man"}.get(int(getattr(f, "gender", -1)), "?"),
                "det_score": float(f.det_score)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def generate(cfg: GenConfig):
    import torch
    from diffusers import StableDiffusionXLPipeline

    out = Path(cfg.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    rng = np.random.default_rng(cfg.seed)
    qc = FaceQC()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(cfg.base_model, torch_dtype=dtype)
    pipe.to(dev)

    # ---------------- Step 1: identity seeds ----------------
    identities = []
    cells = list(itertools.product(cfg.sexes, cfg.age_bins.items(), cfg.ethnicities))
    for sex, (age_bin, age), eth in cells:
        made, attempt = 0, 0
        while made < cfg.n_per_cell and attempt < cfg.n_per_cell * 4:
            attempt += 1
            s = int(rng.integers(0, 2**31 - 1))
            g = torch.Generator(device="cpu").manual_seed(s)
            img = pipe(prompt=seed_prompt(sex, age, eth), negative_prompt=NEG,
                       num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
                       height=cfg.size, width=cfg.size, generator=g).images[0]
            a = qc.analyse(img)
            if a is None:
                continue
            iid = f"{sex[0]}_{age_bin}_{eth.replace(' ', '')}_{made:03d}"
            d = out / "images" / iid
            d.mkdir(exist_ok=True)
            p = d / "reference.png"
            img.save(p)
            identities.append({"identity_id": iid, "sex": sex, "age_bin": age_bin, "age": age,
                               "ethnicity": eth, "seed": s, "emb": a["emb"],
                               "rows": [_row(iid, sex, age_bin, eth, "reference", "reference",
                                             p, s, a, 1.0)]})
            made += 1
        print(f"[seed] {sex} {age_bin} {eth}: {made}/{cfg.n_per_cell}")

    # ---------------- Step 2+3: variations + degradations ----------------
    pipe.load_ip_adapter(cfg.ip_adapter_repo, subfolder=None,
                         weight_name=cfg.ip_adapter_weight, image_encoder_folder=None)
    pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)

    for ident in identities:
        # shape (1, 1, 1, 512) as in the diffusers IP-Adapter-FaceID example
        emb = torch.from_numpy(ident["emb"]).reshape(1, 1, 1, -1)
        id_embeds = torch.cat([torch.zeros_like(emb), emb]).to(dtype=dtype, device=dev)
        for vname, frag in cfg.variations.items():
            s = int(rng.integers(0, 2**31 - 1))
            g = torch.Generator(device="cpu").manual_seed(s)
            img = pipe(prompt=variation_prompt(ident["sex"], ident["age"], ident["ethnicity"], frag),
                       negative_prompt=NEG, ip_adapter_image_embeds=[id_embeds],
                       num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
                       height=cfg.size, width=cfg.size, generator=g).images[0]
            a = qc.analyse(img)
            if a is None:
                continue
            sim = float(np.dot(a["emb"], ident["emb"]))
            if sim < cfg.min_identity_sim:   # identity drift -> discard
                continue
            for dname, dpar in cfg.degradations.items():
                im2 = img if dpar is None else degrade(img, a["width"], rng=rng, **dpar)
                p = out / "images" / ident["identity_id"] / f"{vname}__{dname}.png"
                im2.save(p)
                ident["rows"].append(_row(ident["identity_id"], ident["sex"], ident["age_bin"],
                                          ident["ethnicity"], f"{vname}__{dname}", dname,
                                          p, s, a, sim))
        print(f"[var] {ident['identity_id']}: {len(ident['rows']) - 1} images")

    meta = pd.DataFrame([r for i in identities for r in i["rows"]])
    meta.to_csv(out / "metadata.csv", index=False)
    _qc_summary(meta).to_csv(out / "qc_summary.csv")
    print(f"done: {len(meta)} images, {len(identities)} identities -> {out/'metadata.csv'}")
    return meta


def _row(iid, sex, age_bin, eth, image_type, condition, path, seed, a, sim):
    return {"identity_id": iid, "sex": sex, "age_bin": age_bin, "ethnicity": eth,
            "image_type": image_type, "condition": condition, "path": str(path), "seed": seed,
            "face_width_px": a["width"], "det_score": a["det_score"],
            "est_age": a["est_age"], "est_sex": a["est_sex"], "arcface_sim_to_seed": sim}


def _qc_summary(meta):
    m = meta.assign(sex_ok=meta.est_sex == meta.sex)
    return m.groupby(["sex", "age_bin", "ethnicity"]).agg(
        n_images=("path", "size"), n_ids=("identity_id", "nunique"),
        sex_agreement=("sex_ok", "mean"), est_age_mean=("est_age", "mean"),
        det_score=("det_score", "mean"), id_sim=("arcface_sim_to_seed", "mean"))
