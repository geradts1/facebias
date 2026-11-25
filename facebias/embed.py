"""
Open-source face recognisers as reference systems next to System C.

  adaface : AdaFace IR-101 (Kim et al., CVPR 2022), trained on WebFace12M,
            loaded from Hugging Face via the CVLface wrapper
            (minchul/cvlface_adaface_ir101_webface12m).
  arcface : insightface buffalo_l recogniser (ArcFace loss, ResNet-50, WebFace600K).

Both get the same input: insightface face detection (RetinaFace/SCRFD), 5-point
alignment to the standard 112x112 ArcFace template, one embedding per image.

NB ArcFace (buffalo_l) was also used while *building* the synthetic set
(identity consistency, novelty check, near-duplicate removal). Its scores on
this set are therefore optimistic (mated pairs pre-selected to be similar for
ArcFace, look-alikes pre-removed). AdaFace was not used in generation and is
the fairer open-source reference.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ADAFACE_REPO = "minchul/cvlface_adaface_ir101_webface12m"


class Aligner:
    """Detect the largest face and return a 112x112 aligned BGR crop (or None).
    Retries on a 2x upscaled image for small (CCTV/video) faces."""

    def __init__(self, ctx_id=0, det_size=640):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def __call__(self, bgr):
        import cv2
        from insightface.utils import face_align
        for scale in (1, 2):
            img = bgr if scale == 1 else cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            faces = self.app.get(img)
            if faces:
                f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                return face_align.norm_crop(img, f.kps, image_size=112)
        return None


def _load_cvlface(repo_id, local_dir=None):
    """CVLface models on HF ship their own code (wrapper.py, models/...); they are
    downloaded as plain files (no symlinks, Windows-safe) and loaded from there."""
    from huggingface_hub import snapshot_download
    local_dir = local_dir or str(Path(__file__).resolve().parents[1] / "models" / repo_id.split("/")[-1])
    path = snapshot_download(repo_id, local_dir=local_dir)
    # Build the network directly (what wrapper.py does) instead of AutoModel: the
    # 2023 wrapper is not compatible with current transformers versions.
    cwd = os.getcwd()
    _purge_models_modules()          # every CVLface repo ships its own top-level 'models' package
    os.chdir(path); sys.path.insert(0, path)
    # KP-RPE models try to compile and install a C++ extension (rpe_ops) on import and then call
    # sys.exit(). Register a stub so the pure PyTorch code path is used instead (same results).
    import types
    stub = types.ModuleType("models.vit_kprpe.RPE.rpe_ops.rpe_index")
    stub.RPEIndexFunction = None
    sys.modules[stub.__name__] = stub
    try:
        import yaml
        from omegaconf import OmegaConf
        from models import get_model
        conf = OmegaConf.create(dict(yaml.safe_load(open("pretrained_model/model.yaml"))))
        model = get_model(conf)
        model.load_state_dict_from_path("pretrained_model/model.pt")
    finally:
        os.chdir(cwd); sys.path.remove(path)
        _purge_models_modules()
    return model


def _purge_models_modules():
    for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
        del sys.modules[k]


class AdaFaceEmbedder:
    name = "adaface"

    def __init__(self, device="cuda", repo_id=ADAFACE_REPO, channel_order="rgb"):
        import torch
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = _load_cvlface(repo_id).to(self.device).eval()
        self.channel_order = channel_order   # checked empirically by run_open_matchers.py

    def __call__(self, crops_bgr):
        torch = self.torch
        x = np.stack(crops_bgr).astype(np.float32) / 255.0          # N,112,112,3 (BGR)
        if self.channel_order == "rgb":
            x = x[..., ::-1]
        x = (x - 0.5) / 0.5
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(self.device)
        with torch.inference_mode():
            out = self.model(t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        e = out.float().cpu().numpy().reshape(len(crops_bgr), -1)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


class ArcFaceEmbedder:
    name = "arcface"

    def __init__(self, ctx_id=0):
        from insightface.model_zoo import get_model
        root = Path(os.path.expanduser("~/.insightface/models/buffalo_l"))
        self.model = get_model(str(root / "w600k_r50.onnx"),
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.model.prepare(ctx_id=ctx_id)

    def __call__(self, crops_bgr):
        e = np.stack([self.model.get_feat(c).ravel() for c in crops_bgr])
        return e / np.linalg.norm(e, axis=1, keepdims=True)


class CVLEmbedder(AdaFaceEmbedder):
    """Any CVLface recogniser (same pre-processing as AdaFace: 112x112, (x-0.5)/0.5)."""

    def __init__(self, repo_id, device="cuda", channel_order="rgb"):
        super().__init__(device=device, repo_id=repo_id, channel_order=channel_order)


# 5-point ArcFace template (left eye, right eye, nose, mouth left, mouth right) in the 112x112 crop
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


class KPRPEEmbedder(CVLEmbedder):
    """AdaFace ViT-Base with KeyPoint Relative Position Encoding (Kim et al., CVPR 2024), WebFace12M.
    The network takes the 5 landmarks (normalised x,y in [0,1]) as extra input. Our crops are already
    aligned to the ArcFace template, so the template landmarks are passed for every crop."""

    def __call__(self, crops_bgr):
        torch = self.torch
        x = np.stack(crops_bgr).astype(np.float32) / 255.0
        if self.channel_order == "rgb":
            x = x[..., ::-1]
        x = (x - 0.5) / 0.5
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(self.device)
        kp = torch.from_numpy(np.repeat((ARCFACE_TEMPLATE / 112.0)[None], len(crops_bgr), 0)).to(self.device)
        with torch.inference_mode():
            out = self.model(t, kp)
        if isinstance(out, (tuple, list)):
            out = out[0]
        e = out.float().cpu().numpy().reshape(len(crops_bgr), -1)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


class GlintEmbedder(ArcFaceEmbedder):
    """insightface antelopev2 recogniser: ResNet-100 trained on Glint360K (partial FC). Glint360K has a
    much larger share of East Asian identities than WebFace/MS1M, which is the point of including it."""

    def __init__(self, ctx_id=0):
        from insightface.model_zoo import get_model
        root = Path(os.path.expanduser("~/.insightface/models"))
        hits = sorted(root.glob("antelopev2/**/glintr100.onnx"))
        if not hits:   # official insightface model pack (v0.7 release); unpacks to antelopev2/[antelopev2/]
            import urllib.request, zipfile
            root.mkdir(parents=True, exist_ok=True)
            z = root / "antelopev2.zip"
            if not z.exists():
                urllib.request.urlretrieve(
                    "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip", str(z))
            with zipfile.ZipFile(z) as f:
                f.extractall(root / "antelopev2")
            hits = sorted(root.glob("antelopev2/**/glintr100.onnx"))
        if not hits:
            raise SystemExit("glintr100.onnx not found under %s (antelopev2 download failed)" % root)
        self.model = get_model(str(hits[0]), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.model.prepare(ctx_id=ctx_id)


EMBEDDERS = {"adaface": AdaFaceEmbedder, "arcface": ArcFaceEmbedder}

# reference matchers for the extended study: (kind, HF repo, short description)
MATCHERS = {
    "adaface": ("cvl", ADAFACE_REPO, "AdaFace IR-101, WebFace12M"),
    "arcface": ("insight", None, "ArcFace R50 buffalo_l (WebFace600K) - used while building the set"),
    "adaface_vit": ("cvl", "minchul/cvlface_adaface_vit_base_webface4m", "AdaFace ViT-Base, WebFace4M (transformer)"),
    "arcface_ir101": ("cvl", "minchul/cvlface_arcface_ir101_webface4m", "ArcFace IR-101, WebFace4M"),
    "adaface_ms1mv2": ("cvl", "minchul/cvlface_adaface_ir50_ms1mv2", "AdaFace IR-50, MS1MV2 (celebrity data)"),
    # added Nov 2026: newer / differently trained recognisers
    "adaface_kprpe": ("cvl_kprpe", "minchul/cvlface_adaface_vit_base_kprpe_webface12m",
                      "AdaFace ViT-Base KP-RPE, WebFace12M (CVPR 2024, strong on low quality)"),
    "glint_r100": ("insight_glint", None, "insightface antelopev2 ResNet-100, Glint360K (many East Asian identities)"),
}
NEW_MATCHERS = ["adaface_kprpe", "glint_r100"]


def make_matcher(name, channel_order="rgb"):
    kind, repo, _ = MATCHERS[name]
    if kind == "insight":
        return ArcFaceEmbedder()
    if kind == "insight_glint":
        return GlintEmbedder()
    if kind == "cvl_kprpe":
        return KPRPEEmbedder(repo, channel_order=channel_order)
    return CVLEmbedder(repo, channel_order=channel_order)
