"""
Build the synthetic calibration data set.

    python -m facebias.synth.build run      --config config_synth.yaml [--shard 0 --n-shards 4]
    python -m facebias.synth.build finalize --config config_synth.yaml

`run` is resumable (one folder + identity.json per identity; finished
identities are skipped) and can be split over GPUs with --shard/--n-shards.
`finalize` does the data-set level steps: near-duplicate removal, GDPR
leakage check, OFIQ, metadata.csv (compatible with `facebias.cli pairs`)
and DATASHEET.md.

Default size: 1000 identities x (1 reference + 4 edited images) = 5000
generated faces; the probe edits are additionally degraded to CCTV-like
qualities in code (not new faces, but extra calibration conditions).
"""
from __future__ import annotations

import argparse
import time
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ..generate import degrade
from .attributes import sample_identities
from .prompts import NEGATIVE, PROBE_EDITS, edit_prompt, reference_prompt


@dataclass
class SynthConfig:
    out_dir: str = "data/synth5k"
    n_identities: int = 1000
    edits_per_identity: int = 4            # incl. 'ref2'
    always_ref2: bool = True
    edit_pool: list = field(default_factory=lambda: [e for e in PROBE_EDITS if e != "ref2"])
    degradations: dict = field(default_factory=lambda: {
        "hq": None,
        "cctv_mid": {"face_px": 64, "blur": 0.6, "jpeg": 55, "noise": 5, "gamma": 1.0},
        "cctv_low": {"face_px": 32, "blur": 1.0, "jpeg": 35, "noise": 8, "gamma": 1.0},
    })
    t2i_backend: str = "zimage"
    edit_backend: str = "qwen_edit"
    offload: bool = False
    edit_lightning: bool = True             # 4-step Lightning LoRA for Qwen-Image-Edit
    edit_device_map: str | None = None      # "balanced" = spread editor over all visible GPUs, no CPU offload
    novelty_threshold: float | None = None  # reject a new reference if it looks like an existing identity
    min_ref2_sim: float = 0.60              # second passport photo must clearly be the same person
    balanced_weights: dict | None = None    # override quota weights, e.g. to top up groups
    chunk_size: int = 50                    # identities per refs->edits cycle
    size: int = 1024
    face_model: str = "buffalo_l"
    use_vlm: bool = True
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    min_identity_sim: float = 0.40          # probe vs reference, independent recogniser
    max_ref_attempts: int = 6
    skin_check_tol: int | None = None   # VLM: reject references > this many Monk steps lighter than requested
    twins_of: str | None = None  # csv with identity_id + source_dir: generate an 'attribute twin' of each
                                 # (same attributes and edits, new person/seed) -> hard non-mated pairs
    twin_prefix: str = "tw_"
    ref_seed_salt: str = ""      # change (e.g. 'retry1') to regenerate failed references with new seeds
    save_rejected: bool = True    # keep rejected reference attempts in <id>/_rejected/ for human review
    max_edit_attempts: int = 3
    dedup_threshold: float = 0.65           # near-duplicate identities (high on purpose)
    leakage_gallery_npy: str | None = None  # embeddings of real faces
    leakage_threshold: float = 0.50         # set at the recogniser's FMR ~1e-5 point
    ofiq_command: str | None = None
    seed: int = 2026
    mock: bool = False                      # pipeline test without GPU/models

    @staticmethod
    def from_yaml(path):
        import yaml
        with open(path) as f:
            return SynthConfig(**(yaml.safe_load(f).get("synth", {})))


def _schema(cfg):
    import copy
    from .attributes import DEFAULT_SCHEMA
    sch = copy.deepcopy(DEFAULT_SCHEMA)
    for k, v in (cfg.balanced_weights or {}).items():
        sch["balanced"][k] = v
    return sch


def _twin_identities(cfg: SynthConfig):
    """Attribute twins: for every listed source identity a new identity with exactly the same
    attributes (ancestry, age, gender, face shape, hair, ...) and the same edit list, but a
    different person (new id -> new seeds). Novelty check stays off on purpose."""
    from .attributes import Identity
    src = pd.read_csv(cfg.twins_of)
    ids, eds = [], []
    for r in src.itertuples():
        rec = json.loads((Path(r.source_dir) / "identities" / r.identity_id / "identity.json").read_text())
        attrs = dict(rec["attrs"], twin_of=r.identity_id)
        ids.append(Identity(f"{cfg.twin_prefix}{r.identity_id}", attrs))
        eds.append(rec.get("edits") or ["ref2"])
    return ids, eds


def _assign_edits(cfg: SynthConfig, identities, rng):
    """Balanced: every edit type is used (nearly) equally often, spread over groups."""
    pool = list(rng.permutation(cfg.edit_pool))
    k = cfg.edits_per_identity - (1 if cfg.always_ref2 else 0)
    out = []
    for i, ident in enumerate(identities):
        chosen, j = [], 0
        while len(chosen) < k:
            e = pool[(i * k + j) % len(pool)]; j += 1
            if e in chosen:
                continue
            if e == "hood_up" and ident["headwear"] in ("hijab", "dastar (Sikh turban)", "tichel / headscarf"):
                continue
            chosen.append(e)
        out.append((["ref2"] if cfg.always_ref2 else []) + chosen)
    return out


def _save(img: Image.Image, path: Path, meta: dict):
    info = PngInfo()
    info.add_text("synthetic", "true")
    info.add_text("description", "AI-generated face of a non-existent person; for biometric "
                                 "research only; not a real individual")
    info.add_text("generation", json.dumps(meta))
    img.save(path, pnginfo=info)


def _save_rejected(cfg, d: Path, img: Image.Image, att: int, stage: str, reason) -> None:
    """Keep a small JPEG of a rejected reference attempt (audit of QC bias); never used as data."""
    if not cfg.save_rejected or cfg.mock:
        return
    import re
    rd = d / "_rejected"; rd.mkdir(exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9.]+", "-", str(reason))[:40]
    salt = re.sub(r"[^A-Za-z0-9]+", "", cfg.ref_seed_salt) or "s0"
    im = img.convert("RGB"); im.thumbnail((512, 512))
    im.save(rd / f"REJECTED_SYNTH_{d.name}__{salt}_att{att}_{stage}_{tag}.jpg", quality=88)


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def run(cfg: SynthConfig, shard=0, n_shards=1, max_chunks=None):
    """Returns True if work remains (only possible with max_chunks)."""
    from .backends import EDIT, T2I
    from .qc import FaceCheck, MockFaceCheck, VLMCheck

    out = Path(cfg.out_dir); (out / "identities").mkdir(parents=True, exist_ok=True)
    req = Path("logs/request_ear_scan")
    if req.exists() and not cfg.mock:   # one-off scan of existing references, between chunks
        import subprocess
        req.rename(req.with_suffix(".started"))
        subprocess.run([sys.executable, "-u", "tools/check_pointed_ears.py", "--v2", "--dry-run"])
    (out / "config_used.json").write_text(json.dumps(asdict(cfg), indent=2))
    rng = np.random.default_rng(cfg.seed)
    if cfg.twins_of:
        identities, edits = _twin_identities(cfg)
    else:
        identities = sample_identities(cfg.n_identities, seed=cfg.seed % 100, schema=_schema(cfg))
        edits = _assign_edits(cfg, identities, rng)

    t2i_name = "mock" if cfg.mock else cfg.t2i_backend
    ed_name = "mock" if cfg.mock else cfg.edit_backend
    if ed_name == "qwen_edit" and cfg.edit_lightning:
        import peft  # noqa: F401  fail fast: needed for the Lightning LoRA (pip install peft)
    fc = MockFaceCheck() if cfg.mock else FaceCheck(cfg.face_model)
    size = 256 if cfg.mock else cfg.size
    todo_all = [(ident, ed) for i, (ident, ed) in enumerate(zip(identities, edits)) if i % n_shards == shard]
    bank = None   # embeddings of all existing references (novelty check against doppelgangers)
    if cfg.novelty_threshold:
        embs = [np.load(p) for p in sorted((out / "identities").glob("*/ref_emb.npy"))]
        bank = np.stack(embs) if embs else np.zeros((0, 512), np.float32)
        print(f"novelty check against {len(bank)} existing identities (threshold {cfg.novelty_threshold})")

    def status(ident):
        jf = out / "identities" / ident.identity_id / "identity.json"
        return json.loads(jf.read_text())["status"] if jf.exists() else None

    # Work in chunks (references -> edits per chunk) so finished identities, with all
    # their images, appear early and models are swapped only once per chunk.
    n_chunks_done = 0
    for c0 in range(0, len(todo_all), cfg.chunk_size):
        todo = todo_all[c0:c0 + cfg.chunk_size]
        if all(status(i) in ("done", "reference_failed") or str(status(i)).startswith("excluded") for i, _ in todo):
            continue
        if max_chunks is not None and n_chunks_done >= max_chunks:
            print(f"{time.strftime('%H:%M:%S')} max_chunks={max_chunks} reached; exit for a fresh process")
            return True
        n_chunks_done += 1
        print(f"{time.strftime('%H:%M:%S')} === chunk {c0 // cfg.chunk_size + 1}: identities {c0}-{c0 + len(todo) - 1} of {len(todo_all)} ===")

        # ---------- phase 1: references (T2I + VLM in memory) ----------
        if any(status(i) is None for i, _ in todo):
            t2i = T2I[t2i_name](offload=cfg.offload)
            vlm = VLMCheck(cfg.vlm_model, skin_tol=cfg.skin_check_tol) if (cfg.use_vlm and not cfg.mock) else None
            for ident, ed in todo:
                if status(ident) is not None:
                    continue
                d = out / "identities" / ident.identity_id
                jf = d / "identity.json"
                d.mkdir(exist_ok=True)
                rec = {"identity_id": ident.identity_id, "attrs": ident.attrs, "edits": ed,
                       "status": "reference_failed", "rejections": [], "images": []}
                prompt = reference_prompt(ident.attrs)
                for att in range(cfg.max_ref_attempts):
                    s = (_seed(cfg.seed, ident.identity_id, "ref", att) if not cfg.ref_seed_salt
                         else _seed(cfg.seed, ident.identity_id, "ref", cfg.ref_seed_salt, att))
                    img = t2i.generate(prompt, NEGATIVE, s, size)
                    qc = fc(img)
                    if not qc["ok"]:
                        rec["rejections"].append({"stage": "ref_face", "reason": qc["reason"]})
                        _save_rejected(cfg, d, img, att, "face", qc["reason"]); continue
                    if bank is not None and len(bank):
                        e = qc["emb"] / np.linalg.norm(qc["emb"])
                        nov = float((bank @ e).max()) if True else 0
                        if nov > cfg.novelty_threshold:
                            rec["rejections"].append({"stage": "ref_novelty", "reason": f"lookalike {nov:.2f}"})
                            _save_rejected(cfg, d, img, att, "novelty", f"{nov:.2f}"); continue
                    v = vlm(img, ident.attrs, True, bbox=qc.get("bbox")) if vlm else {"vlm_ok": True, "vlm_reason": ""}
                    if not v["vlm_ok"]:
                        rec["rejections"].append({"stage": "ref_vlm", "reason": v["vlm_reason"],
                                                  "vlm_raw": v.get("vlm_raw", "")})
                        _save_rejected(cfg, d, img, att, "vlm", v["vlm_reason"]); continue
                    p = d / f"SYNTH_{ident.identity_id}__reference.png"
                    _save(img, p, {"model": t2i_name, "seed": s, "prompt": prompt})
                    np.save(d / "ref_emb.npy", qc["emb"])
                    if bank is not None:
                        bank = np.vstack([bank, qc["emb"][None] / np.linalg.norm(qc["emb"])])
                    rec["images"].append({"path": str(p), "edit": "reference", "condition": "reference",
                                          "seed": s, "sim_to_ref": 1.0,
                                          **{k: qc[k] for k in ("face_width_px", "det_score", "yaw")},
                                          "vlm_raw": v.get("vlm_raw", "")})
                    rec["status"] = "reference_ok"
                    break
                jf.write_text(json.dumps(rec, indent=1, default=float))
                print(f"{time.strftime('%H:%M:%S')} [ref] {ident.identity_id}: {rec['status']} ({len(rec['rejections'])} rejections)")
            del t2i, vlm
            _free_gpu()

        # ---------- phase 2: edits (editor only in memory) ----------
        kw = {"lightning": cfg.edit_lightning, "device_map": cfg.edit_device_map} if ed_name == "qwen_edit" else {}
        editor = EDIT[ed_name](offload=cfg.offload, **kw)
        for ident, ed in todo:
            d = out / "identities" / ident.identity_id
            jf = d / "identity.json"
            rec = json.loads(jf.read_text())
            if rec["status"] != "reference_ok":
                continue
            ed = rec.get("edits", ed)   # keep the edit assignment made when the reference was created
            t0 = time.time()
            ref_img = Image.open(d / f"SYNTH_{ident.identity_id}__reference.png").convert("RGB")
            ref_emb = np.load(d / "ref_emb.npy")
            for e in ed:
                prompt = edit_prompt(e, rec.get("attrs", ident.attrs))   # attrs the reference was made with
                for att in range(cfg.max_edit_attempts):
                    s = _seed(cfg.seed, ident.identity_id, e, att)
                    img = editor.edit(ref_img, prompt, NEGATIVE, s).convert("RGB")
                    qc = fc(img, require_size=False)
                    if not qc["ok"]:
                        rec["rejections"].append({"stage": f"edit_face:{e}", "reason": qc["reason"]}); continue
                    sim = float(np.dot(qc["emb"], ref_emb))
                    if sim < cfg.min_identity_sim:
                        rec["rejections"].append({"stage": f"edit_identity:{e}", "reason": f"sim {sim:.2f}"}); continue
                    degs = {"hq": None} if e == "ref2" else cfg.degradations
                    for dname, dpar in degs.items():
                        im2 = img if dpar is None else degrade(img, qc["face_width_px"],
                                                              rng=np.random.default_rng(s), **dpar)
                        p = d / f"SYNTH_{ident.identity_id}__{e}__{dname}.png"
                        _save(im2, p, {"model": ed_name, "lightning": cfg.edit_lightning, "seed": s,
                                       "prompt": prompt, "degradation": dpar})
                        rec["images"].append({"path": str(p), "edit": e,
                                              "condition": "ref2" if e == "ref2" else dname,
                                              "seed": s, "sim_to_ref": sim,
                                              **{k: qc[k] for k in ("face_width_px", "det_score", "yaw")}})
                    break
            rec["status"] = "done"
            jf.write_text(json.dumps(rec, indent=1, default=float))
            print(f"{time.strftime('%H:%M:%S')} [edit] {ident.identity_id}: {len(rec['images'])} images ({time.time() - t0:.0f}s)")
        del editor
        _free_gpu()


def _free_gpu():
    try:
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------
def finalize(cfg: SynthConfig):
    from .qc import leakage_scores, near_duplicate_identities, ofiq_scores

    out = Path(cfg.out_dir)
    recs = [json.loads(p.read_text()) for p in sorted((out / "identities").glob("*/identity.json"))]
    for r in recs:   # merge forensic-video stills produced by facebias.synth.video
        vf = out / "identities" / r["identity_id"] / "video.json"
        if vf.exists() and r["status"] == "done":
            r["images"] += json.loads(vf.read_text())["images"]
    ok = [r for r in recs if r["status"] == "done"]
    ids = [r["identity_id"] for r in ok]
    embs = np.stack([np.load(out / "identities" / i / "ref_emb.npy") for i in ids]) if ids else np.zeros((0, 512))

    dup = near_duplicate_identities(ids, embs, cfg.dedup_threshold) if len(ids) > 1 else set()
    leak = {}
    if cfg.leakage_gallery_npy:
        ls = leakage_scores(embs, np.load(cfg.leakage_gallery_npy))
        leak = dict(zip(ids, ls))
    leaked = {i for i, v in leak.items() if v >= cfg.leakage_threshold}

    rows = []
    for r in recs:
        a = r["attrs"]
        excluded = ("near_duplicate" if r["identity_id"] in dup else
                    "leakage" if r["identity_id"] in leaked else
                    None if r["status"] == "done" else r["status"])
        for im in r["images"]:
            rows.append({"identity_id": r["identity_id"], "path": im["path"], "edit": im["edit"],
                         "condition": im["condition"], "sim_to_ref": im["sim_to_ref"],
                         "face_width_px": im["face_width_px"], "yaw": im.get("yaw"),
                         "max_sim_real_gallery": leak.get(r["identity_id"], np.nan),
                         "excluded": excluded,
                         # compatibility with facebias.cli pairs / evaluate
                         "sex": a["gender_presentation"], "age_bin": a["age_band"],
                         "ethnicity": a["ancestry"], **a})
    meta = pd.DataFrame(rows)
    q = ofiq_scores(meta.path.tolist(), cfg.ofiq_command)
    meta["ofiq_unified"] = meta.path.map(q) if q else np.nan
    meta.to_csv(out / "metadata_all.csv", index=False)
    clean = meta[meta.excluded.isna()]
    n_ref2_low = int(((clean.condition == "ref2") & (clean.sim_to_ref < cfg.min_ref2_sim)).sum())
    clean = clean[~((clean.condition == "ref2") & (clean.sim_to_ref < cfg.min_ref2_sim))].copy()
    clean = add_balance(clean, seed=cfg.seed)
    clean.to_csv(out / "metadata.csv", index=False)
    print(f"ref2 below {cfg.min_ref2_sim}: {n_ref2_low} removed")
    rej = pd.DataFrame([{"identity_id": r["identity_id"], **{k: r["attrs"][k] for k in
                         ("gender_presentation", "age_band", "ancestry", "headwear")},
                         "stage": x["stage"].split(":")[0], "reason": x["reason"]}
                        for r in recs for x in r["rejections"]])
    rej.to_csv(out / "rejections.csv", index=False)
    write_datasheet(cfg, recs, clean, rej, dup, leaked, out / "DATASHEET.md")
    print(f"{clean.identity_id.nunique()} identities, {len(clean)} images in metadata.csv "
          f"(dropped: {len(dup)} near-duplicates, {len(leaked)} leakage)")
    return clean


BALANCE_AXES = ("age_band", "gender_presentation", "ancestry")


def add_balance(clean: pd.DataFrame, seed=0, iters=50):
    """Raking (iterative proportional fitting) of identity weights so that the
    weighted marginals of age band, gender presentation and ancestry match the
    design proportions of the sampling schema. Adds
      w_balance            identity weight (mean 1) - use in calibration / bias stats
      in_balanced_subset   identities drawn with probability proportional to
                           w_balance (max 1): an unweighted, balanced subset."""
    from .attributes import DEFAULT_SCHEMA
    ids = clean.drop_duplicates("identity_id")[["identity_id", *BALANCE_AXES]].reset_index(drop=True)
    w = np.ones(len(ids))
    tgt = {}
    for ax in BALANCE_AXES:
        d = DEFAULT_SCHEMA["balanced"][ax]
        levels = [l for l in d if (ids[ax] == l).any()]
        tot = sum(d[l] for l in levels)
        tgt[ax] = {l: d[l] / tot for l in levels}
    for _ in range(iters):
        for ax in BALANCE_AXES:
            cur = pd.Series(w).groupby(ids[ax]).sum() / w.sum()
            w *= ids[ax].map(lambda l: tgt[ax][l] / cur[l]).values
    w /= w.mean()
    p = w / w.max()
    keep = np.random.default_rng(seed).random(len(ids)) < p
    ids["w_balance"], ids["in_balanced_subset"] = w, keep
    return clean.merge(ids[["identity_id", "w_balance", "in_balanced_subset"]], on="identity_id", how="left")


def write_datasheet(cfg, recs, clean, rej, dup, leaked, path):
    refs = clean[clean.condition == "reference"]
    gen = clean[clean.condition.isin(["reference", "ref2", "hq"])]
    L = [f"# Datasheet - synthetic face set `{Path(cfg.out_dir).name}`", "",
         "All faces are AI-generated images of non-existent persons. No real face image was used "
         "as generator input. Labels describe the *intended* appearance of a fictitious person.", "",
         "## Composition",
         f"- identities: {refs.identity_id.nunique()} (requested {cfg.n_identities})",
         f"- generated faces (reference + edits): {len(gen)}; total image files incl. degradations: {len(clean)}",
         (f"- generators: MOCK (pipeline test, no real faces)" if cfg.mock else
          f"- generators: text-to-image `{cfg.t2i_backend}`, editing `{cfg.edit_backend}`; seed {cfg.seed}"),
         f"- removed near-duplicate identities: {len(dup)} (threshold {cfg.dedup_threshold})",
         f"- removed for possible memorisation of real persons: {len(leaked)} "
         f"(gallery: {cfg.leakage_gallery_npy or 'NOT CHECKED'}, threshold {cfg.leakage_threshold})", ""]
    for col in ("gender_presentation", "age_band", "ancestry", "headwear", "monk_skin_tone", "glasses", "facial_hair"):
        t = refs[col].value_counts().sort_index()
        L += [f"### {col}", "", "| level | identities |", "|---|---|"] + [f"| {k} | {v} |" for k, v in t.items()] + [""]
    if "w_balance" in refs.columns:
        L += ["## Balance (raking weights and balanced subset)", "",
              f"- identities in balanced subset: {int(refs.in_balanced_subset.sum())}",
              f"- weight range w_balance: {refs.w_balance.min():.2f} - {refs.w_balance.max():.2f}", ""]
        for col in BALANCE_AXES:
            t = pd.DataFrame({"all": refs[col].value_counts(),
                              "weighted": refs.groupby(col).w_balance.sum().round(0),
                              "balanced subset": refs[refs.in_balanced_subset][col].value_counts()}).fillna(0).astype(int)
            L += [f"### {col}", "", "| level | all | weighted | balanced subset |", "|---|---|---|---|"] + \
                 [f"| {k} | {r['all']} | {r['weighted']} | {r['balanced subset']} |" for k, r in t.iterrows()] + [""]
    L += ["## Images per condition", "", "| condition | images |", "|---|---|"] + \
         [f"| {k} | {v} |" for k, v in clean.condition.value_counts().items()] + [""]
    if len(rej):
        n_att = pd.DataFrame([{"ancestry": r["attrs"]["ancestry"], "n": 1} for r in recs]).groupby("ancestry").n.sum()
        rr = rej.groupby(["ancestry", "stage"]).size().unstack(fill_value=0)
        rr = rr.div(n_att, axis=0).round(3)
        L += ["## Rejections per identity, by ancestry (check for unequal filtering!)", "",
              rr.to_markdown() if hasattr(rr, "to_markdown") else rr.to_string(), ""]
    L += ["## Intended use / limitations",
          "- Calibration and validation of score-based LR systems and demographic bias testing of face "
          "comparison systems (e.g. System C).",
          "- Generator biases (realism, quality, attractiveness) may differ by group: compare quality "
          "(OFIQ) and score distributions with real benchmark data before relying on group results.",
          "- Identity consistency was filtered with an independent recogniser "
          f"(`{cfg.face_model}`, sim >= {cfg.min_identity_sim}); this truncates the hardest mated pairs.",
          "- Do not use for training systems intended to identify real persons from these labels; "
          "the labels of race/religion are prompts, not facts about any person.", ""]
    path.write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "finalize", "plan"])
    p.add_argument("--config", default="config_synth.yaml")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="stop after N chunks (exit code 3 if work remains); restarting per chunk avoids GPU-memory slowdown")
    a = p.parse_args(argv)
    if not Path(a.config).exists():
        raise SystemExit(f"config {a.config} not found")  # never fall back to defaults silently
    cfg = SynthConfig.from_yaml(a.config)
    if a.mock:
        cfg.mock, cfg.use_vlm = True, False
    if a.cmd == "plan":   # show the sampled design without generating anything
        ids = sample_identities(cfg.n_identities, seed=cfg.seed % 100, schema=_schema(cfg))
        df = pd.DataFrame([i.attrs for i in ids])
        for c in ("gender_presentation", "age_band", "ancestry", "headwear"):
            print(df[c].value_counts().to_string(), "\n")
        print(pd.crosstab(df.ancestry, df.headwear).to_string())
        from .prompts import reference_prompt as rp
        print("\nexample prompt:\n", rp(ids[0].attrs))
    elif a.cmd == "run":
        if run(cfg, a.shard, a.n_shards, a.max_chunks):
            sys.exit(3)
    else:
        finalize(cfg)


if __name__ == "__main__":
    main()
