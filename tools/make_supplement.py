"""
Build Supplementary Tables S1-S10 (one Excel workbook, one sheet per table, plus CSVs) from the result files.
Run from the facebias folder:  python3 tools/make_supplement.py [--synth <synth1000 dir>]
The commercial system is reported as 'System C'. Only group-level numbers; no images, no per-pair data.
"""
import argparse, io, re
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--synth", default="data/synth1000")
p.add_argument("--out", default="supplement")
a = p.parse_args()
R, E, P = Path("."), Path("results_extended"), Path("results_precise")
OUT = Path(a.out); OUT.mkdir(exist_ok=True)

NAMES = {"system_c_ed015": "System C (second eye-distance setting)", "system_c": "System C", "adaface": "AdaFace IR101 WebFace12M",
         "adaface_kprpe": "AdaFace KP-RPE", "adaface_ms1mv2": "AdaFace IR101 MS1MV2", "adaface_vit": "AdaFace ViT",
         "arcface": "ArcFace R100 (buffalo_l)", "arcface_ir101": "ArcFace IR101", "glint_r100": "Glint360K R100"}
TERMS = {"passfoto": "passport photo", "alle": "all", "gewoon": "ordinary", "globaal": "global",
         "per conditie": "per condition", "kwaliteit": "quality"}

def clean(d):
    d = d.copy()
    for c in d.columns:
        if d[c].dtype == object:
            s = d[c].astype(str)
            for k in sorted(NAMES, key=len, reverse=True):
                s = s.str.replace(rf"^{k}(?=$|\|)", NAMES[k], regex=True)
            for k, v in TERMS.items():
                s = s.str.replace(rf"\b{k}\b", v, regex=True)
            s = s.str.replace("system_c", "System C", case=False)
            d[c] = s.where(d[c].notna(), None)
        elif d[c].dtype.kind == "f":
            d[c] = d[c].round(4)
    d.columns = [re.sub(r"(?i)system_c", "System C", str(c)).replace("_gewoon", "_ordinary").replace("_doppel", "_lookalike")
                 for c in d.columns]
    return d

def md_tables(path):
    """All pipe tables in a markdown file, with the preceding heading as 'section'."""
    out, sec, buf = [], "", []
    for line in list(Path(path).read_text(encoding="utf-8").splitlines()) + [""]:
        if line.startswith("|"):
            buf.append(line); continue
        if buf:
            rows = [r for r in buf if not re.match(r"^\|[\s:|-]+\|$", r)]
            t = pd.read_csv(io.StringIO("\n".join(r.strip().strip("|") for r in rows)), sep="|", skipinitialspace=True)
            t.columns = [c.strip() for c in t.columns]
            t.insert(0, "section", sec); out.append(t); buf = []
        if line.startswith("#"):
            sec = line.lstrip("#").strip()
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def cat(files, tag=None):
    fs = [f for f in files if Path(f).exists()]
    ds = []
    for f in fs:
        d = pd.read_csv(f)
        if tag:
            d.insert(0, tag, tag_of(f))
        ds.append(d)
    return pd.concat(ds, ignore_index=True) if ds else pd.DataFrame()

def tag_of(f):
    s = Path(f).stem
    for k in ("k50", "k200", "c1200", "d080"):
        if k in s:
            return {"k50": "top-50", "k200": "top-200", "c1200": "cohort 1,200", "d080": "duplicate cut-off 0.80"}[k]
    return "default (top-100, cohort 600, cut-off 0.75)"

T = {}
# S1 attribute distribution of the synthetic identities per ancestry
m = pd.read_csv(Path(a.synth) / "metadata.csv").drop_duplicates("identity_id")
rows = []
for col in ["gender_presentation", "age_band", "monk_skin_tone", "headwear", "facial_hair", "glasses", "hair_style", "build"]:
    ct = pd.crosstab(m.ancestry, m[col], normalize="index").mul(100).round(1)
    ct = ct.stack().rename("percent").reset_index().rename(columns={col: "value"}); ct.insert(1, "attribute", col)
    rows.append(ct)
s1 = pd.concat(rows).pivot_table(index=["attribute", "value"], columns="ancestry", values="percent").reset_index()
n = m.groupby("ancestry").size().rename("n identities").reset_index()
T["S1"] = ("Synthetic identities: number per ancestry group and attribute distribution (% within group). "
           "Prompt templates and the 13 situations are in facebias/synth/prompts.py of the code repository.",
           [n, s1])
# S2 face finding
T["S2"] = ("Failure to enrol (FTE, %) per condition and group: SCRFD (two passes) and after a second detector (YuNet).",
           [md_tables(R / "results_fte_second_detector.md")])
# S3 / S4 per group, class and recogniser
models = ["system_c", "system_c_ed015", "adaface", "adaface_kprpe", "adaface_ms1mv2", "adaface_vit", "arcface",
          "arcface_ir101", "glint_r100"]
T["S3"] = ("FMR and FNMR per group, condition class and recogniser at the global threshold for FMR 0.1%, with 95% bootstrap "
           "CIs over identities; FMR_ratio = group FMR / mean group FMR.", [cat([P / f"{x}_fmr.csv" for x in models])])
T["S4"] = ("Mean log10 LR of the global calibration minus a per-group calibration (bias_mated, bias_nonmated), relative to "
           "the group mean (_rel), with 95% CIs; rates of misleading evidence (RMED/RMEP) for global and per-group calibration.",
           [cat([P / f"{x}_lr.csv" for x in models])])
# S5 confounders and quality-aware calibration
T["S5"] = ("Mixed-model effect of ancestry on mated scores, raw and adjusted for image quality and skin tone (a); "
           "quality-aware calibration models M0-M3 per group (b).",
           [pd.read_csv(E / "confounders_regression.csv"), cat(sorted(E.glob("quality_lr_*.csv")))])
# S6 twins and look-alikes
T["S6"] = ("Generated twins (a) and look-alikes (b): false-match rate and misleading LRs per recogniser, class and ancestry.",
           [pd.read_csv(E / "twins.csv"), pd.read_csv(E / "doppelgangers.csv")])
# S7 sensitivity
T["S7"] = ("Sensitivity of cohort normalisation to top-k and cohort size on synthetic faces (a: accuracy and FMR gap; "
           "b: LR shift per group), and to the duplicate cut-off on FairFace (c).",
           [cat([E / "mitigation_norm_accuracy.csv", E / "mitigation_norm_k50_accuracy.csv",
                 E / "mitigation_norm_k200_accuracy.csv", E / "mitigation_norm_c1200_accuracy.csv"], tag="setting"),
            cat([E / "mitigation_norm_lr.csv", E / "mitigation_norm_k50_lr.csv", E / "mitigation_norm_k200_lr.csv",
                 E / "mitigation_norm_c1200_lr.csv"], tag="setting"),
            cat([E / "fairface_real2_summary.csv", E / "fairface_real2_d080_summary.csv",
                 E / "fairface_real2_k50_summary.csv", E / "fairface_real2_k200_summary.csv"], tag="setting")])
# S8 FairFace, BFW, RFW
T["S8"] = ("Real faces: FairFace non-mated FMR per group, all conditions and normalisations (a); BFW (b) and RFW (c) "
           "per group with mated pairs (FMR ratio, FNMR, Cllr, LR shift).",
           [pd.read_csv(E / "fairface_real2_all_table.csv"),
            cat([E / "real_mated_bfw.csv"]), cat([E / "real_mated_rfw.csv"])])
# S9 second generator
T["S9"] = ("Second generator (FLUX.2 klein) against the main set: FMR gap, LR spread, FNMR and highest group per recogniser "
           "and class.", [pd.read_csv(E / "gen2_comparison_summary.csv"), pd.read_csv(E / "gen2_comparison.csv")])
# S10 near duplicates
T["S10"] = ("Near-duplicate and homogeneity check of generated identities per ancestry and generator (a, b) and per-group "
            "FMR of the generators (c).",
            [pd.read_csv(E / "generator_near_duplicates.csv"), pd.read_csv(E / "generator_homogeneity.csv"),
             pd.read_csv(E / "generator_fmr.csv")])

xl = OUT / "Supplementary_Tables_S1-S10.xlsx"
with pd.ExcelWriter(xl, engine="openpyxl") as w:
    pd.DataFrame({"table": list(T), "caption": [v[0] for v in T.values()]}).to_excel(w, sheet_name="Contents", index=False)
    for k, (cap, parts) in T.items():
        r = 2
        pd.DataFrame({k: [cap]}).to_excel(w, sheet_name=k, index=False)
        for i, d in enumerate(parts):
            d = clean(d)
            if len(parts) > 1:
                pd.DataFrame({f"({'abc'[i]})": []}).to_excel(w, sheet_name=k, index=False, startrow=r)
                r += 1
            d.to_excel(w, sheet_name=k, index=False, startrow=r)
            d.to_csv(OUT / f"{k}{'abc'[i] if len(parts) > 1 else ''}.csv", index=False)
            r += len(d) + 3
for k, (cap, parts) in T.items():
    print(k, [len(d) for d in parts])
print("->", xl)
