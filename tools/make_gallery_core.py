"""Build an offline overview page (gallery.html) for a synthetic data set.

    python tools/make_gallery.py data/synth1000 [--max-seconds 150]

Creates small JPEG thumbnails in <dir>/gallery_thumbs (incremental: existing
thumbnails are reused) and <dir>/gallery.html with filters on ancestry,
headwear, gender presentation, age band and status. Click a thumbnail to open
the full-resolution PNG. Everything stays local on this computer.
"""
import argparse, glob, html, json, os, sys, time
from pathlib import Path
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--max-seconds", type=float, default=1e9)
ap.add_argument("--size", type=int, default=160)
a = ap.parse_args()
root = Path(a.dir); thumbs = root / "gallery_thumbs"; thumbs.mkdir(exist_ok=True)
t0 = time.time(); made = 0; pending = 0
rows = []
order = {"reference": 0, "ref2": 1}
for jf in sorted(root.glob("identities/*/identity.json")):
    try:
        r = json.loads(jf.read_text())
    except Exception:
        continue
    imgs = []
    for im in r.get("images", []):
        if im["condition"] not in ("reference", "ref2", "hq"):
            continue  # degraded copies are not shown separately
        src = Path(str(im["path"]).replace("\\", "/"))
        src = root / "identities" / r["identity_id"] / src.name
        th = thumbs / (src.stem + ".jpg")
        if not th.exists():
            if time.time() - t0 < a.max_seconds:
                try:
                    with Image.open(src) as I:
                        I = I.convert("RGB"); I.thumbnail((a.size, a.size)); I.save(th, quality=82)
                    made += 1
                except Exception:
                    continue
            else:
                pending += 1
                continue
        imgs.append({"t": f"gallery_thumbs/{th.name}", "f": f"identities/{r['identity_id']}/{src.name}",
                     "e": im["edit"], "s": round(float(im.get("sim_to_ref", 1)), 2)})
    imgs.sort(key=lambda x: order.get(x["e"], 2))
    at = r["attrs"]
    rows.append({"id": r["identity_id"], "st": r["status"], "g": at["gender_presentation"], "age": at["age"],
                 "ab": at["age_band"], "an": at["ancestry"], "hw": at["headwear"], "sk": at["monk_skin_tone"],
                 "gl": at["glasses"], "rej": len(r.get("rejections", [])), "im": imgs})

page = """<!doctype html><html lang="nl"><meta charset="utf-8"><title>Synthetische gezichten</title>
<style>
:root{--bg:#fafafa;--fg:#1c1c1c;--mut:#6b6b6b;--line:#e3e3e3;--card:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--mut:#9a9a9a;--line:#2c2c2c;--card:#1d1d1d}}
body{margin:0;font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:2}
h1{font-size:17px;margin:0 0 8px}
.f{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
select,input{font:inherit;padding:4px 6px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
#n{color:var(--mut);margin-left:auto}
main{padding:8px 16px}
.r{display:flex;gap:12px;align-items:flex-start;padding:8px 0;border-bottom:1px solid var(--line)}
.m{width:210px;flex:none;color:var(--mut);font-size:12px;line-height:1.45}.m b{color:var(--fg);font-size:13px}
.ims{display:flex;gap:6px;flex-wrap:wrap}
figure{margin:0;text-align:center;font-size:11px;color:var(--mut)}
figure img{width:120px;height:120px;object-fit:cover;border-radius:6px;display:block;background:var(--line)}
.low{color:#c0392b}.bad{opacity:.55}
</style>
<header><h1>Synthetische gezichten &mdash; alle personen zijn niet-bestaand (AI-gegenereerd)</h1>
<div class="f">
<select id="st"><option value="">alle statussen</option><option value="done" selected>klaar</option><option value="reference_failed">referentie afgekeurd</option><option value="reference_ok">wacht op bewerkingen</option></select>
<select id="g"><option value="">alle genderpresentaties</option></select>
<select id="ab"><option value="">alle leeftijden</option></select>
<select id="an"><option value="">alle herkomsten</option></select>
<select id="hw"><option value="">alle hoofdbedekking</option></select>
<label>min. gelijkenis <input id="sim" type="number" step="0.05" min="0" max="1" value="0" style="width:64px"></label>
<span id="n"></span></div></header><main id="list"></main>
<script>
const D=__DATA__;
const sel=["g","ab","an","hw"];
sel.forEach(k=>{const s=document.getElementById(k);[...new Set(D.map(r=>r[k]))].sort().forEach(v=>{const o=document.createElement("option");o.value=o.textContent=v;s.appendChild(o)})});
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function draw(){const f={st:st.value};sel.forEach(k=>f[k]=document.getElementById(k).value);const ms=parseFloat(sim.value)||0;
let rs=D.filter(r=>(!f.st||r.st==f.st)&&sel.every(k=>!f[k]||r[k]==f[k]));
if(ms>0)rs=rs.filter(r=>r.im.some(i=>i.e!="reference"&&i.s<ms));
n.textContent=rs.length+" personen";
list.innerHTML=rs.slice(0,400).map(r=>`<div class="r ${r.st=="done"?"":"bad"}"><div class="m"><b>${esc(r.id)}</b><br>${esc(r.g)}, ${r.age} jr<br>${esc(r.an)}<br>hoofd: ${esc(r.hw)}<br>huid (Monk): ${r.sk} &middot; bril: ${esc(r.gl)}<br>status: ${esc(r.st)} &middot; afgekeurd: ${r.rej}</div><div class="ims">${r.im.map(i=>`<figure><a href="${esc(i.f)}" target="_blank"><img loading="lazy" src="${esc(i.t)}" alt=""></a>${esc(i.e)}${i.e!="reference"?` <span class="${i.s<0.6?"low":""}">${i.s}</span>`:""}</figure>`).join("")}</div></div>`).join("")+(rs.length>400?`<p>Eerste 400 van ${rs.length} getoond &mdash; verfijn het filter.</p>`:"")}
["st","sim",...sel].forEach(k=>document.getElementById(k).addEventListener("input",draw));draw();
</script></html>"""
(root / "gallery.html").write_text(page.replace("__DATA__", json.dumps(rows)), encoding="utf-8")
print(f"{len(rows)} personen, {made} nieuwe thumbnails, {pending} nog te maken -> {root/'gallery.html'}")
