"""
Attribute schema and balanced sampling of synthetic identities.

Design goals
------------
* Balanced on the attributes we want to test bias on (gender presentation,
  age, ancestry/region, skin tone, religious/cultural headwear) - quota
  sampling, so every level gets (nearly) the same number of identities.
* Lots of *identity-level* morphological variation (face shape, nose, eyes,
  lips, brows, build, skin features ...). Generators otherwise collapse to a
  few 'average', attractive faces -> doppelgangers between synthetic
  identities, which corrupts the non-mated pairs used for LR calibration.
* Only plausibility constraints that are needed for realism
  (e.g. hijab -> woman presentation, beard -> man presentation). Religion is
  deliberately NOT tied to ancestry, so effects of headwear and ancestry can
  be separated in the bias analysis.

All labels describe a fictitious person and the *intended* appearance; they
are verified afterwards (VLM check / human annotation sample).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

# ancestry/region -> plausible Monk skin tone range (1 = lightest, 10 = darkest)
DEFAULT_ANCESTRY = {
    "West African": (6, 10), "East African": (6, 10), "North African": (3, 7),
    "Middle Eastern": (3, 7), "Turkish": (2, 6), "South Asian": (4, 9),
    "East Asian": (2, 5), "Southeast Asian": (3, 7), "Indonesian": (4, 8),
    "Northern European": (1, 3), "Southern European": (2, 5), "Eastern European": (1, 3),
    "Latin American (mestizo)": (3, 7), "Afro-Caribbean / Afro-Surinamese": (6, 10),
    "Hindustani-Surinamese": (5, 8), "Indigenous American": (4, 7),
    "Pacific Islander": (5, 8), "Mixed ancestry": (2, 9),
}

DEFAULT_SCHEMA = {
    # ---------- balanced (quota) attributes: bias analysis axes ----------
    "balanced": {
        "gender_presentation": {"woman": 0.45, "man": 0.45, "androgynous": 0.10},
        "age_band": {"18-29": 1, "30-44": 1, "45-59": 1, "60-74": 1, "75+": 0.6},
        "ancestry": {k: 1 for k in DEFAULT_ANCESTRY},
        "headwear": {"none": 0.60, "hijab": 0.10, "dastar (Sikh turban)": 0.06,
                     "kippah": 0.06, "kufi / taqiyah": 0.06, "tichel / headscarf": 0.06,
                     "secular cap": 0.06},
    },
    "age_range": {"18-29": (18, 29), "30-44": (30, 44), "45-59": (45, 59),
                  "60-74": (60, 74), "75+": (75, 88)},
    # ---------- free (random) identity attributes: diversity ----------
    "free": {
        "face_shape": ["oval", "round", "square", "long", "heart-shaped", "diamond-shaped", "rectangular"],
        "nose": ["narrow", "broad", "straight", "aquiline", "small upturned", "large", "flat-bridged", "crooked"],
        "eyes": ["deep-set", "hooded", "wide-set", "close-set", "almond-shaped", "round", "monolid", "downturned"],
        "lips": ["thin", "full", "medium", "wide", "narrow", "asymmetric"],
        "eyebrows": ["thick", "thin", "arched", "straight", "bushy", "sparse", "unibrow-tendency"],
        "jaw": ["strong", "soft", "narrow", "wide", "receding chin", "prominent chin"],
        "build": ["very slim", "slim", "average", "stocky", "overweight", "obese"],
        "ears": ["small", "large", "protruding", "flat against head"],
        "skin_features": ["none", "none", "none", "freckles", "acne scars", "moles", "vitiligo patches",
                          "rosacea", "a small scar on the cheek", "a birthmark", "deep wrinkles", "eye bags"],
        "hair_style": ["short", "very short buzz cut", "medium length", "long", "curly", "coily afro",
                       "braids", "locs", "receding hairline", "bald", "ponytail", "bun", "wavy"],
        "hair_color": ["black", "dark brown", "brown", "light brown", "blond", "red", "dyed", "grey"],
        "glasses": ["none", "none", "none", "clear prescription glasses"],
        "makeup": ["none", "none", "light", "heavy"],
        "facial_hair": ["none", "stubble", "moustache", "goatee", "short beard", "full beard", "long beard"],
        "jewellery": ["none", "none", "earrings", "nose piercing", "necklace", "eyebrow piercing"],
        "expression_base": ["neutral"],
    },
}


@dataclass
class Identity:
    identity_id: str
    attrs: dict

    def __getitem__(self, k):
        return self.attrs[k]


def _quota_sequence(levels: dict, n: int, rng) -> list:
    """Exact quotas proportional to weights, then shuffled (largest remainder)."""
    keys = list(levels)
    w = np.array([levels[k] for k in keys], float)
    raw = w / w.sum() * n
    cnt = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - cnt))[: n - cnt.sum()]:
        cnt[i] += 1
    seq = [k for k, c in zip(keys, cnt) for _ in range(c)]
    rng.shuffle(seq)
    return seq


def _plausible(a: dict) -> bool:
    g, hw = a["gender_presentation"], a["headwear"]
    if hw in ("hijab", "tichel / headscarf") and g == "man":
        return False
    if hw in ("kippah", "kufi / taqiyah", "dastar (Sikh turban)") and g == "woman":
        return False
    return True


def _fix_dependent(a: dict, rng):
    g, age = a["gender_presentation"], a["age"]
    if g == "woman":
        a["facial_hair"] = "none"
    elif g == "androgynous":
        a["facial_hair"] = rng.choice(["none", "none", "stubble"])
        a["makeup"] = rng.choice(["none", "light"])
    else:
        a["makeup"] = "none" if rng.random() < 0.95 else "light"
    if a["headwear"] == "dastar (Sikh turban)" and rng.random() < 0.7:
        a["facial_hair"] = rng.choice(["full beard", "long beard"])
    if a["headwear"] in ("hijab", "tichel / headscarf", "dastar (Sikh turban)"):
        a["hair_style"] = "covered"
    if age >= 60 and rng.random() < 0.6:
        a["hair_color"] = rng.choice(["grey", "white", "salt-and-pepper"])
    if age < 30 and a["hair_color"] == "grey":
        a["hair_color"] = "dark brown"
    if age < 35 and a["skin_features"] == "deep wrinkles":
        a["skin_features"] = "none"
    if a["hair_style"] == "bald" and g == "woman" and rng.random() < 0.8:
        a["hair_style"] = "short"
    return a


def sample_identities(n: int, seed: int = 0, schema: dict | None = None,
                      ancestry_skin: dict | None = None, max_tries: int = 50) -> list[Identity]:
    schema = copy.deepcopy(schema or DEFAULT_SCHEMA)
    skin = ancestry_skin or DEFAULT_ANCESTRY
    rng = np.random.default_rng(seed)
    bal = schema["balanced"]
    seqs = {k: _quota_sequence(v, n, rng) for k, v in bal.items()}

    # repair implausible combos by swapping headwear with another identity
    hw = seqs["headwear"]
    for i in range(n):
        for _ in range(max_tries):
            a = {k: seqs[k][i] for k in bal}
            if _plausible(a):
                break
            j = int(rng.integers(n))
            b = {k: seqs[k][j] for k in bal}
            b_sw = dict(b, headwear=hw[i]); a_sw = dict(a, headwear=hw[j])
            if _plausible(a_sw) and _plausible(b_sw):
                hw[i], hw[j] = hw[j], hw[i]
                break

    out = []
    for i in range(n):
        a = {k: seqs[k][i] for k in bal}
        if not _plausible(a):
            a["headwear"] = "none"
        lo, hi = schema["age_range"][a["age_band"]]
        a["age"] = int(rng.integers(lo, hi + 1))
        s_lo, s_hi = skin[a["ancestry"]]
        a["monk_skin_tone"] = int(rng.integers(s_lo, s_hi + 1))
        for k, levels in schema["free"].items():
            a[k] = str(rng.choice(levels))
        a = _fix_dependent(a, rng)
        out.append(Identity(f"syn{seed:02d}_{i:05d}", a))
    return out
