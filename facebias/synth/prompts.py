"""
Prompt construction.

Reference image  : text-to-image, frontal 'custody / ID photo' style
                   (the good-quality reference in a forensic comparison).
Probe images     : *image editing* of the reference with an instruction that
                   changes pose / expression / light / camera / clothing but
                   keeps the person. Using an editing model (Qwen-Image-Edit,
                   FLUX.2 multi-reference) instead of face-embedding
                   conditioning means identity is NOT defined by ArcFace or any
                   other recogniser -> no circularity with the systems under test.
Forensic degradations (CCTV resolution, compression, noise, low light) are
applied afterwards in code, so they are exactly controlled.
"""
from __future__ import annotations

SKIN_WORDS = {1: "very fair", 2: "fair", 3: "light", 4: "light-medium", 5: "medium",
              6: "medium-tan", 7: "tan brown", 8: "brown", 9: "dark brown", 10: "very dark brown"}

REALISM = ("candid unretouched photograph, natural skin texture with visible pores and small "
           "imperfections, ordinary-looking real person, not a model, no beauty filter, "
           "realistic proportions, sharp focus")

NEGATIVE = ("cartoon, anime, illustration, painting, 3d render, cgi, doll, plastic skin, airbrushed, "
            "beauty filter, glamour, heavy retouching, deformed, extra fingers, two people, text, "
            "watermark, logo, sunglasses, pointed ears, elf ears, fantasy creature")


# Run 5: with Qwen-Image-2512, young women described with masculine-coded traits
# (thick/bushy brows + strong jaw/chin + short/bald hair) were rendered as men in
# 12 of 14 cases, whatever the seed. For women these traits are worded more mildly.
def _brows(a):
    if a["gender_presentation"] == "woman" and a["eyebrows"] in ("thick", "bushy"):
        return "full natural"
    return a["eyebrows"]


def _jaw(a):
    if a["gender_presentation"] == "woman" and a["jaw"] in ("strong", "prominent chin"):
        return "well-defined"
    return a["jaw"]


# Qwen-Image renders 'large'/'protruding' ears grotesquely big and often pointed
# (elf-like), even with a negative prompt (run 5/6 review). Ear shape is therefore
# no longer prompted; the sampled value stays in the metadata but is not depicted.
def _ears(a):
    return "normal rounded human ears"


def describe(a: dict) -> str:
    g = {"woman": "woman", "man": "man", "androgynous": "androgynous-looking person"}[a["gender_presentation"]]
    parts = [f"a {a['age']}-year-old {a['ancestry']} {g} with {SKIN_WORDS[a['monk_skin_tone']]} skin",
             f"{a['face_shape']} face", f"{a['nose']} nose", f"{a['eyes']} eyes", f"{a['lips']} lips",
             f"{_brows(a)} eyebrows", f"{_jaw(a)} jaw", _ears(a), f"{a['build']} build"]
    if a["hair_style"] != "covered":
        parts.append("bald head" if a["hair_style"] == "bald" else f"{a['hair_color']} {a['hair_style']} hair")
    if a["facial_hair"] != "none":
        parts.append(a["facial_hair"])
    if a["skin_features"] != "none":
        parts.append(a["skin_features"])
    if a["glasses"] != "none":
        parts.append(a["glasses"])
    if a["makeup"] != "none":
        parts.append(f"{a['makeup']} makeup")
    if a["jewellery"] != "none":
        parts.append(a["jewellery"])
    return ", ".join(parts)


HEADWEAR_DESC = {
    "hijab": "a hijab headscarf that covers all of the hair, the ears and the neck",
    "dastar (Sikh turban)": "a neatly wrapped Sikh turban (dastar) that covers all of the hair",
    "kippah": "a small Jewish kippah (skullcap) on the crown of the head",
    "kufi / taqiyah": "a white kufi prayer cap (taqiyah) on the head",
    "tichel / headscarf": ("a tichel: a soft patterned fabric headscarf wrapped around the head and tied in a knot at the "
                           "back of the neck, covering all of the hair, as worn by married Orthodox Jewish women"),
    "secular cap": "a plain baseball cap",
}


def age_cues(age: int) -> str:
    # The generator makes people look 15-25 years too young (run 1: VLM age median
    # 14-24 y below the intended age from 30+), so every band from 35 gets explicit cues.
    if age >= 75:
        return " Clearly elderly, looks over 75: deep wrinkles, sagging skin, age spots, thin white or grey hair."
    if age >= 60:
        return " Looks over 60: wrinkles around eyes and mouth, age spots, grey hair, slightly sagging skin."
    if age >= 45:
        return (" Middle-aged, looks about 50: visible crow's feet and forehead wrinkles, nasolabial folds, "
                "some grey hair, slightly sagging jawline.")
    if age >= 35:
        return " Looks in their late thirties: fine lines around the eyes and on the forehead."
    return ""


def reference_prompt(a: dict) -> str:
    hw = (f" The person is wearing {HEADWEAR_DESC[a['headwear']]}; the face itself is fully visible."
          if a["headwear"] != "none" else "")
    sex = " She is clearly a woman." if a["gender_presentation"] == "woman" else ""
    return (f"Frontal passport-style identity photograph of {describe(a)}.{sex}{hw}{age_cues(a['age'])} "
            f"Head and shoulders, wearing a plain dark crew-neck t-shirt, looking straight into the camera, "
            f"neutral expression, mouth closed, plain light grey background, even diffuse lighting. {REALISM}.")


KEEP = ("Keep exactly the same person: identical facial features, face shape, skin tone, "
        "age, scars and marks{hw}. Photorealistic. ")

# name -> (instruction, short description). Edit list per identity is sampled from these.
PROBE_EDITS = {
    "ref2": "A second passport-style photo of this person taken on a different day: different clothing, "
            "slightly different hair, same neutral frontal pose, plain background.",
    "cctv_ceiling": "Show this person walking in a shop, photographed from a ceiling-mounted security "
                    "camera high above, looking down at a steep angle, fluorescent lighting, wide shot.",
    "cctv_street": "Show this person on a city street at night under orange street lights, "
                   "seen from a security camera at about 3 metres height, slight motion, three-quarter view.",
    "profile45": "Turn the head about 45 degrees to the left, natural indoor light, neutral expression.",
    "profile_r30": "Turn the head about 30 degrees to the right and slightly down, daylight from a window.",
    "smile_selfie": "A smartphone selfie of this person smiling broadly, outdoors, natural daylight, "
                    "arm-length distance, slight wide-angle distortion.",
    "talking": "This person talking with mouth open mid-sentence, candid photo in a cafe, warm light.",
    "harsh_side": "Harsh side lighting from one side leaving half of the face in shadow, dark background.",
    "backlit": "Standing in front of a bright window, backlit, face underexposed.",
    "hood_up": "Wearing a hooded sweatshirt with the hood up, face visible, outdoors, overcast.",
    "cctv_far": "Show this person standing in a large train station hall, seen from a security camera "
                "mounted high on a pillar about 10 metres away, wide shot, the person is small in the image, "
                "fluorescent light.",
    "cctv_entrance": "Show this person walking in through glass entrance doors, seen from a camera above the door "
                     "at about 3 metres height looking down at 30 degrees, daylight from behind.",
    "doorbell": "Seen through a doorbell camera at the front door, close range, fisheye distortion, daylight.",
}


def edit_prompt(edit_name: str, a: dict) -> str:
    hw = f", and the same {a['headwear']}" if a["headwear"] not in ("none", "secular cap") else ""
    # secular cap is removed in a hood edit; religious headwear is always kept
    return KEEP.format(hw=hw) + PROBE_EDITS[edit_name]
