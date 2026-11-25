# facebias

Code for *Synthetic faces for measuring demographic differentials and calibrating likelihood ratios in forensic facial comparison* (Geradts, preprint 2026).

It generates demographically balanced synthetic identities (text-to-image plus instruction edits), simulates forensic capture conditions (CCTV stills, H.264 video, doorbell, body-worn, infrared), scores pairs with open face recognisers or with scores exported from a commercial system, and analyses demographic differentials as FMR/FNMR per group and as shifts in calibrated likelihood ratios (logistic regression with ELUB bounds, identity-disjoint cross-validation, Cllr).

## Layout

| Path | Content |
|---|---|
| `facebias/synth/` | identity attributes, prompts, generation back-ends, quality control, video degradation |
| `facebias/` | pair building, open recognisers (embeddings), score import, LR calibration, bias analysis, reports |
| `tools/` | analysis scripts used for the paper (per-condition bias, LR direction, cohort normalisation, real-face checks on FairFace/BFW/RFW, second generator) |
| `tests/` | unit tests (`python -m pytest tests`) |

## Commercial system

The commercial recogniser (System C in the paper) is licensed and is not called from this code. Its scores are imported from a CSV with columns `image_a,image_b,score` (`tools/import_external_scores.py`); `tools/make_external_export.py` writes the image and pair lists to score on the licensed workstation.

## Data

No face images are distributed here. Synthetic sets are regenerated from the configs (`config_1000.yaml`) with the generators named in the paper. The real-face sets (FairFace, BFW, RFW) must be obtained from their owners under their own licences; only group-level results are reported.

## Install

```
pip install -r requirements.txt            # analysis
pip install -r requirements-synth.txt      # generation (GPU)
```

## Licence

MIT, see `LICENSE`. Generator and recogniser model weights keep their own licences.
