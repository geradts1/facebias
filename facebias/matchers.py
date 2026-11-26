"""
Comparison back-ends.

the commercial system (System C) is a licensed SDK, so it is not called directly here.
Two ways to plug it in:

1. ExternalSDKMatcher  - you provide a small wrapper executable written with
   the vendor SDK (C++/Java/.NET) that follows this contract:

       <wrapper> <pairs.csv> <scores.csv>

   pairs.csv  : header 'image_a,image_b'
   scores.csv : header 'image_a,image_b,score'   (score = System C similarity,
                empty if enrolment failed, i.e. no face found / FTE)

   Inside the wrapper: enrol each image once (cache the FIR template),
   then compare the FIR of image_a with that of image_b.
   Record the SDK version and configuration file in the metadata - scores,
   and therefore calibration, are version-specific.

2. ScoreFileMatcher - you already ran the comparisons (e.g. a batch run in a
   vendor tool) and exported a CSV with the same 'image_a,image_b,score' layout.

ArcFaceMatcher (open source, insightface) is provided as a reference system,
to see whether a bias is System C-specific or present in the data itself.
"""
from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


class ExternalSDKMatcher:
    name = "system_c"

    def __init__(self, command: str):
        self.command = command  # e.g. "C:/tools/fvcompare.exe" or "java -jar fv.jar"

    def score_pairs(self, pairs: pd.DataFrame) -> pd.Series:
        with tempfile.TemporaryDirectory() as tmp:
            pin, pout = Path(tmp) / "pairs.csv", Path(tmp) / "scores.csv"
            pairs[["image_a", "image_b"]].drop_duplicates().to_csv(pin, index=False)
            subprocess.run(shlex.split(self.command) + [str(pin), str(pout)], check=True)
            return _merge_scores(pairs, pd.read_csv(pout))


class ScoreFileMatcher:
    name = "scorefile"

    def __init__(self, path: str):
        self.path = path

    def score_pairs(self, pairs: pd.DataFrame) -> pd.Series:
        return _merge_scores(pairs, pd.read_csv(self.path))


def _merge_scores(pairs, scores):
    key = ["image_a", "image_b"]
    s = scores.copy()
    for k in key:  # match on file name so absolute/relative paths don't matter
        s[k] = s[k].map(lambda p: Path(str(p)).name)
    left = pairs[key].apply(lambda c: c.map(lambda p: Path(str(p)).name))
    merged = left.merge(s, on=key, how="left")
    return pd.Series(merged["score"].values, index=pairs.index, dtype=float)


class ArcFaceMatcher:
    name = "arcface"

    def __init__(self, model="buffalo_l", ctx_id=0):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name=model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._cache = {}

    def embed(self, path):
        if path not in self._cache:
            import cv2
            faces = self.app.get(cv2.imread(str(path)))
            if not faces:
                self._cache[path] = None
            else:
                f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                self._cache[path] = f.normed_embedding
        return self._cache[path]

    def score_pairs(self, pairs: pd.DataFrame) -> pd.Series:
        out = []
        for a, b in zip(pairs.image_a, pairs.image_b):
            ea, eb = self.embed(a), self.embed(b)
            out.append(np.nan if ea is None or eb is None else float(np.dot(ea, eb)))
        return pd.Series(out, index=pairs.index)


def get_matcher(kind: str, **kw):
    return {"system_c": ExternalSDKMatcher, "scorefile": ScoreFileMatcher,
            "arcface": ArcFaceMatcher}[kind](**kw)
