import time
import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sklearn

import os
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "outputs", "rf_final_10trees.joblib")
model = joblib.load(MODEL_PATH)
model.n_jobs = 1          # single-threaded: 6x faster for single messages
_warm = np.zeros((1, 7), dtype=np.float32)
for _ in range(50):
    model.predict(_warm)
# 3GPP TS 22.186 latency budgets
TIERS = [("Basic safety messaging", 100.0),
         ("Platooning", 10.0),
         ("Advanced eV2X", 3.0)]

app = FastAPI(title="V2X Misbehavior Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    sendTime: float
    posx: float
    posy: float
    spdx: float
    spdy: float
    aclx: float
    acly: float
    hedx: float
    hedy: float


class DetectRequest(BaseModel):
    current: Message
    previous: Message


def build_features(cur: Message, prev: Message) -> np.ndarray:
    dt = cur.sendTime - prev.sendTime
    dx = cur.posx - prev.posx
    dy = cur.posy - prev.posy

    dist = float(np.sqrt(dx * dx + dy * dy))
    claimed_spd = float(np.sqrt(cur.spdx ** 2 + cur.spdy ** 2))
    implied_spd = dist / dt if dt > 0 else 0.0
    spd_mismatch = abs(claimed_spd - implied_spd)
    acl_mag = float(np.sqrt(cur.aclx ** 2 + cur.acly ** 2))

    norm = dist if dist > 0 else 1.0
    head_mismatch = 1.0 - ((dx / norm) * cur.hedx + (dy / norm) * cur.hedy)

    return np.array([[dt, dist, claimed_spd, implied_spd,
                      spd_mismatch, acl_mag, head_mismatch]], dtype=np.float32)


@app.get("/")
def root():
    return {"status": "running",
            "model": "RandomForest, 10 trees",
            "features": 7}


@app.post("/detect")
def detect(req: DetectRequest):
    features = build_features(req.current, req.previous)

    # time ONLY the inference, not feature building or JSON parsing
    with sklearn.config_context(assume_finite=True):
        t0 = time.perf_counter()
        pred = model.predict(features)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

    proba = float(model.predict_proba(features)[0][1])

    compliance = {name: (latency_ms <= budget) for name, budget in TIERS}

    return {
        "verdict": "ATTACK" if int(pred) == 1 else "NORMAL",
        "confidence": round(proba if int(pred) == 1 else 1 - proba, 4),
        "inference_latency_ms": round(latency_ms, 3),
        "compliance_3gpp": compliance,
        "features": {
            "dt": round(float(features[0][0]), 3),
            "dist": round(float(features[0][1]), 3),
            "claimed_spd": round(float(features[0][2]), 3),
            "implied_spd": round(float(features[0][3]), 3),
            "spd_mismatch": round(float(features[0][4]), 3),
            "acl_mag": round(float(features[0][5]), 3),
            "head_mismatch": round(float(features[0][6]), 3),
        },
    }