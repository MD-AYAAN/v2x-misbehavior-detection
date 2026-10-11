# V2X Misbehavior Detection System

Latency-budget-aware misbehavior detection for 5G/6G-enabled C-V2X networks.

## Overview

Detects falsified V2X safety messages using a Random Forest classifier, and
benchmarks per-message inference latency against 3GPP TS 22.186 requirements
for each V2X service tier.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.96254 |
| F1 score | 0.94580 |
| p99 inference latency | 1.758 ms |
| Model size | 0.59 MB |

Passes all three 3GPP tiers: basic safety (100 ms), platooning (10 ms),
advanced eV2X (3 ms).

## Dataset

VeReMi Extension (Kamel et al., IEEE ICC 2020) — 3,194,808 labelled V2X messages.

## Stack

Python, scikit-learn, FastAPI, HTML/CSS/JavaScript.

## Running locally

    pip install -r requirements.txt
    uvicorn src.api:app --reload

Then open `frontend/dashboard.html`.