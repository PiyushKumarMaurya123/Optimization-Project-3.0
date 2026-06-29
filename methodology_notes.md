# Methodology notes — C2 / C3 / C5 optimization

## Objective
Maximize **C2** (main-product yield), minimize byproducts **C5** and **C3**.

## Feature set (8 model features): `T, P1, P2, RM1, RM2, X, YM23, V`
- **YM23 is a function of M2 and M3** (the stoichiometric composition). It is the
  model feature because it is the strongest single predictor of C2. You still set
  **M2 and M3** as the knobs; YM23 is computed from them.
- Tested feature choice (LOOCV): **YM23 only** beats both alternatives —
  C2 R²=0.62 / C3 R²=0.23, vs raw M2,M3 only (C2 0.49) or all three (C2 0.61).
  Raw M2/M3 alone can't carry the signal; all three is redundant.
- **V** kept (levels 8/16/24/32/40): improves C2 and correlates −0.67 with C5
  (higher V → less C5). Low-V levels are under-sampled (1–2 runs) → directional.
- `Y_RM12_a` (=RM1/RM2) and `M1` (constant) dropped. 36 usable runs.

## Models & validation (LOOCV, n=36)
- **C2** → Extra Trees (Extremely Randomized Trees), R² ≈ 0.60 (reliable).
  Chosen over Random Forest because RF's averaging compressed predictions (couldn't
  reach the ~71% peak); Extra Trees reaches the correct high values at equal accuracy.
- **C3, C5** → Ridge, R² ≈ 0.23 / 0.16 (directional only).
- Optimizer varies the settable knobs (T,P1,P2,RM1,RM2,X,M2,M3 + V), computing
  YM23 from M2/M3 — within observed ranges (no extrapolation).

## The three components
- **C2 — maximize (≈71%).** T=75, X≈45–60, high RM, high YM23, V=40. Drivers: YM23, T, X, V.
- **C5 — suppress for free.** Kinetic intermediate; spikes at short residence time
  (32.7% at X=12), <2% for X≥45. Long X and high V both lower it, both align with high C2.
- **C3 — coupled to C2 (+0.73); cannot be tuned away.** Best selectivity
  C2/(C2+C3)≈0.77 is already at the C2 optimum; forcing C3≤20% drops C2 to ~12%.
  Reducing C3 is a chemistry/selectivity problem, not a tuning one.

## Recommended operating point (best real run, C5-aware)
T=75, P1=6.2, P2=1.5, RM1=13.7, RM2=39.6, X=45, M2=4.0, M3=10, V=40.
**Measured: C2 ≈ 71.3%, C5 ≈ 1.3%, C3 ≈ 23.2%.**

## Honesty point
The optimum search returns ~71.6%, landing on this same real run — optimization
**re-identifies** your best existing condition rather than finding a new one. Beating it needs
**new experiments**; treat recommendations as the best next experiment.
