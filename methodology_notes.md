# Methodology notes — C2 / C3 / C5 optimization

## Objective (unchanged)
Maximize **C2** (main-product yield), minimize byproducts **C5** and **C3**.

## Feature set (10): `T, P1, P2, RM1, RM2, X, M2, M3, YM23, V`
- `Y_RM12_a = RM1/RM2×100` dropped (redundant with RM1, RM2). `M1` dropped (constant).
- `YM23` kept — the M-charges act on the product through this combined quantity
  (dropping it collapses C2 R² 0.58→0.45). M2, M3 also kept as explicit inputs.
- **`V` added** (updated data: levels 8/16/24/32/40). It improves C2
  (LOOCV R² 0.58→0.60, importance rank 3) and correlates **−0.67 with C5** —
  higher V suppresses C5, helping the minimize-C5 goal.
  *Caveat:* only ~6 runs have V≠40 (just 1 each at V=24, 32), so V's effect is
  under-sampled — solid as a hypothesis, worth a few confirming low-V runs.
36 usable runs.

## Models & validation (LOOCV, n=36)
- **C2** → Random Forest, R² ≈ 0.60 (reliable).
- **C3, C5** → Ridge, R² ≈ 0.17 / 0.16 (directional only).
- Optimizer varies 6 continuous knobs + V {8,16,24,32,40} + an observed (M2,M3)
  recipe (fixes YM23), strictly within observed ranges.

## The three components
- **C2 — maximize (≈71%).** T=75, X≈45–60, high RM, high YM23, V=40. Drivers: YM23, T, X, V.
- **C5 — suppress for free.** Kinetic intermediate; spikes at short residence time
  (32.7% at X=12), <2% for X≥45. Long X *and* high V both lower C5, and both align
  with high C2. P1≈6.2, P2≈1.5 trims C5 to ≈1.3% at negligible C2 cost.
- **C3 — coupled to C2 (+0.73); cannot be tuned away.** Best selectivity
  C2/(C2+C3)≈0.77 is already at the C2 optimum; forcing C3≤20% drops C2 to ~12%.
  Reducing C3 is a chemistry/selectivity problem, not a tuning one.

## Recommended operating point (best real run, C5-aware)
T=75, P1=6.2, P2=1.5, RM1=13.7, RM2=39.6, X=45, M2=4.0, M3=10 (→YM23≈68.5), **V=40**.
**Measured: C2 ≈ 71.3%, C5 ≈ 1.3%, C3 ≈ 23.2%.**

## Honesty point
The model's predicted optimum (~69%) does **not** beat this real run (~71.3%) —
optimization **re-identifies** your best existing condition. Beating it needs
**new experiments**; treat recommendations as the best next experiment.
