# C2 / C3 / C5 Optimizer

Maximize **C2**, minimize **C5** and **C3**, with an interactive Streamlit app.

## Features (8 model features)
`T, P1, P2, RM1, RM2, X, YM23, V`
- **YM23 is a function of M2 and M3.** You set M2 & M3 as knobs; YM23 is computed
  from them and used by the model (it's the strongest C2 predictor — chosen over
  raw M2/M3 or all-three by a LOOCV test).
- **V** kept (levels {8,16,24,32,40}); improves C2 and lowers C5 at higher V.
  Low-V levels under-sampled (caveat in notes).
- `Y_RM12_a` (=RM1/RM2) and `M1` (constant) dropped. 36 runs.

## Files
| File | What it is |
|---|---|
| `train_optimize.py` | Clean → train C2/C3/C5 → optimizer → `models.pkl` |
| `app.py` | Streamlit app (Recommended Condition + What-If Explorer) |
| `methodology_notes.md` | Mentor-facing write-up |
| `models.pkl` | Saved model bundle |
| `cleaned_data.csv` | Tidy table |
| `requirements.txt` | Dependencies |
| `_My_Task_-_Sheet1__1_.csv` | Latest raw export |

## Run
```bash
pip install -r requirements.txt
python train_optimize.py     # optional rebuild
streamlit run app.py
```

## Headline
- **C2** best ≈ 71% (T=75, X≈45, high RM, high YM23, V=40). Drivers: YM23, T, X, V.
- **C5** suppressed for free (long residence time and high V both help, both align with C2).
- **C3** coupled to C2 (+0.73); ~22% is the structural tax for ~70% C2 — needs new chemistry.
- Model optimum does not beat the best real run; beating ~71% needs new experiments.

## Validation (LOOCV, n=36)
C2 R² ≈ 0.62 (reliable) · C3 R² ≈ 0.23 · C5 R² ≈ 0.16 (directional).
