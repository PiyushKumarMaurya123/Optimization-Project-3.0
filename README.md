# C2 / C3 / C5 Optimizer

Maximize **C2**, minimize **C5** and **C3**, with an interactive Streamlit app.

## Features (10)
`T, P1, P2, RM1, RM2, X, M2, M3, YM23, V`
- RM group: RM1, RM2 kept; `Y_RM12_a` (=RM1/RM2) dropped (redundant).
- M group: M2, M3 + `YM23` (combined M2/M3 signal).
- **V** kept — settable at {8,16,24,32,40}; improves C2 and correlates −0.67 with
  C5 (higher V → less C5). Low-V levels are under-sampled (caveat in notes).
- `M1` dropped (constant). 36 runs.

## Files
| File | What it is |
|---|---|
| `train_optimize.py` | Clean → train C2/C3/C5 → optimizer → `models.pkl` |
| `app.py` | Streamlit app (Recommended Condition + What-If Explorer) |
| `methodology_notes.md` | Mentor-facing write-up |
| `models.pkl` | Saved model bundle |
| `cleaned_data.csv` | Tidy table |
| `requirements.txt` | Dependencies |
| `_My_Task_-_Sheet1__1_.csv` | Latest raw export (updated V column) |

## Run
```bash
pip install -r requirements.txt
python train_optimize.py     # optional rebuild
streamlit run app.py
```

## Headline
- **C2** best ≈ 71% (T=75, X≈45, high RM, high YM23, V=40). Drivers: YM23, T, X, V.
- **C5** suppressed for free — long residence time *and* high V both lower it, both align with high C2.
- **C3** coupled to C2 (+0.73); ~22% is the structural tax for ~70% C2 — needs new chemistry.
- Model optimum does not beat the best real run; beating ~71% needs new experiments.

## Validation (LOOCV, n=36)
C2 R² ≈ 0.60 (reliable) · C3 R² ≈ 0.17 · C5 R² ≈ 0.16 (directional).
