"""
train_optimize.py  -  C2 / C3 / C5 study  (with V)
--------------------------------------------------
Features (10): T, P1, P2, RM1, RM2, X, M2, M3, YM23, V
  - V is a settable knob at discrete levels {8,16,24,32,40} (multiples of 8).
    It improves C2 (LOOCV R^2 0.58->0.60, importance rank 3) and correlates
    strongly with C5 (-0.67), so it helps the minimize-C5 goal. NOTE: low-V
    levels are under-sampled (1-2 runs each) -> directional evidence there.
  - Y_RM12_a (=RM1/RM2) and M1 (constant) dropped. YM23 kept (combined M2/M3).
Settable knobs: T,P1,P2,RM1,RM2,X (continuous) + V (5 levels) + (M2,M3) recipe.
Targets: maximize C2 ; minimize C3, C5.
Run:  python train_optimize.py
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error

RAW_CSV, CLEAN_CSV, PKL = "_My_Task_-_Sheet1__1_.csv", "cleaned_data.csv", "models.pkl"
MODEL_COLS = ["T", "P1", "P2", "RM1", "RM2", "X", "M2", "M3", "YM23", "V"]
KNOBS_CONT = ["T", "P1", "P2", "RM1", "RM2", "X"]
V_LEVELS = [8, 16, 24, 32, 40]                 # multiples of 8, 8..40
TARGETS = ["C2", "C3", "C5"]


def _to_num(s):
    s = s.astype(str).str.strip().replace({"NA": np.nan, "nan": np.nan, "": np.nan})
    return pd.to_numeric(s.str.replace("%", "", regex=False), errors="coerce")


def load_and_clean(path=RAW_CSV):
    df = pd.read_csv(path, header=None)
    raw = df.iloc[4:].copy().reset_index(drop=True)
    raw.columns = ["_drop", "Trial", "Cond", "V", "T", "P1", "P2", "RM1", "RM2",
                   "Y_RM12_a", "X", "YM23", "M1", "M2", "M3", "TARGET",
                   "C1", "C2", "C3", "C4", "C5", "C6", "Y1", "Y2", "Y1Y2"]
    for c in raw.columns:
        raw[c] = _to_num(raw[c])
    raw["V"] = raw["V"].ffill()                 # V only on trial-start rows
    tidy = raw[MODEL_COLS + TARGETS].dropna().reset_index(drop=True)
    tidy.to_csv(CLEAN_CSV, index=False)
    print(f"[clean] {len(tidy)} rows -> {CLEAN_CSV}")
    return tidy


def loocv(X, y, make):
    p = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        m = make(); m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
    return r2_score(y, p), mean_absolute_error(y, p)


def main():
    d = load_and_clean()
    X = d[MODEL_COLS].values
    mk_c2 = lambda: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    mk_lin = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    cv = {"C2": loocv(X, d.C2.values, mk_c2),
          "C3": loocv(X, d.C3.values, mk_lin),
          "C5": loocv(X, d.C5.values, mk_lin)}
    for k, (r2, mae) in cv.items():
        print(f"[cv] {k}: R2={r2:.3f}  MAE={mae:.2f}")

    m_c2 = mk_c2().fit(X, d.C2.values)
    m_c3 = mk_lin().fit(X, d.C3.values)
    m_c5 = mk_lin().fit(X, d.C5.values)

    recipes = (d.groupby(["M2", "M3"])["YM23"].mean().reset_index()
               .round({"YM23": 2}).values.tolist())
    bc = {c: [float(d[c].min()), float(d[c].max())] for c in KNOBS_CONT}

    rng = np.random.default_rng(0); N = 300_000
    cont = {c: rng.uniform(*bc[c], size=N) for c in KNOBS_CONT}
    ridx = rng.integers(0, len(recipes), size=N)
    M2 = np.array([recipes[i][0] for i in ridx]); M3 = np.array([recipes[i][1] for i in ridx])
    YM23 = np.array([recipes[i][2] for i in ridx])
    Vv = np.array(V_LEVELS)[rng.integers(0, len(V_LEVELS), size=N)]
    feat = np.column_stack([cont["T"], cont["P1"], cont["P2"], cont["RM1"],
                            cont["RM2"], cont["X"], M2, M3, YM23, Vv])
    pc2 = m_c2.predict(feat); bi = int(pc2.argmax())
    opt = {c: float(v) for c, v in zip(MODEL_COLS, feat[bi])}
    print(f"[opt] model-predicted best C2={pc2[bi]:.2f}% (synthetic)")

    rec = d[d.C5 <= 2].sort_values("C2", ascending=False).iloc[0]
    rec_inputs = {c: float(rec[c]) for c in MODEL_COLS}
    rec_meas = {t: float(rec[t]) for t in TARGETS}
    print(f"[rec] real run C2={rec_meas['C2']:.1f}% C3={rec_meas['C3']:.1f}% C5={rec_meas['C5']:.2f}% V={rec_inputs['V']:.0f}")

    joblib.dump({
        "model_cols": MODEL_COLS, "knobs_cont": KNOBS_CONT, "v_levels": V_LEVELS,
        "bounds_cont": bc, "recipes": recipes,
        "model_C2": m_c2, "model_C3": m_c3, "model_C5": m_c5,
        "loocv": {k: [float(v[0]), float(v[1])] for k, v in cv.items()},
        "opt_pred_inputs": opt, "opt_pred_c2": float(pc2[bi]),
        "rec_inputs": rec_inputs, "rec_meas": rec_meas,
        "importances_C2": {c: float(i) for c, i in zip(MODEL_COLS, m_c2.feature_importances_)},
    }, PKL)
    print(f"[save] -> {PKL}")


if __name__ == "__main__":
    main()
