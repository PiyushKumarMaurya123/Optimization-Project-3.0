"""
train_optimize.py  -  C2 / C3 / C5 study
----------------------------------------
Model features (8): T, P1, P2, RM1, RM2, X, YM23, V
  - YM23 is a function of M2 and M3 (the stoichiometric mole-fraction). It is the
    strongest predictor of C2; raw M2/M3 alone predict poorly, and including all
    three is redundant. So YM23 is the model feature.
  - M2 and M3 remain the SETTABLE knobs: the operator sets them, YM23 is computed
    from them. (Formula kept internal; not surfaced in the UI.)
  - V is a settable knob at levels {8,16,24,32,40}. Y_RM12_a and M1 dropped.
Settable knobs: T,P1,P2,RM1,RM2,X,M2,M3 (continuous) + V (5 levels).
Targets: maximize C2 ; minimize C3, C5.
Run:  python train_optimize.py
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error

RAW_CSV, CLEAN_CSV, PKL = "_My_Task_-_Sheet1__1_.csv", "cleaned_data.csv", "models.pkl"
MODEL_COLS = ["T", "P1", "P2", "RM1", "RM2", "X", "YM23", "V"]
KNOBS_CONT = ["T", "P1", "P2", "RM1", "RM2", "X", "M2", "M3"]   # M2,M3 set YM23
V_LEVELS = [8, 16, 24, 32, 40]
K_YM23 = 0.1836                      # internal: YM23 = 100*M2/(M2 + K*M3)
TARGETS = ["C2", "C3", "C5"]


def ym23_from(m2, m3):
    return 100.0 * m2 / (m2 + K_YM23 * m3)


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
    raw["V"] = raw["V"].ffill()
    keep = MODEL_COLS + ["M2", "M3"] + TARGETS
    tidy = raw[keep].dropna().reset_index(drop=True)
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
    mk_c2 = lambda: ExtraTreesRegressor(n_estimators=600, min_samples_leaf=1, random_state=42, n_jobs=-1)
    mk_lin = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    cv = {"C2": loocv(X, d.C2.values, mk_c2),
          "C3": loocv(X, d.C3.values, mk_lin),
          "C5": loocv(X, d.C5.values, mk_lin)}
    for k, (r2, mae) in cv.items():
        print(f"[cv] {k}: R2={r2:.3f}  MAE={mae:.2f}")

    m_c2 = mk_c2().fit(X, d.C2.values)
    m_c3 = mk_lin().fit(X, d.C3.values)
    m_c5 = mk_lin().fit(X, d.C5.values)

    # bounds for every settable knob (incl. M2, M3 ranges for the UI/optimizer)
    bc = {c: [float(d[c].min()), float(d[c].max())] for c in KNOBS_CONT}

    # optimizer: random exploration PLUS the observed points (ExtraTrees predicts
    # highest near real data, so include them to report a sensible optimum)
    rng = np.random.default_rng(0); N = 300_000
    cont = {c: rng.uniform(*bc[c], size=N) for c in KNOBS_CONT}
    YM23 = ym23_from(cont["M2"], cont["M3"])
    Vv = np.array(V_LEVELS)[rng.integers(0, len(V_LEVELS), size=N)]
    feat_rand = np.column_stack([cont["T"], cont["P1"], cont["P2"], cont["RM1"],
                                 cont["RM2"], cont["X"], YM23, Vv])
    feat_obs = d[MODEL_COLS].values
    p_rand = m_c2.predict(feat_rand)
    p_obs = m_c2.predict(feat_obs)
    if p_obs.max() >= p_rand.max():
        j = int(p_obs.argmax()); best_c2 = float(p_obs[j])
        opt = {c: float(d.iloc[j][c]) for c in KNOBS_CONT}
        opt["YM23"] = float(d.iloc[j]["YM23"]); opt["V"] = float(d.iloc[j]["V"])
    else:
        bi = int(p_rand.argmax()); best_c2 = float(p_rand[bi])
        opt = {c: float(cont[c][bi]) for c in KNOBS_CONT}
        opt["YM23"] = float(YM23[bi]); opt["V"] = float(Vv[bi])
    print(f"[opt] model-predicted best C2={best_c2:.2f}%")

    rec = d[d.C5 <= 2].sort_values("C2", ascending=False).iloc[0]
    rec_inputs = {c: float(rec[c]) for c in MODEL_COLS + ["M2", "M3"]}
    rec_meas = {t: float(rec[t]) for t in TARGETS}
    print(f"[rec] real run C2={rec_meas['C2']:.1f}% C3={rec_meas['C3']:.1f}% C5={rec_meas['C5']:.2f}%")

    joblib.dump({
        "model_cols": MODEL_COLS, "knobs_cont": KNOBS_CONT, "v_levels": V_LEVELS,
        "k_ym23": K_YM23, "bounds_cont": bc,
        "model_C2": m_c2, "model_C3": m_c3, "model_C5": m_c5,
        "loocv": {k: [float(v[0]), float(v[1])] for k, v in cv.items()},
        "opt_pred_inputs": opt, "opt_pred_c2": float(best_c2),
        "rec_inputs": rec_inputs, "rec_meas": rec_meas,
        "importances_C2": {c: float(i) for c, i in zip(MODEL_COLS, m_c2.feature_importances_)},
    }, PKL)
    print(f"[save] -> {PKL}")


if __name__ == "__main__":
    main()
