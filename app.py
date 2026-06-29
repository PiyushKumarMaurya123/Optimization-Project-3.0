"""
app.py  -  C2 / C3 / C5 Optimizer
---------------------------------
Model features: T, P1, P2, RM1, RM2, X, YM23, V
YM23 is a function of M2 and M3 (computed internally). You set M2 and M3.
Run:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="C2 / C3 / C5 Optimizer", page_icon="🧪", layout="wide")
PKL = "models.pkl"


REQUIRED_KEYS = {"model_cols", "knobs_cont", "v_levels", "k_ym23", "bounds_cont",
                 "model_C2", "model_C3", "model_C5", "loocv", "importances_C2",
                 "opt_pred_inputs", "opt_pred_c2", "rec_inputs", "rec_meas"}


@st.cache_resource(show_spinner="Loading models…")
def load():
    import joblib
    if os.path.exists(PKL):
        try:
            b = joblib.load(PKL)
            if REQUIRED_KEYS.issubset(b.keys()):   # ignore a stale/old bundle
                return b
        except Exception:
            pass
    # Missing, unreadable, or out-of-date bundle -> rebuild from the raw CSV.
    import train_optimize as t
    t.main()
    return joblib.load(PKL)


b = load()
COLS = b["model_cols"]                       # T,P1,P2,RM1,RM2,X,YM23,V
CONT = b["knobs_cont"]                        # T,P1,P2,RM1,RM2,X,M2,M3
BC = b["bounds_cont"]
VLEV = b["v_levels"]
K = b.get("k_ym23", 0.1836)
rec, rec_meas = b["rec_inputs"], b["rec_meas"]
opt, opt_c2 = b["opt_pred_inputs"], b["opt_pred_c2"]
loocv, imp = b["loocv"], b["importances_C2"]


def ym23_from(m2, m3):
    return 100.0 * m2 / (m2 + K * m3)        # YM23 = f(M2, M3)


def predict(vals):
    ym23 = ym23_from(vals["M2"], vals["M3"])
    row = {**vals, "YM23": ym23}
    arr = np.array([[row[c] for c in COLS]])
    return (float(b["model_C2"].predict(arr)[0]),
            max(0.0, float(b["model_C3"].predict(arr)[0])),
            max(0.0, float(b["model_C5"].predict(arr)[0])), ym23)


st.title("🧪 C2 / C3 / C5 Optimizer")
st.caption(
    "Inputs: T, P1, P2, RM1, RM2, X, M2, M3, V. YM23 is a function of M2 and M3. "
    f"LOOCV: C2 R²={loocv['C2'][0]:.2f} (reliable) · "
    f"C3 R²={loocv['C3'][0]:.2f} · C5 R²={loocv['C5'][0]:.2f} (directional).")

t1, t2 = st.tabs(["① Recommended Condition", "② What-If Explorer"])

with t1:
    st.subheader("Recommended operating point (best real measured run)")
    a, c, e = st.columns(3)
    a.metric("C2 (maximize)", f"{rec_meas['C2']:.1f} %")
    c.metric("C5 (minimized)", f"{rec_meas['C5']:.2f} %")
    e.metric("C3 (co-product)", f"{rec_meas['C3']:.1f} %",
             help="C3 tracks C2 (corr +0.73); already near the best split the data allows.")
    show = ["T", "P1", "P2", "RM1", "RM2", "X", "M2", "M3", "V"]
    st.dataframe(pd.DataFrame({"Set": show, "Value": [round(rec[c], 2) for c in show]}),
                 hide_index=True, use_container_width=True)
    st.caption("YM23 (a function of M2 and M3) is computed automatically from the M2/M3 you set.")

    st.divider()
    st.markdown(
        f"**Model's predicted optimum (synthetic):** C2 ≈ **{opt_c2:.1f}%** — does *not* beat "
        f"the best real run ({rec_meas['C2']:.1f}%). Optimization re-identifies your best "
        "existing condition; beating it needs new experiments.")

    st.subheader("What drives C2 (feature importance)")
    st.bar_chart(pd.Series(imp).reindex(COLS).sort_values(), horizontal=True)

with t2:
    st.subheader("Adjust the knobs → live predictions")
    st.caption("C2 reliable; C3 & C5 directional. YM23 is a function of M2 and M3 (auto-computed).")
    ca, cb = st.columns(2)
    vals = {}
    for i, cname in enumerate(CONT):
        lo, hi = BC[cname]
        tgt = ca if i % 2 == 0 else cb
        step = max((hi - lo) / 100.0, 0.01)
        vals[cname] = tgt.slider(cname, float(lo), float(hi), float(rec[cname]),
                                 step=float(round(step, 3)))
    vals["V"] = ca.select_slider("V (multiples of 8)", options=VLEV, value=int(rec["V"]))

    pc2, pc3, pc5, ym23 = predict(vals)
    st.caption(f"YM23 = {ym23:.1f}  (function of M2={vals['M2']:.2f}, M3={vals['M3']:.2f})")
    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("C2 predicted", f"{pc2:.1f} %", delta=f"{pc2-rec_meas['C2']:+.1f} vs rec")
    k2.metric("C3 predicted", f"{pc3:.1f} %")
    k3.metric("C5 predicted", f"{pc5:.2f} %")
    if vals["X"] <= 36:
        st.warning("Short residence time (X ≤ 36): C5 spikes in this zone.")
    if vals["V"] <= 16:
        st.info("Low V is under-sampled in the data — treat results here as a hypothesis.")
    if pc3 > 25:
        st.warning("C3 is high here — it rises together with C2.")
