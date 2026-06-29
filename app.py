"""
app.py  -  C2 / C3 / C5 Optimizer (10 features incl. V)
-------------------------------------------------------
Model features: T, P1, P2, RM1, RM2, X, M2, M3, YM23, V
Knobs: T,P1,P2,RM1,RM2,X (continuous) + V {8,16,24,32,40} + (M2,M3) recipe.
Run:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="C2 / C3 / C5 Optimizer", page_icon="🧪", layout="wide")
PKL = "models.pkl"


@st.cache_resource(show_spinner="Loading models…")
def load():
    import joblib
    if os.path.exists(PKL):
        try:
            return joblib.load(PKL)
        except Exception:
            pass
    import train_optimize as t
    t.main()
    return joblib.load(PKL)


b = load()
COLS = b["model_cols"]
CONT = b["knobs_cont"]
BC = b["bounds_cont"]
RECIPES = b["recipes"]
VLEV = b["v_levels"]
rec, rec_meas = b["rec_inputs"], b["rec_meas"]
opt, opt_c2 = b["opt_pred_inputs"], b["opt_pred_c2"]
loocv, imp = b["loocv"], b["importances_C2"]


def predict(cont_vals, m2, m3, ym23, v):
    row = {**cont_vals, "M2": m2, "M3": m3, "YM23": ym23, "V": v}
    arr = np.array([[row[c] for c in COLS]])
    return (float(b["model_C2"].predict(arr)[0]),
            max(0.0, float(b["model_C3"].predict(arr)[0])),
            max(0.0, float(b["model_C5"].predict(arr)[0])))


st.title("🧪 C2 / C3 / C5 Optimizer")
st.caption(
    "10 inputs incl. V (levels 8/16/24/32/40). "
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
    st.caption(f"M2={rec['M2']}, M3={rec['M3']} → YM23={rec['YM23']:.1f}. "
               "(V=40 also keeps C5 low — V correlates −0.67 with C5.)")

    st.divider()
    st.markdown(
        f"**Model's predicted optimum (synthetic):** C2 ≈ **{opt_c2:.1f}%** — does *not* beat "
        f"the best real run ({rec_meas['C2']:.1f}%). Optimization re-identifies your best "
        "existing condition; beating it needs new experiments.")

    st.subheader("What drives C2 (feature importance)")
    st.bar_chart(pd.Series(imp).reindex(COLS).sort_values(), horizontal=True)

with t2:
    st.subheader("Adjust the knobs → live predictions")
    st.caption("C2 reliable; C3 & C5 directional. M2/M3 set as a recipe (fixes YM23).")
    ca, cb = st.columns(2)
    cont_vals = {}
    for i, cname in enumerate(CONT):
        lo, hi = BC[cname]
        tgt = ca if i % 2 == 0 else cb
        step = max((hi - lo) / 100.0, 0.01)
        cont_vals[cname] = tgt.slider(cname, float(lo), float(hi), float(rec[cname]),
                                      step=float(round(step, 3)))
    v = ca.select_slider("V (multiples of 8)", options=VLEV, value=int(rec["V"]))
    labels = [f"M2={r[0]}, M3={r[1]}  (→ YM23={r[2]})" for r in RECIPES]
    didx = int(np.argmin([abs(r[0]-rec["M2"])+abs(r[1]-rec["M3"]) for r in RECIPES]))
    ridx = cb.selectbox("M2 / M3 recipe", range(len(RECIPES)),
                        format_func=lambda i: labels[i], index=didx)
    m2, m3, ym23 = RECIPES[ridx]

    pc2, pc3, pc5 = predict(cont_vals, m2, m3, ym23, v)
    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("C2 predicted", f"{pc2:.1f} %", delta=f"{pc2-rec_meas['C2']:+.1f} vs rec")
    k2.metric("C3 predicted", f"{pc3:.1f} %")
    k3.metric("C5 predicted", f"{pc5:.2f} %")
    if cont_vals["X"] <= 36:
        st.warning("Short residence time (X ≤ 36): C5 spikes in this zone.")
    if v <= 16:
        st.info("Low V is under-sampled in the data — treat C5/C2 here as a hypothesis.")
    if pc3 > 25:
        st.warning("C3 is high here — it rises together with C2.")
