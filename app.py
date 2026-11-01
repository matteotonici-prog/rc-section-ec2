import hashlib
import json
import math

import pandas as pd
import streamlit as st

from core import views
from core.analysis import run_all
from core.geometry import MAX_LAYERS, build_layout
from core.materials import CLASSES
from core.plots import apply_theme, plot_preview, show_fig

st.set_page_config(page_title="RC section check - EC2", layout="wide")
apply_theme()

LAYER_COLS = {
    "n": st.column_config.NumberColumn("n bars", min_value=1, step=1, format="%d"),
    "dia": st.column_config.NumberColumn("dia (mm)", min_value=6.0),
    "gap": st.column_config.NumberColumn("gap to previous layer (mm)", min_value=0.0),
}


def _ser(o):
    return o.to_json() if hasattr(o, "to_json") else str(o)


st.title("RC section check - Eurocode 2")
top = st.container()
(t_geo, t_mat, t_act, t_res, t_uls, t_shear, t_sls, t_beam) = st.tabs(
    ["1 Geometry", "2 Materials", "3 Actions", "4 Section results",
     "5 ULS bending", "6 ULS shear", "7 SLS", "8 Long-term beam"])

# ============================================================ 1 GEOMETRY
with t_geo:
    shape = st.selectbox(
        "Section type",
        ["Rectangular", "T-section", "Rectangular wall", "Circular",
         "Custom (coordinates)"], key="shape")
    g = {"shape": shape, "elem": "Beam"}

    st.subheader("Geometry (mm)")
    if shape == "Rectangular":
        c1, c2 = st.columns(2)
        g["b"] = c1.number_input("Width b", 100.0, 3000.0, 300.0, key="rect_b")
        g["h"] = c2.number_input("Overall depth h", 100.0, 3000.0, 500.0, key="rect_h")
    elif shape == "T-section":
        c1, c2 = st.columns(2)
        g["b"] = c1.number_input("Web width b", 100.0, 3000.0, 300.0, key="tee_b")
        g["h"] = c2.number_input("Overall depth h", 100.0, 3000.0, 600.0, key="tee_h")
        g["bf"] = c1.number_input("Flange width bf", 100.0, 6000.0, 1200.0, key="tee_bf")
        g["hf"] = c2.number_input("Flange depth hf", 50.0, 1000.0, 150.0, key="tee_hf")
    elif shape == "Rectangular wall":
        c1, c2 = st.columns(2)
        g["t_w"] = c1.number_input("Wall thickness t", 100.0, 1500.0, 200.0, key="wall_t")
        g["l_w"] = c2.number_input("Wall length L", 500.0, 20000.0, 2000.0, key="wall_l")
        g["orient"] = st.radio(
            "Wall orientation in the section",
            ["In-plane: length vertical (bending about the strong axis)",
             "Out-of-plane: length horizontal (bending about the weak axis)"],
            key="wall_orient")
    elif shape == "Circular":
        g["diam"] = st.number_input("Diameter D", 200.0, 3000.0, 500.0, key="circ_d")
    else:
        st.caption("Concrete outline: vertices in order (x, y) in mm, y pointing up. "
                   "Add or delete rows as needed.")
        g["poly_df"] = st.data_editor(
            pd.DataFrame({"x": [0.0, 300.0, 300.0, 0.0],
                          "y": [0.0, 0.0, 600.0, 600.0]}),
            num_rows="dynamic", key="poly")

    st.subheader("Reinforcement")
    if shape != "Custom (coordinates)":
        c1, c2, c3 = st.columns(3)
        c_nom = c1.number_input("Nominal cover to links (mm)", 10.0, 100.0, 30.0, key="c_nom")
        link = c2.number_input("Link / horizontal bar dia (mm)", 0.0, 25.0, 10.0, key="link")
        g["dg"] = c3.number_input("Max aggregate size dg (mm)", 8.0, 40.0, 20.0, key="dg")
        g["cover_tot"] = c_nom + link      # cover to the surface of the longitudinal bars
    else:
        g["dg"] = st.number_input("Max aggregate size dg (mm)", 8.0, 40.0, 20.0, key="dg_c")
        g["cover_tot"] = 0.0

    if shape in ("Rectangular", "T-section"):
        if shape == "Rectangular":
            g["elem"] = st.radio("Element type for the maximum spacing check",
                                 ["Beam", "Slab (one-way, main bars)"],
                                 horizontal=True, key="elem")
        st.caption(
            "Bottom layers are counted from the bottom face upwards, top layers "
            "from the top face downwards. The first layer of each group sits at "
            "cover + link dia + bar dia/2 from the face; 'gap' is the clear vertical "
            "distance to the previous layer (ignored for the first layer). Outer "
            "bars of every layer sit at cover + link dia from the sides. "
            f"Up to {MAX_LAYERS} layers in total."
            + (" In a T-section, a layer whose centre is inside the flange spreads "
               "across the flange width, otherwise across the web."
               if shape == "T-section" else ""))
        st.markdown("**Bottom layers**")
        g["bottom_df"] = st.data_editor(
            pd.DataFrame({"n": [4], "dia": [25.0], "gap": [25.0]}),
            num_rows="dynamic", key=f"bot_{shape}", column_config=LAYER_COLS)
        st.markdown("**Top layers**")
        g["top_df"] = st.data_editor(
            pd.DataFrame({"n": [3 if shape == "Rectangular" else 6],
                          "dia": [16.0], "gap": [25.0]}),
            num_rows="dynamic", key=f"top_{shape}", column_config=LAYER_COLS)
    elif shape == "Rectangular wall":
        st.caption("Vertical bars on each face, given as diameter @ spacing along the "
                   "wall length. Bars per face = floor((L − 2·(cover + link + φ/2)) / "
                   "spacing) + 1, centred along the wall.")
        c1, c2 = st.columns(2)
        g["f1_dia"] = c1.number_input("Face 1 bar dia (mm)", 6, 40, 12, key="f1d")
        g["f1_s"] = c1.number_input("Face 1 spacing (mm)", 50, 600, 200, key="f1s")
        g["f2_dia"] = c2.number_input("Face 2 bar dia (mm)", 6, 40, 12, key="f2d")
        g["f2_s"] = c2.number_input("Face 2 spacing (mm)", 50, 600, 200, key="f2s")
        g["d_min"] = st.number_input(
            "Minimum distance between the two layers, centre to centre (mm)",
            50, 1000, 200, key="dmin")
    elif shape == "Circular":
        c1, c2 = st.columns(2)
        g["n_bar"] = c1.number_input("Number of bars (equally spaced)", 2, 60, 8, key="nbar")
        g["dia_bar"] = c2.number_input("Bar dia (mm)", 6, 40, 25, key="diabar")
    else:
        st.caption("Bars: centre coordinates (x, y) and diameter, in mm.")
        g["bar_df"] = st.data_editor(
            pd.DataFrame({"x": [50.0, 150.0, 250.0, 50.0, 250.0],
                          "y": [50.0, 50.0, 50.0, 550.0, 550.0],
                          "dia": [25.0, 25.0, 25.0, 16.0, 16.0]}),
            num_rows="dynamic", key="bars")

    lay = build_layout(g)

    st.subheader("Section preview and bar spacing checks")
    for e in lay["errors"]:
        st.error(e)
    if lay["poly"] is not None and lay["bars"]:
        show_fig(plot_preview(
            lay["poly"], lay["bars"],
            f"{shape}: {lay['n_bars']} bars, As = {lay['as_total']:,.0f} mm²"), width=1)
    if lay["rows"]:
        st.dataframe(pd.DataFrame(lay["rows"]), hide_index=True)
        if any(r["Status"].startswith("⚠️") for r in lay["rows"]):
            st.warning("Some bar spacings do not meet the EC2 limits below.")
        lim_txt = "EC2 8.2(2): clear spacing ≥ max(φ, dg + 5 mm, 20 mm). "
        if shape == "Rectangular wall":
            lim_txt += "EC2 9.6.2(1): vertical bars at most min(3t, 400 mm) apart. "
        elif shape == "Rectangular" and g["elem"].startswith("Slab"):
            lim_txt += "EC2 9.3.1.1(3): main bars at most min(3h, 400 mm) apart. "
        elif shape in ("Rectangular", "T-section"):
            lim_txt += ("Beams: the maximum spacing is governed by crack control "
                        "(EC2 7.3.3), see the crack width check in tab 7. ")
        st.caption(lim_txt + "Recommended values; check the National Annex.")

# ============================================================ 2 MATERIALS
with t_mat:
    st.subheader("Concrete")
    c1, c2 = st.columns(2)
    cls = c1.selectbox("Concrete class", list(CLASSES), index=5, key="cls")
    acc = c1.number_input("αcc", 0.7, 1.0, 0.85, key="acc")
    gamma_c = c1.number_input("γc", 1.0, 2.0, 1.5, key="gc")
    ult_model = c2.radio("Ultimate stress-strain model (EC2 3.1.7)",
                         ["Rectangular stress block", "Parabola-rectangle"], key="ult")

    st.subheader("Reinforcing steel")
    c1, c2 = st.columns(2)
    fyk = c1.number_input("fyk (MPa)", 200.0, 700.0, 500.0, key="fyk")
    gamma_s = c1.number_input("γs", 1.0, 1.5, 1.15, key="gs")
    es = c2.number_input("Es (MPa)", 100e3, 250e3, 200e3, key="es")
    eud = c2.number_input("Design ultimate strain εud", 0.01, 0.1, 0.0225,
                          format="%.4f", key="eud")

    st.subheader("Stiffness and service stresses")
    c1, c2 = st.columns(2)
    basis = c1.radio("Concrete modulus",
                     ["Short-term (Ecm)", "Long-term (modular ratio n)"], key="basis")
    n_long = c2.number_input("Long-term modular ratio n = Es / Ec,eff",
                             5.0, 30.0, 15.0, key="nlong")
    crack_basis = st.radio(
        "Tensile strength for the cracked / uncracked decision, fct,eff",
        ["fctm", "fctm,fl = max((1.6 - h/1000)·fctm, fctm)"], horizontal=True,
        key="cbasis")

    st.subheader("Crack width (EC2 7.3.4)")
    c1, c2 = st.columns(2)
    wmax = c1.number_input("Limiting crack width wmax (mm)", 0.1, 0.5, 0.3, key="wmax")
    crack_dur = c1.radio(
        "Crack width duration",
        ["Long term (kt = 0.4, αe = n)", "Short term (kt = 0.6, αe = Es/Ecm)"],
        key="cdur")
    k1 = c2.number_input("k1 (0.8 high bond, 1.6 plain bars)", 0.8, 1.6, 0.8, key="k1")
    k3 = c2.number_input("k3", 1.0, 5.0, 3.4, key="k3")
    k4 = c2.number_input("k4", 0.1, 1.0, 0.425, key="k4")
    to_face = st.checkbox("Extrapolate the crack width from the bar level to the "
                          "tension face (strain ratio)", value=False, key="toface")

# ============================================================== 3 ACTIONS
with t_act:
    st.subheader("Bending direction")
    direction = st.radio(
        "Direction", ["Sagging (compression at top)",
                      "Hogging (compression at bottom)"], horizontal=True, key="dir")
    st.subheader("Ultimate limit state")
    c1, c2 = st.columns(2)
    n_ed = c1.number_input("N_Ed (kN, compression +ve)", -5000.0, 50000.0, 0.0, key="ned")
    m_ed = c2.number_input("M_Ed (kNm, enter as positive)", 0.0, 50000.0, 150.0, key="med")
    st.subheader("Service combination (stress limits)")
    c1, c2 = st.columns(2)
    n_ser = c1.number_input("N_ser (kN, compression +ve)", -5000.0, 50000.0, 0.0, key="nser")
    m_ser = c2.number_input("M_ser (kNm, enter as positive)", 0.0, 50000.0, 100.0, key="mser")
    st.subheader("Quasi-permanent combination (crack width)")
    c1, c2 = st.columns(2)
    n_qp = c1.number_input("N_qp (kN, compression +ve)", -5000.0, 50000.0, 0.0, key="nqp")
    m_qp = c2.number_input("M_qp (kNm, enter as positive)", 0.0, 50000.0, 80.0, key="mqp")

inp = dict(
    cls=cls, acc=acc, gamma_c=gamma_c, ult_model=ult_model, fyk=fyk,
    gamma_s=gamma_s, es=es, eud=eud, basis_long=basis.startswith("Long"),
    n_long=n_long, crack_basis_fl=crack_basis.startswith("fctm,fl"), wmax=wmax,
    crack_long=crack_dur.startswith("Long"), k1=k1, k3=k3, k4=k4, to_face=to_face,
    n_ed=n_ed, m_ed=m_ed, n_ser=n_ser, m_ser=m_ser, n_qp=n_qp, m_qp=m_qp,
    sagging=direction.startswith("Sagging"))

# ============================================================== CALCULATE
cur_hash = hashlib.md5(json.dumps({"g": g, "inp": inp}, sort_keys=True,
                                  default=_ser).encode()).hexdigest()
with top:
    c1, c2 = st.columns([1, 3])
    calc = c1.button("▶ Calculate", type="primary", disabled=not lay["ok"])
    status = c2.empty()
    if calc:
        with st.spinner("Running the analyses (this can take a minute)..."):
            R_new = run_all(lay, g, inp)
            R_new["hash"] = cur_hash
            st.session_state["R"] = R_new

R = st.session_state.get("R")
if R is None:
    status.info("Fill in tabs 1 to 3, then press Calculate.")
elif R["hash"] != cur_hash:
    status.warning("Inputs changed since the last calculation. "
                   "Press Calculate to update the results.")
else:
    status.success("Results are up to date.")


def need_results():
    if R is None:
        st.info("No results yet. Fill in tabs 1 to 3 and press Calculate.")
        return False
    return True


# =============================================================== RESULTS
with t_res:
    if need_results():
        views.geometry_results(R)
with t_uls:
    if need_results():
        views.uls_bending(R)
with t_shear:
    st.info("ULS shear (links, VRd,s, VRd,max, VRd versus VEd) comes in phase 2.")
with t_sls:
    if need_results():
        views.sls(R)
with t_beam:
    st.info("Long-term beam deflection (simply supported or cantilever) "
            "comes in phase 3.")
