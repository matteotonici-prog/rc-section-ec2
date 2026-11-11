import hashlib
import json
import io
import re

import pandas as pd
import streamlit as st

from core import views
from core.analysis import build_ctx, geo_results, run_sls, run_uls
from core.geometry import BEAM_SHAPES, FLANGED, MAX_LAYERS, build_layout
from core.materials import CLASSES
from core.plots import apply_theme, plot_preview, show_fig
from core.shear import auto_shear_dims

st.set_page_config(page_title="RC section check - EC2", layout="wide")
apply_theme()

LAYER_COLS = {
    "n": st.column_config.NumberColumn("n bars", min_value=1, step=1, format="%d"),
    "dia": st.column_config.NumberColumn("dia (mm)", min_value=6.0),
    "gap": st.column_config.NumberColumn("gap to previous layer (mm)", min_value=0.0),
}
COMBO_COLS = ["N (kN)", "V major (kN)", "V minor (kN)", "M major (kNm)", "M minor (kNm)"]


def _ser(o):
    return o.to_json() if hasattr(o, "to_json") else str(o)


def digest(*objs):
    return hashlib.md5(json.dumps(list(objs), sort_keys=True,
                                  default=_ser).encode()).hexdigest()


def status_line(box, res, cur_hash, what):
    if res is None:
        box.info(f"Fill in the inputs and press the {what} button.")
    elif res["hash"] != cur_hash:
        box.warning(f"Inputs changed since the last {what} run. Press the button to update.")
    else:
        box.success(f"{what} results are up to date.")


st.title("RC section check - Eurocode 2")
(t_geo, t_mat, t_res, t_uls, t_sls, t_beam) = st.tabs(
    ["1 Geometry", "2 Materials", "3 Section results", "4 ULS check",
     "5 SLS check", "6 Long-term beam"])

# ============================================================ 1 GEOMETRY
with t_geo:
    element = st.radio("What do you want to analyse?",
                       ["Beam", "Column", "RC wall", "Generic section"],
                       horizontal=True, key="element")
    if element == "Beam":
        shapes = BEAM_SHAPES
    elif element == "Column":
        shapes = ["Rectangular", "Circular"]
    elif element == "RC wall":
        shapes = ["Rectangular wall"]
    else:
        shapes = ["Rectangular", "T-section", "Rectangular wall", "Circular",
                  "Custom (coordinates)"]
    shape = st.selectbox("Section type", shapes, key=f"shape_{element}")
    if shape == "Rectangular wall":
        reinf = "wall"
    elif shape == "Circular":
        reinf = "ring"
    elif shape == "Custom (coordinates)":
        reinf = "custom"
    elif element == "Column":
        reinf = "perimeter"
    else:
        reinf = "layers"
    g = {"element": element, "shape": shape, "reinf": reinf, "elem": "Beam"}
    k = f"{element}_{shape}"

    st.subheader("Geometry (mm)")
    if shape == "Rectangular" or shape in FLANGED:
        c1, c2 = st.columns(2)
        g["b"] = c1.number_input("Web width b" if shape in FLANGED else "Width b",
                                 100.0, 3000.0, 300.0, key=f"{k}_b")
        g["h"] = c2.number_input("Overall depth h", 100.0, 5000.0, 500.0, key=f"{k}_h")
        if shape in FLANGED:
            g["bf"] = c1.number_input("Flange width bf (total, including the web)",
                                      100.0, 6000.0, 1200.0, key=f"{k}_bf")
            g["hf"] = c2.number_input("Flange depth hf", 50.0, 1000.0, 150.0,
                                      key=f"{k}_hf")
            if shape in ("L-section", "Inverted L-section"):
                st.caption("The flange extends to the +x side of the web.")
    elif shape == "Rectangular wall":
        c1, c2 = st.columns(2)
        g["t_w"] = c1.number_input("Wall thickness t", 100.0, 1500.0, 200.0, key="wall_t")
        g["l_w"] = c2.number_input("Wall length L", 500.0, 20000.0, 2000.0, key="wall_l")
        g["orient"] = st.radio(
            "Wall orientation in the section",
            ["In-plane: length vertical (bending about the strong axis)",
             "Out-of-plane: length horizontal (bending about the weak axis)"],
            key=f"wall_orient_{element}")
    elif shape == "Circular":
        g["diam"] = st.number_input("Diameter D", 200.0, 3000.0, 500.0, key=f"{k}_d")
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
        g["c_nom"], g["link"] = c_nom, link
        g["cover_tot"] = c_nom + link      # cover to the surface of the longitudinal bars
    else:
        c1, c2 = st.columns(2)
        g["dg"] = c1.number_input("Max aggregate size dg (mm)", 8.0, 40.0, 20.0, key="dg_c")
        g["link"] = c2.number_input("Link diameter, for the shear check (mm)",
                                    0.0, 25.0, 10.0, key="link_c")
        g["c_nom"] = None
        g["cover_tot"] = 0.0

    if reinf == "layers":
        if shape == "Rectangular":
            g["elem"] = st.radio("Element type for the maximum spacing check",
                                 ["Beam", "Slab (one-way, main bars)"],
                                 horizontal=True, key=f"elem_{element}")
        st.caption(
            "Bottom layers are counted from the bottom face upwards, top layers "
            "from the top face downwards. The first layer of each group sits at "
            "cover + link dia + bar dia/2 from the face; 'gap' is the clear vertical "
            "distance to the previous layer (ignored for the first layer). Outer "
            "bars of every layer sit at cover + link dia from the sides, and a layer "
            "spans the section width at its height. "
            f"Up to {MAX_LAYERS} layers in total.")
        st.markdown("**Bottom layers**")
        g["bottom_df"] = st.data_editor(
            pd.DataFrame({"n": [4], "dia": [25.0], "gap": [25.0]}),
            num_rows="dynamic", key=f"bot_{k}", column_config=LAYER_COLS)
        st.markdown("**Top layers**")
        g["top_df"] = st.data_editor(
            pd.DataFrame({"n": [3 if shape == "Rectangular" else 6],
                          "dia": [16.0], "gap": [25.0]}),
            num_rows="dynamic", key=f"top_{k}", column_config=LAYER_COLS)
    elif reinf == "perimeter":
        st.caption("Perimeter bars: bars along the top and bottom faces (corner bars "
                   "included) plus intermediate bars on each side face.")
        c1, c2, c3 = st.columns(3)
        g["nb"] = c1.number_input("Bars along the top and bottom faces (min 2)",
                                  2, 20, 3, key="col_nb")
        g["nh"] = c2.number_input("Intermediate bars on each side face", 0, 20, 1,
                                  key="col_nh")
        g["col_dia"] = c3.number_input("Bar dia (mm)", 8, 40, 25, key="col_dia")
    elif reinf == "wall":
        st.caption("Vertical bars on each face, given as diameter @ spacing along the "
                   "wall length. Bars per face = floor((L − 2·(cover + link + φ/2)) / "
                   "spacing) + 1, centred along the wall.")
        c1, c2 = st.columns(2)
        g["f1_dia"] = c1.number_input("Face 1 bar dia (mm)", 6, 40, 12, key="f1d")
        g["f1_s"] = c1.number_input("Face 1 spacing (mm)", 50, 600, 200, key="f1s")
        g["f2_dia"] = c2.number_input("Face 2 bar dia (mm)", 6, 40, 12, key="f2d")
        g["f2_s"] = c2.number_input("Face 2 spacing (mm)", 50, 600, 200, key="f2s")
    elif reinf == "ring":
        c1, c2 = st.columns(2)
        g["n_bar"] = c1.number_input("Number of bars (equally spaced)", 2, 60, 8, key=f"nbar_{element}")
        g["dia_bar"] = c2.number_input("Bar dia (mm)", 6, 40, 25, key=f"diabar_{element}")
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
        if reinf == "wall":
            lim_txt += "EC2 9.6.2(1): vertical bars at most min(3t, 400 mm) apart. "
        elif reinf == "layers" and shape == "Rectangular" and g["elem"].startswith("Slab"):
            lim_txt += "EC2 9.3.1.1(3): main bars at most min(3h, 400 mm) apart. "
        elif reinf == "layers" and element in ("Beam", "Generic section"):
            lim_txt += ("Beams: the maximum spacing is governed by crack control "
                        "(EC2 7.3.3), see the crack width check in tab 5. ")
        st.caption(lim_txt + "Recommended values; check the National Annex.")

# ============================================================ 2 MATERIALS
with t_mat:
    st.subheader("Concrete")
    c1, c2 = st.columns(2)
    cls = c1.selectbox("Concrete class", list(CLASSES), index=5, key="cls")
    acc = c1.number_input("αcc (flexure and axial load)", 0.7, 1.0, 0.85, key="acc")
    acc_shear = c1.number_input("αcc for shear (UK NA: 1.0)", 0.7, 1.0, 1.0, key="accs")
    gamma_c = c1.number_input("γc", 1.0, 2.0, 1.5, key="gc")
    ult_model = c2.radio("Ultimate stress-strain model (EC2 3.1.7)",
                         ["Rectangular stress block", "Parabola-rectangle"], key="ult")

    st.subheader("Reinforcing steel")
    c1, c2 = st.columns(2)
    fyk = c1.number_input("fyk, longitudinal bars (MPa)", 200.0, 700.0, 500.0, key="fyk")
    fywk = c1.number_input("fywk, shear links (MPa)", 200.0, 700.0, 500.0, key="fywk")
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

mat = dict(
    cls=cls, acc=acc, acc_shear=acc_shear, gamma_c=gamma_c, ult_model=ult_model,
    fyk=fyk, fywk=fywk, gamma_s=gamma_s, es=es, eud=eud,
    basis_long=basis.startswith("Long"), n_long=n_long,
    crack_basis_fl=crack_basis.startswith("fctm,fl"), wmax=wmax,
    crack_long=crack_dur.startswith("Long"), k1=k1, k3=k3, k4=k4, to_face=to_face)

# ===================================== section model and automatic results
geo_hash = digest(g, mat)
ctx, G = None, None
if lay["ok"]:
    cached = st.session_state.get("ctx")
    if cached and cached[0] == geo_hash:
        ctx = cached[1]
    else:
        try:
            with st.spinner("Building the section model..."):
                ctx = build_ctx(lay, g, mat)
            st.session_state["ctx"] = (geo_hash, ctx)
        except Exception as exc:
            st.error(f"The section model could not be built: {exc!r}")
    if ctx is not None:
        cached = st.session_state.get("geo")
        if cached and cached[0] == geo_hash:
            G = cached[1]
        else:
            with st.spinner("Calculating the section properties..."):
                G = geo_results(ctx, lay)
            st.session_state["geo"] = (geo_hash, G)

# ============================================================ 4 ULS CHECK
with t_uls:
    st.subheader("Design actions (ultimate limit state)")
    st.caption("One row per load combination. You can paste straight from Excel: "
               "click the first cell and press Ctrl+V. Axial force: compression "
               "positive. M major = about the x axis (M > 0: compression at the top), "
               "M minor = about the y axis. Shear signs are ignored.")
    default_combos = pd.DataFrame({
        "Combo": ["ULS 1"], "N (kN)": [0.0], "V major (kN)": [0.0],
        "V minor (kN)": [0.0], "M major (kNm)": [150.0], "M minor (kNm)": [0.0]})
    combos_df = st.data_editor(
        default_combos, num_rows="dynamic", key=f"combos_{element}",
        column_config={c: st.column_config.NumberColumn(c, format="%.1f")
                       for c in COMBO_COLS})
    cdf = combos_df.dropna(how="all").copy()
    for col in COMBO_COLS:
        cdf[col] = pd.to_numeric(cdf[col], errors="coerce").fillna(0.0)
    cdf["Combo"] = cdf["Combo"].fillna("").astype(str)
    combos = [dict(name=(r[0] or f"Combo {i + 1}"), n=float(r[1]), vy=float(r[2]),
                   vx=float(r[3]), mx=float(r[4]), my=float(r[5]))
              for i, r in enumerate(cdf[["Combo"] + COMBO_COLS]
                                    .itertuples(index=False, name=None))]
    bb_points = st.select_slider("Biaxial diagram points (used when a combination "
                                 "has a minor-axis moment)",
                                 options=[16, 24, 32, 48], value=24, key="bbp")

    st.subheader("Shear links")
    c1, c2 = st.columns(2)
    c1.metric("Link diameter (from the geometry tab)", f"{g['link']:.0f} mm")
    s_link = c2.number_input("Link spacing s (mm)", 25.0, 1000.0, 200.0, key="slink")
    c1, c2 = st.columns(2)
    legs_y = c1.number_input("Number of legs resisting the major shear (along y)",
                             0, 20, 2, key="legsy")
    legs_x = c2.number_input("Number of legs resisting the minor shear (along x)",
                             0, 20, 2, key="legsx")
    c1, c2 = st.columns(2)
    cot_mode_txt = c1.radio("Strut angle", ["Optimum (VRd,s = VRd,max)", "Manual"],
                            key="cotmode")
    cot_manual = c2.number_input("cot θ if manual (1.0 to 2.5)", 1.0, 2.5, 2.5,
                                 key="cotm")
    with st.expander("Shear dimensions (automatic unless you enter a value)"):
        if lay["ok"]:
            da = auto_shear_dims(lay, True)
            st.caption(f"Automatic values (sagging): bw,y = {da['bw_y']:.0f} mm, "
                       f"d,y = {da['d_y']:.0f} mm, bw,x = {da['bw_x']:.0f} mm, "
                       f"d,x = {da['d_x']:.0f} mm. bw = smallest width between the "
                       "faces; d = depth to the centroid of the tension half of the "
                       "bars, so it follows the sign of each combination's moment. "
                       "They are approximate for T, L, circular and custom sections. "
                       "Enter 0 to keep the automatic value.")
        c1, c2 = st.columns(2)
        bw_y = c1.number_input("bw for the major shear (mm), 0 = automatic",
                               0.0, 50000.0, 0.0, key="bwy")
        d_y = c1.number_input("d for the major shear (mm), 0 = automatic",
                              0.0, 50000.0, 0.0, key="dy")
        bw_x = c2.number_input("bw for the minor shear (mm), 0 = automatic",
                               0.0, 50000.0, 0.0, key="bwx")
        d_x = c2.number_input("d for the minor shear (mm), 0 = automatic",
                              0.0, 50000.0, 0.0, key="dx")

    act_uls = dict(
        combos=combos, bb_points=bb_points, s_link=s_link, legs_y=legs_y,
        legs_x=legs_x, link_dia=g["link"],
        cot_mode="optimum" if cot_mode_txt.startswith("Optimum") else "manual",
        cot_manual=cot_manual, bw_y=bw_y, d_y=d_y, bw_x=bw_x, d_x=d_x)
    uls_hash = digest(g, mat, act_uls)

    run_uls_btn = st.button("▶ Run ULS check", type="primary",
                            disabled=(ctx is None or not combos), key="uls_btn")
    uls_box = st.empty()
    if run_uls_btn and ctx is not None and combos:
        with st.spinner("Running the ULS check (biaxial combinations take longer)..."):
            R_new = run_uls(ctx, lay, g, act_uls)
            R_new["hash"] = uls_hash
            st.session_state["R_uls"] = R_new
    R_uls = st.session_state.get("R_uls")
    status_line(uls_box, R_uls, uls_hash, "ULS check")
    if R_uls is not None:
        if R_uls.get("time"):
            with st.expander("Calculation time (seconds)"):
                st.write({k_: round(v, 1) for k_, v in R_uls["time"].items()})
        views.uls_view(R_uls, ctx, R_uls["hash"] == uls_hash)

# ============================================================ 5 SLS CHECK
has_sls = element in ("Beam", "Generic section")
sls_hash = None
with t_sls:
    if not has_sls:
        st.info("The SLS check is not provided for columns and walls.")
    else:
        st.subheader("Design actions (serviceability limit state)")
        sls_dir = st.radio("Bending direction",
                           ["Sagging (compression at top)",
                            "Hogging (compression at bottom)"],
                           horizontal=True, key="sdir")
        st.caption("Service combination (stress limits)")
        c1, c2 = st.columns(2)
        n_ser = c1.number_input("N_ser (kN, compression +ve)", -5000.0, 50000.0, 0.0, key="nser")
        m_ser = c2.number_input("M_ser (kNm, enter as positive)", 0.0, 50000.0, 100.0, key="mser")
        st.caption("Quasi-permanent combination (crack width)")
        c1, c2 = st.columns(2)
        n_qp = c1.number_input("N_qp (kN, compression +ve)", -5000.0, 50000.0, 0.0, key="nqp")
        m_qp = c2.number_input("M_qp (kNm, enter as positive)", 0.0, 50000.0, 80.0, key="mqp")

        act_sls = dict(sagging=sls_dir.startswith("Sagging"), n_ser=n_ser,
                       m_ser=m_ser, n_qp=n_qp, m_qp=m_qp)
        sls_hash = digest(g, mat, act_sls)

        run_sls_btn = st.button("▶ Run SLS check", type="primary",
                                disabled=ctx is None, key="sls_btn")
        sls_box = st.empty()
        if run_sls_btn and ctx is not None:
            with st.spinner("Running the SLS check (this can take a minute)..."):
                S_new = run_sls(ctx, lay, g, act_sls)
                S_new["hash"] = sls_hash
                st.session_state["R_sls"] = S_new
        R_sls = st.session_state.get("R_sls")
        status_line(sls_box, R_sls, sls_hash, "SLS check")
        if R_sls is not None:
            if R_sls.get("time"):
                with st.expander("Calculation time (seconds)"):
                    st.write({k_: round(v, 1) for k_, v in R_sls["time"].items()})
                    for line in R_sls.get("mk_log", []):
                        st.caption(line)
            views.sls_view(R_sls)

# ====================================================== 3 SECTION RESULTS
with t_res:
    S_res = st.session_state.get("R_sls") if has_sls else None
    views.section_results(
        ctx, G, S_res,
        bool(S_res) and S_res["hash"] == sls_hash, has_sls)

# ======================================================== 6 BEAM (phase 3)
with t_beam:
    st.info("Long-term beam deflection (simply supported or cantilever) comes in step 3.")
