import math
import time

import pandas as pd
import streamlit as st

from core.analysis import compute_mn, ensure_combo_images, max_spacing_73n
from core.geometry import fmt
from core.plots import show_png


def _verdict(ok, text):
    (st.success if ok else st.error)(text)


# ================================================== 3 SECTION RESULTS
def _eff_block(S):
    act = S["act"]
    dir_txt, comp_face = (("sagging", "top") if S["sagging"] else ("hogging", "bottom"))
    hom, cr, eff = S.get("hom"), S.get("cr"), S.get("eff")
    st.caption(f"N_ser = {act['n_ser']:,.1f} kN (compression +ve), "
               f"M_ser = {act['m_ser']:,.1f} kNm, {dir_txt} bending.")
    if eff is None:
        st.warning(f"Not available: {S['err'].get('eff')}")
        return
    if eff["kind"] == "zero":
        st.info("M_ser is zero, so no secant stiffness M/κ exists.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("I effective (×10⁹ mm⁴)", f"{eff['i_eff'] / 1e9:.3f}")
    c2.metric("EI effective (kNm²)", f"{eff['ei_eff'] * 1e-9:,.0f}")
    if hom:
        c3.metric("Ieff / Ihom", f"{eff['i_eff'] / hom['i_hom']:.3f}")
    c1, c2 = st.columns(2)
    c1.metric("Concrete tension, uncracked analysis (MPa)", f"{eff['t_conc']:.3f}")
    c2.metric("fct,eff (MPa)", f"{S['fct_eff']:.3f}")
    if eff["kind"] == "uncracked":
        st.success("Uncracked at the service load: Ieff = Ihom.")
        return
    c1, c2 = st.columns(2)
    c1.metric("Curvature (1/mm)", f"{eff['kappa']:.4e}")
    if eff.get("na_depth") is not None:
        c2.metric(f"Zero-strain axis from {comp_face} face (mm)",
                  f"{eff['na_depth']:,.1f}")
    if eff.get("na_msg"):
        st.caption(eff["na_msg"])
    if eff.get("na_outside"):
        st.caption("The zero-strain axis lies outside the section (the axial load "
                   "keeps the whole section at one strain sign).")
    st.success("Cracked at the service load: Ieff is the secant stiffness "
               "M_ser/κ from the no-tension moment-curvature analysis, including N_ser.")
    if cr and hom:
        st.caption(f"For comparison: Icr/Ihom = {cr['i_cr'] / hom['i_hom']:.3f} "
                   f"(pure bending) and Ieff/Ihom = "
                   f"{eff['i_eff'] / hom['i_hom']:.3f} at the service load.")


def section_results(ctx, G, S, s_fresh, has_sls=True):
    if ctx is None or G is None:
        st.info("Complete the geometry tab without errors to see the section "
                "properties here.")
        return
    es, e = ctx["es"], ctx["e_service"]
    st.markdown(f"**{ctx['basis_txt']} stiffness basis**: Ec = {e / 1e3:.1f} GPa, "
                f"n = Es/Ec = {es / e:.1f}. These results update automatically.")

    st.subheader("a) Homogeneous uncracked section")
    hom = G.get("hom")
    if hom is None:
        st.warning(f"Not available: {G['err'].get('hom')}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("A homogeneous (mm²)", f"{hom['a_hom']:,.0f}")
        c2.metric("Centroid above bottom face (mm)", f"{hom['y_hom']:,.1f}")
        c3.metric("I homogeneous (×10⁹ mm⁴)", f"{hom['i_hom'] / 1e9:.3f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("EI homogeneous (kNm²)", f"{hom['ei_hom'] * 1e-9:,.0f}")
        c2.metric("Concrete area Ac (mm²)", f"{hom['a_c']:,.0f}")
        c3.metric("Steel area As (mm²)", f"{hom['a_s']:,.0f}")
        st.caption("Reinforcement transformed to concrete with n = Es/Ec; the whole "
                   "concrete section is active.")

    st.subheader("b) Cracked section, bending only")
    cols = st.columns(2)
    for col, key, name, face in ((cols[0], "sag", "Sagging", "top"),
                                 (cols[1], "hog", "Hogging", "bottom")):
        with col:
            st.markdown(f"**{name}** (compression at {face})")
            cr = G.get(key)
            if cr is None:
                st.warning(f"Not available: {G['err'].get(key)}")
                continue
            st.metric("I cracked (×10⁹ mm⁴)", f"{cr['i_cr'] / 1e9:.3f}")
            st.metric("EI cracked (kNm²)", f"{cr['ei_cr'] * 1e-9:,.0f}")
            if hom:
                st.metric("Icr / Ihom", f"{cr['i_cr'] / hom['i_hom']:.3f}")
            st.metric("M_cr (kNm)", f"{cr['m_cr']:,.1f}")
            st.metric(f"Neutral axis from the {face} face (mm)", f"{cr['d_nc']:,.1f}")
    st.caption("Fully cracked, pure bending (no axial force); concrete in tension "
               "is ignored. M_cr uses fctm.")

    if not has_sls:
        return

    st.subheader("c) Section at the service forces")
    if S is None:
        st.info("Run the SLS check (tab 5) to see the stiffness at the service forces.")
    else:
        if not s_fresh:
            st.warning("These values come from an earlier SLS run; inputs have "
                       "changed since then.")
        _eff_block(S)


# ======================================================== 4 ULS CHECK
def _shear_block(title, S):
    st.markdown(f"**{title}**")
    if S is None:
        st.info("Not calculated.")
        return
    if "error" in S:
        st.error(S["error"])
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("V_Ed (kN)", f"{S['v_ed']:,.1f}")
    c2.metric("V_Rd (kN)", f"{S['vrd']:,.1f}")
    c3.metric("V_Ed / V_Rd", f"{S['util']:.2f}")
    _verdict(S["util"] <= 1.0, "Shear resistance satisfied."
             if S["util"] <= 1.0 else "Shear resistance NOT satisfied.")
    st.caption(f"Governing: {S['governs']}.")
    c1, c2, c3 = st.columns(3)
    c1.metric("V_Rd,c, no links (kN)", f"{S['vrdc']:,.1f}")
    if S["has_links"]:
        c2.metric("V_Rd,s, links (kN)", f"{S['vrd_s']:,.1f}")
        c3.metric("V_Rd,max, struts (kN)", f"{S['vrd_max']:,.1f}")
    if S["links_not_needed"]:
        st.caption("V_Ed ≤ V_Rd,c: no calculated shear reinforcement is needed "
                   "(minimum links still apply, EC2 9.2.2).")
    else:
        st.caption("V_Ed > V_Rd,c: shear reinforcement is required.")
    for text, ok in S["checks"]:
        st.markdown(("✅ " if ok else "⚠️ ") + text)
    with st.expander("Calculation details"):
        rows = [
            ("bw (mm)", fmt(S["bw"], 0)), ("d (mm)", fmt(S["d"], 0)),
            ("z = 0.9d (mm)", fmt(S["z"], 0)),
            ("Asl, tension steel (mm²)", fmt(S["asl"], 0)),
            ("ρl = Asl/(bw·d) ≤ 0.02", f"{S['rho_l']:.4f}"),
            ("k = 1 + √(200/d) ≤ 2", f"{S['k']:.3f}"),
            ("CRd,c = 0.18/γc", f"{S['crdc']:.3f}"),
            ("vmin (MPa)", f"{S['v_min']:.3f}"),
            ("σcp = N_Ed/Ac (MPa)", f"{S['sig_cp']:.2f}"),
            ("Asw (mm²)", fmt(S["asw"], 0)),
            ("fywd (MPa)", fmt(S["fywd"], 0)),
            ("ν1", f"{S['nu1']:.3f}"), ("αcw", f"{S['acw']:.3f}"),
        ]
        if S["cot"]:
            rows += [("cot θ", f"{S['cot']:.2f}"), ("θ (deg)", f"{S['theta']:.1f}")]
        st.dataframe(pd.DataFrame(rows, columns=["Quantity", "Value"]),
                     hide_index=True)

def _fmt_u(u):
    if u is None:
        return "-"
    return "inf" if math.isinf(u) else f"{u:.2f}"


def _combo_utils(c):
    b, sh = c.get("bend"), c.get("shear")
    ub = b.get("util") if b else None
    uy = sh["y"].get("util") if sh else None
    ux = sh["x"].get("util") if sh else None
    vals = [u for u in (ub, uy, ux) if u is not None]
    return ub, uy, ux, (max(vals) if vals else None)


def _combo_detail(c, R, ctx, fresh, idx):
    act, mat = R["act"], R["mat"]
    st.caption(f"N = {c['n']:,.1f} kN, M_x = {c['mx']:,.1f} kNm, M_y = {c['my']:,.1f} kNm, "
               f"V_y = {c['vy']:,.1f} kN, V_x = {c['vx']:,.1f} kN.")
    st.subheader("a) Bending")
    b = c.get("bend")
    if b is None:
        st.error(f"The bending check failed: {c.get('err_bend')}")
    elif b["kind"] == "uniaxial":
        c1, c2, c3 = st.columns(3)
        c1.metric("M_Rd (kNm)", f"{b['m_rd']:,.1f}")
        c2.metric("M_Ed,x (kNm)", f"{b['m_ed']:,.1f}")
        c3.metric("M_Ed / M_Rd", _fmt_u(b["util"]))
        if b["util"] is not None:
            ok = b["util"] <= 1.0
            _verdict(ok, "Bending resistance satisfied." if ok
                     else "Bending resistance NOT satisfied.")
        face = "top" if b["sagging"] else "bottom"
        st.caption(f"{'Sagging' if b['sagging'] else 'Hogging'} bending about the x "
                   f"axis, compression at the {face}. Concrete: {mat['ult_model']}; "
                   "steel: elastic-plastic.")
        if b.get("d_n") is not None:
            st.metric(f"Neutral axis depth from the {face} face (mm)", f"{b['d_n']:,.1f}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("M_Ed,x (kNm)", f"{b['mx_ed']:,.1f}")
        c2.metric("M_Ed,y (kNm)", f"{b['my_ed']:,.1f}")
        c3.metric("Resultant M_Ed (kNm)", f"{b['m_res']:,.1f}")
        if b["cap"] is None:
            st.error("N_Ed is outside the capacity of the section, so no biaxial "
                     "capacity exists for this axial load.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Capacity along the same direction (kNm)", f"{b['cap']:,.1f}")
            c2.metric("Demand / capacity", _fmt_u(b["util"]))
            ok = b["util"] <= 1.0 if b["inside"] is None else bool(b["inside"])
            _verdict(ok, "Biaxial bending resistance satisfied: the design point is "
                     "inside the diagram." if ok else
                     "Biaxial bending resistance NOT satisfied: the design point is "
                     "outside the diagram.")
        st.caption("Capacity is the boundary of the biaxial diagram at N_Ed, measured "
                   "along the applied moment direction. M_x > 0: compression at the "
                   "top. " + (f"M_y > 0: compression at the {b['my_side']} face."
                              if b.get("my_side") else
                              "M_y sign: could not be determined automatically."))

    if b is not None:
        if ctx is not None and fresh:
            ensure_combo_images(ctx, c)
        if "uls_bars" in c["img"]:
            if "biax" in c["img"]:
                show_png(c["img"]["biax"])
            show_png(c["img"]["uls_conc"])
            show_png(c["img"]["uls_bars"], width=1)
            st.dataframe(b["df"].round(1), hide_index=True)
            st.caption("Stresses are those of the section at its capacity for the "
                       "applied axial force, not at the applied moment. Compression "
                       "is positive.")
        if "ss_conc" in R["img"]:
            with st.expander("Stress-strain curves used"):
                show_png(R["img"]["ss_conc"])
                show_png(R["img"]["ss_steel"])
        st.markdown("**M-N interaction diagram (about the x axis)**")
        both = st.checkbox("Also compute the opposite bending direction",
                           key=f"mn_both_{idx}")
        if st.button("Compute M-N diagram", key=f"mn_btn_{idx}",
                     disabled=(not fresh or ctx is None)):
            with st.spinner("Computing the M-N diagram..."):
                compute_mn(ctx, c, both)
        for e in c.get("mn_err", []):
            st.warning(f"The M-N curve could not be calculated ({e}). If this keeps "
                       "happening, choose Python 3.12 in the app's advanced settings.")
        if "mn" in c["img"]:
            show_png(c["img"]["mn"])

    st.subheader("b) Shear")
    sh = c.get("shear")
    if sh is None:
        st.error(f"The shear check failed: {c.get('err_shear')}")
    else:
        d = sh["dims"]
        st.caption(f"Links: Ø{act['link_dia']:.0f} at {act['s_link']:.0f} mm; "
                   f"{act['legs_y']} legs along y, {act['legs_x']} legs along x. "
                   f"fywk = {mat['fywk']:.0f} MPa, αcc (shear) = {mat['acc_shear']}. "
                   f"Strut angle: {'optimum' if act['cot_mode'] == 'optimum' else 'manual'}, "
                   "cot θ limited to 1.0 to 2.5. z = 0.9d. Each direction is checked "
                   "separately.")
        c1, c2 = st.columns(2)
        with c1:
            _shear_block(f"Shear along y, major axis (bw = {d['bw_y']:.0f} mm, "
                         f"d = {d['d_y']:.0f} mm)", sh["y"])
        with c2:
            _shear_block(f"Shear along x, minor axis (bw = {d['bw_x']:.0f} mm, "
                         f"d = {d['d_x']:.0f} mm)", sh["x"])


def uls_view(R, ctx, fresh):
    combos = R["combos"]
    st.subheader("Summary of the load combinations")
    rows, overall = [], []
    for c in combos:
        ub, uy, ux, u = _combo_utils(c)
        overall.append(u if u is not None else 0.0)
        rows.append({"Combination": c["name"], "N (kN)": f"{c['n']:,.0f}",
                     "M major (kNm)": f"{c['mx']:,.0f}", "M minor (kNm)": f"{c['my']:,.0f}",
                     "Bending": _fmt_u(ub), "Shear major": _fmt_u(uy),
                     "Shear minor": _fmt_u(ux), "Max": _fmt_u(u),
                     "Status": "✅ OK" if (u is not None and u <= 1.0) else "❌ NOT OK"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    gov = max(range(len(combos)), key=lambda i: overall[i])
    ok_all = all(u <= 1.0 for u in overall)
    _verdict(ok_all, "All combinations satisfy the ULS checks." if ok_all else
             f"The ULS checks are NOT satisfied. Governing combination: "
             f"{combos[gov]['name']} (utilisation {_fmt_u(overall[gov])}).")
    names = [c["name"] for c in combos]
    sel = st.selectbox("Show the details of", names, index=gov, key="uls_sel")
    idx = names.index(sel)
    st.markdown(f"### Combination {sel}")
    _combo_detail(combos[idx], R, ctx, fresh, idx)

# =========================================================== 5 SLS CHECK
def sls_view(R):
    mat, act, fck = R["mat"], R["act"], R["fck"]
    fyk = mat["fyk"]
    dir_txt = "sagging" if R["sagging"] else "hogging"

    st.subheader("a) Concrete and steel stresses (EC2 7.2)")
    s = R.get("svc")
    if s is None:
        st.error(f"The service analysis failed: {R['err'].get('svc')}")
    else:
        if s.get("warn"):
            st.warning(s["warn"])
        st.markdown(f"**{R['basis_txt']} basis**: Ec = {R['e_service'] / 1e3:.1f} GPa "
                    f"(n = {R['es'] / R['e_service']:.1f}), M_ser = {act['m_ser']:,.0f} kNm, "
                    f"N_ser = {act['n_ser']:,.0f} kN, {dir_txt}.")
        if s["t_unc"] is not None:
            st.markdown(f"**Section state at service: {s['state']}.** Largest concrete "
                        f"tension in the uncracked analysis = {s['t_unc']:.2f} MPa, "
                        f"fct,eff = {R['fct_eff']:.2f} MPa.")
        else:
            st.markdown(f"**Section state at service: {s['state']}.**")
        lim_c, lim_s = 0.6 * fck, 0.8 * fyk
        c1, c2, c3 = st.columns(3)
        c1.metric("Concrete compression (MPa)", f"{s['sig_c']:.1f}")
        c2.metric("Limit 0.6·fck (MPa)", f"{lim_c:.1f}")
        c3.metric("σc / (0.6·fck)", f"{s['sig_c'] / lim_c:.2f}")
        _verdict(s["sig_c"] <= lim_c, "Concrete compression limit (EC2 7.2(2)) "
                 + ("satisfied." if s["sig_c"] <= lim_c else "EXCEEDED."))
        st.caption(f"For the quasi-permanent combination EC2 7.2(3) uses "
                   f"0.45·fck = {0.45 * fck:.1f} MPa "
                   f"(ratio {s['sig_c'] / (0.45 * fck):.2f}).")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Max steel stress, bar {s['worst_bar']} (MPa)", f"{s['sig_s']:.0f}")
        c2.metric("Limit 0.8·fyk (MPa)", f"{lim_s:.0f}")
        c3.metric("σs / (0.8·fyk)", f"{s['sig_s'] / lim_s:.2f}")
        _verdict(s["sig_s"] <= lim_s, "Steel stress limit (EC2 7.2(5)) "
                 + ("satisfied." if s["sig_s"] <= lim_s else "EXCEEDED."))
        st.metric("Max concrete tension (MPa)", f"{s['t_conc']:.2f}")
        show_png(R["img"]["svc_conc"])
        show_png(R["img"]["svc_bars"], width=1)
        st.dataframe(s["df"].round(1), hide_index=True)
        st.caption(f"Method: {s['method']}. Compression is positive, so bars in "
                   "tension show negative stress. M and N are taken about the gross "
                   "centroid of the section.")

    st.subheader("b) Crack width (EN 1992-1-1 7.3.4)")
    c = R.get("crack")
    if c is None:
        st.error(f"The crack width analysis failed: {R['err'].get('crack')}")
        return
    if c["status"] == "circular":
        st.info("The crack width check is not provided for circular sections.")
        return
    st.markdown(f"Quasi-permanent combination: N_qp = {act['n_qp']:,.0f} kN, "
                f"M_qp = {act['m_qp']:,.0f} kNm.")
    if c["status"] == "uncracked":
        st.success(f"Uncracked under this combination (concrete tension "
                   f"{c['t_q']:.2f} MPa ≤ fct,eff = {R['fct_eff']:.2f} MPa): "
                   "no crack width check is needed.")
        return
    if c["status"] == "error":
        st.error(c["msg"])
        return
    wmax, rws = mat["wmax"], c["rows"]
    wk_gov = max(r["wk"] for r in rws)
    c1, c2, c3 = st.columns(3)
    c1.metric("Governing crack width wk (mm)", f"{wk_gov:.3f}")
    c2.metric("Limit wmax (mm)", f"{wmax:.2f}")
    c3.metric("wk / wmax", f"{wk_gov / wmax:.2f}")
    _verdict(wk_gov <= wmax, "Crack width limit satisfied." if wk_gov <= wmax
             else "Crack width limit EXCEEDED.")
    st.markdown(f"**{'Long' if c['long'] else 'Short'}-term check**: "
                f"Ec = {c['e_crack'] / 1e3:.1f} GPa, αe = Es/Ec = {c['alpha_e']:.2f}, "
                f"kt = {c['kt']}.")
    tab_rows = [{
        "Layer y (mm)": fmt(r["y"]), "bars": str(r["n"]),
        "σs (MPa)": fmt(r["sigma"], 0), "cover c (mm)": fmt(r["c"]),
        "φ (mm)": fmt(r["phi"]),
        "spacing (mm)": "-" if math.isinf(r["spacing"]) else fmt(r["spacing"]),
        "sr,max (mm)": fmt(r["sr"]) + (" (1.3(h-x))" if r["wide"] else ""),
        "εsm-εcm (×10⁻³)": f"{r['diff'] * 1e3:.3f}",
        "wk (mm)": f"{r['wk']:.3f}",
    } for r in rws]
    st.dataframe(pd.DataFrame(tab_rows), hide_index=True)
    st.caption(
        f"x = {c['x']:.0f} mm, h − d = {c['d_t']:.0f} mm, hc,eff = {c['hc_eff']:.0f} mm, "
        f"Ac,eff = {c['ac_eff']:,.0f} mm², ρp,eff = {c['rho']:.4f}, "
        f"φeq = {c['phi_eq']:.1f} mm, k2 = {c['k2']:.2f}, "
        f"fct,eff = {R['p']['fctm']:.2f} MPa, k1 = {mat['k1']}, k3 = {mat['k3']}, "
        f"k4 = {mat['k4']}. σs and αe use the same Ec. Layers are listed from the "
        "tension face inwards.")
    ext = rws[0]
    lim73 = max_spacing_73n(ext["sigma"], wmax)
    if math.isinf(ext["spacing"]):
        st.caption("EC2 7.3.3 (Table 7.3N): not applicable, the outer tension layer "
                   "has a single bar.")
    elif lim73 is None:
        st.caption(f"EC2 7.3.3 (Table 7.3N): σs = {ext['sigma']:.0f} MPa and "
                   f"wmax = {wmax} mm are outside the table.")
    else:
        ok73 = ext["spacing"] <= lim73 + 1e-6
        st.caption(f"EC2 7.3.3 (Table 7.3N), outer tension layer: spacing "
                   f"{ext['spacing']:.0f} mm vs maximum {lim73} mm at "
                   f"σs = {ext['sigma']:.0f} MPa → "
                   + ("✅ OK" if ok73 else "⚠️ exceeds the table value")
                   + " (table assumes c = 25 mm and fct,eff = 2.9 MPa).")
    if "crack" in R["img"]:
        show_png(R["img"]["crack"])
