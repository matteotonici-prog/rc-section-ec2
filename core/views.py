import math

import pandas as pd
import streamlit as st

from core.analysis import max_spacing_73n
from core.geometry import fmt
from core.plots import show_png


def _dir(R):
    return ("sagging", "top") if R["sagging"] else ("hogging", "bottom")


def _verdict(ok, text):
    (st.success if ok else st.error)(text)


# ===================================================== 4 SECTION RESULTS
def geometry_results(R):
    p = R["p"]
    dir_txt, comp_face = _dir(R)
    if "section" in R["img"]:
        show_png(R["img"]["section"])
    st.caption(f"fcd = {p['fcd']:.1f} MPa, fyd = {R['fyd']:.0f} MPa, "
               f"λ = {p['lam']:.3f}, η = {p['eta']:.3f}, εcu3 = {p['ecu']:.4f}, "
               f"Ecm = {p['ecm'] / 1e3:.1f} GPa")
    st.markdown(f"**{R['basis_txt']} stiffness basis**: Ec = "
                f"{R['e_service'] / 1e3:.1f} GPa, n = Es/Ec = "
                f"{R['es'] / R['e_service']:.1f}")

    # a) homogeneous uncracked
    st.subheader("a) Homogeneous uncracked section")
    hom = R.get("hom")
    if hom is None:
        st.warning(f"Not available: {R['err'].get('hom')}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("A homogeneous (mm²)", f"{hom['a_hom']:,.0f}")
        c2.metric("Centroid above bottom face (mm)", f"{hom['y_hom']:,.1f}")
        c3.metric("I homogeneous (×10⁹ mm⁴)", f"{hom['i_hom'] / 1e9:.3f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("EI homogeneous (kNm²)", f"{hom['ei_hom'] * 1e-9:,.0f}")
        c2.metric("Concrete area Ac (mm²)", f"{hom['a_c']:,.0f}")
        c3.metric("Steel area As (mm²)", f"{hom['a_s']:,.0f}")
        st.caption("Reinforcement transformed to concrete with n = Es/Ec; the "
                   "whole concrete section is active.")

    # b) cracked, bending only
    st.subheader("b) Cracked section, bending only")
    cr = R.get("cr")
    if cr is None:
        st.warning(f"Not available: {R['err'].get('cr')}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("I cracked (×10⁹ mm⁴)", f"{cr['i_cr'] / 1e9:.3f}")
        c2.metric("EI cracked (kNm²)", f"{cr['ei_cr'] * 1e-9:,.0f}")
        if hom:
            c3.metric("Icr / Ihom", f"{cr['i_cr'] / hom['i_hom']:.3f}")
        c1, c2 = st.columns(2)
        c1.metric("M_cr (kNm)", f"{cr['m_cr']:,.1f}")
        c2.metric(f"Neutral axis from {comp_face} face (mm)", f"{cr['d_nc']:,.1f}")
        st.caption(f"Fully cracked, {dir_txt} pure bending (no axial force); "
                   "concrete in tension is ignored. M_cr uses fctm.")

    # c) cracked with the actual service forces
    st.subheader("c) Section at the service forces")
    inp = R["inp"]
    st.caption(f"N_ser = {inp['n_ser']:,.1f} kN (compression +ve), "
               f"M_ser = {inp['m_ser']:,.1f} kNm, {dir_txt} bending.")
    eff = R.get("eff")
    if eff is None:
        st.warning(f"Not available: {R['err'].get('eff')}")
    elif eff["kind"] == "zero":
        st.info("M_ser is zero, so no secant stiffness M/κ exists.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("I effective (×10⁹ mm⁴)", f"{eff['i_eff'] / 1e9:.3f}")
        c2.metric("EI effective (kNm²)", f"{eff['ei_eff'] * 1e-9:,.0f}")
        if hom:
            c3.metric("Ieff / Ihom", f"{eff['i_eff'] / hom['i_hom']:.3f}")
        c1, c2 = st.columns(2)
        c1.metric("Concrete tension, uncracked analysis (MPa)", f"{eff['t_conc']:.3f}")
        c2.metric("fct,eff (MPa)", f"{R['fct_eff']:.3f}")
        if eff["kind"] == "uncracked":
            st.success("Uncracked at the service load: Ieff = Ihom.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Curvature (1/mm)", f"{eff['kappa']:.4e}")
            if eff.get("na_depth") is not None:
                c2.metric(f"Zero-strain axis from {comp_face} face (mm)",
                          f"{eff['na_depth']:,.1f}")
            if eff.get("na_msg"):
                st.caption(eff["na_msg"])
            if eff.get("na_outside"):
                st.caption("The zero-strain axis lies outside the section (axial "
                           "load keeps the whole section at one strain sign).")
            st.success("Cracked at the service load: Ieff is the secant stiffness "
                       "M_ser/κ from the no-tension moment-curvature analysis, "
                       "including N_ser.")
            if cr and hom:
                st.caption(f"For comparison: Icr/Ihom = {cr['i_cr'] / hom['i_hom']:.3f} "
                           f"(pure bending) and Ieff/Ihom = "
                           f"{eff['i_eff'] / hom['i_hom']:.3f} at the service load.")


# ========================================================= 5 ULS BENDING
def uls_bending(R):
    inp = R["inp"]
    dir_txt, comp_face = _dir(R)
    u = R.get("uls")
    st.subheader("a) Resisting moment")
    if u is None:
        st.error(f"The ultimate capacity could not be calculated: {R['err'].get('uls')}")
    else:
        m_rd, m_ed = u["m_rd"], inp["m_ed"]
        c1, c2, c3 = st.columns(3)
        c1.metric("M_Rd (kNm)", f"{m_rd:,.1f}")
        c2.metric("M_Ed (kNm)", f"{m_ed:,.1f}")
        c3.metric("Utilisation M_Ed / M_Rd", f"{m_ed / m_rd:.2f}" if m_rd > 0 else "n/a")
        if m_rd > 0:
            _verdict(m_ed <= m_rd, "Bending resistance satisfied." if m_ed <= m_rd
                     else "Bending resistance NOT satisfied.")
        st.caption(f"{dir_txt.capitalize()} bending, compression at {comp_face}, "
                   f"N_Ed = {inp['n_ed']:,.0f} kN. Concrete: {inp['ult_model']}; "
                   "steel: elastic-plastic.")

    st.subheader("b) M-N interaction diagram")
    for e in R.get("mn_err", []):
        st.warning(f"The M-N curve could not be calculated ({e}). If this keeps "
                   "happening, choose Python 3.12 in the app's advanced settings.")
    if "mn" in R["img"]:
        show_png(R["img"]["mn"])

    if u is not None:
        st.subheader("c) Neutral axis at the ULS")
        if u["d_n"] is not None:
            st.metric(f"Neutral axis depth from the {comp_face} face (mm)",
                      f"{u['d_n']:,.1f}")
        else:
            st.caption("The neutral axis depth is not available from the analysis.")

        st.subheader("d) Stresses at the ULS")
        show_png(R["img"]["uls_conc"])
        show_png(R["img"]["uls_bars"], width=1)
        st.dataframe(u["df"].round(1), hide_index=True)
        st.caption("The stresses are those of the section at its capacity M_Rd for "
                   "the applied axial force, not at M_Ed. Compression is positive.")
        if "ss_conc" in R["img"]:
            with st.expander("Stress-strain curves used"):
                show_png(R["img"]["ss_conc"])
                show_png(R["img"]["ss_steel"])


# ================================================================= 7 SLS
def sls(R):
    inp, fck = R["inp"], R["fck"]
    fyk = inp["fyk"]
    dir_txt, _ = _dir(R)

    st.subheader("a) Concrete and steel stresses (EC2 7.2)")
    s = R.get("svc")
    if s is None:
        st.error(f"The service analysis failed: {R['err'].get('svc')}")
    else:
        if s.get("warn"):
            st.warning(s["warn"])
        st.markdown(f"**{R['basis_txt']} basis**: Ec = {R['e_service'] / 1e3:.1f} GPa "
                    f"(n = {R['es'] / R['e_service']:.1f}), M_ser = {inp['m_ser']:,.0f} kNm, "
                    f"N_ser = {inp['n_ser']:,.0f} kN, {dir_txt}.")
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

    # b) crack width
    st.subheader("b) Crack width (EN 1992-1-1 7.3.4)")
    c = R.get("crack")
    if c is None:
        st.error(f"The crack width analysis failed: {R['err'].get('crack')}")
        return
    if c["status"] == "circular":
        st.info("The crack width check is not provided for circular sections.")
        return
    st.markdown(f"Quasi-permanent combination: N_qp = {inp['n_qp']:,.0f} kN, "
                f"M_qp = {inp['m_qp']:,.0f} kNm.")
    if c["status"] == "uncracked":
        st.success(f"Uncracked under this combination (concrete tension "
                   f"{c['t_q']:.2f} MPa ≤ fct,eff = {R['fct_eff']:.2f} MPa): "
                   "no crack width check is needed.")
        return
    if c["status"] == "error":
        st.error(c["msg"])
        return

    wmax = inp["wmax"]
    rws = c["rows"]
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
        f"fct,eff = {R['p']['fctm']:.2f} MPa, k1 = {inp['k1']}, k3 = {inp['k3']}, "
        f"k4 = {inp['k4']}. σs and αe use the same Ec. Layers are listed from the "
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
