import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from concreteproperties import Concrete, ConcreteSection, SteelBar
from concreteproperties.stress_strain_profile import (
    ConcreteLinear, RectangularStressBlock, SteelElasticPlastic,
)
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library import (
    concrete_rectangular_section, concrete_tee_section, concrete_circular_section,
)
from sectionproperties.pre.library.concrete_sections import add_bar

CLASSES = {"C20/25": 20, "C25/30": 25, "C30/37": 30, "C32/40": 32, "C35/45": 35,
           "C40/50": 40, "C45/55": 45, "C50/60": 50, "C55/67": 55, "C60/75": 60,
           "C70/85": 70, "C80/95": 80, "C90/105": 90}


def ec2_concrete(fck, acc, gamma_c):
    """EN 1992-1-1 Table 3.1 / cl. 3.1.7 parameters (MPa)."""
    fcm = fck + 8
    fcd = acc * fck / gamma_c
    if fck <= 50:
        lam, eta, ecu3 = 0.8, 1.0, 0.0035
        fctm = 0.30 * fck ** (2 / 3)
    else:
        lam = 0.8 - (fck - 50) / 400
        eta = 1.0 - (fck - 50) / 200
        ecu3 = (2.6 + 35 * ((90 - fck) / 100) ** 4) / 1000
        fctm = 2.12 * math.log(1 + fcm / 10)
    ecm = 22000 * (fcm / 10) ** 0.3
    return dict(fcd=fcd, lam=lam, eta=eta, ecu3=ecu3, fctm=fctm, ecm=ecm)


def area(dia):
    return math.pi * dia**2 / 4


st.title("RC section check - Eurocode 2")

# Outside the form so the inputs below update as soon as the type changes
shape = st.selectbox(
    "Section type",
    ["Rectangular", "T-section", "Circular", "Custom (coordinates)"],
)

with st.form("inputs"):
    st.subheader("Geometry (mm)")
    if shape == "Rectangular":
        c1, c2, c3 = st.columns(3)
        b = c1.number_input("Width b", 100.0, 3000.0, 300.0)
        h = c2.number_input("Overall depth h", 100.0, 3000.0, 600.0)
        cover = c3.number_input("Cover to bar surface", 10.0, 100.0, 35.0)
    elif shape == "T-section":
        c1, c2 = st.columns(2)
        b = c1.number_input("Web width b", 100.0, 3000.0, 300.0)
        h = c2.number_input("Overall depth h", 100.0, 3000.0, 600.0)
        bf = c1.number_input("Flange width bf", 100.0, 6000.0, 1200.0)
        hf = c2.number_input("Flange depth hf", 50.0, 1000.0, 150.0)
        cover = st.number_input("Cover to bar surface", 10.0, 100.0, 35.0)
    elif shape == "Circular":
        c1, c2 = st.columns(2)
        diam = c1.number_input("Diameter D", 200.0, 3000.0, 500.0)
        cover = c2.number_input("Cover to bar surface", 10.0, 100.0, 40.0)
    else:
        st.caption("Concrete outline: vertices in order (x, y) in mm, y pointing up. "
                   "Add or delete rows as needed. Compression is at the top (+y).")
        poly_df = st.data_editor(
            pd.DataFrame({"x": [0.0, 300.0, 300.0, 0.0],
                          "y": [0.0, 0.0, 600.0, 600.0]}),
            num_rows="dynamic", key="poly",
        )

    st.subheader("Reinforcement")
    if shape in ("Rectangular", "T-section"):
        c1, c2 = st.columns(2)
        label_top = "Top bars (no.)" if shape == "Rectangular" else "Top bars in flange (no.)"
        n_top = c1.number_input(label_top, 0, 30, 3)
        dia_top = c1.number_input("Top bar dia (mm)", 6, 40, 16)
        n_bot = c2.number_input("Bottom bars (no.)", 0, 30, 4)
        dia_bot = c2.number_input("Bottom bar dia (mm)", 6, 40, 25)
    elif shape == "Circular":
        c1, c2 = st.columns(2)
        n_bar = c1.number_input("Number of bars (equally spaced)", 2, 60, 8)
        dia_bar = c2.number_input("Bar dia (mm)", 6, 40, 25)
    else:
        st.caption("Bars: centre coordinates (x, y) and diameter, in mm.")
        bar_df = st.data_editor(
            pd.DataFrame({"x": [50.0, 150.0, 250.0, 50.0, 250.0],
                          "y": [50.0, 50.0, 50.0, 550.0, 550.0],
                          "dia": [25.0, 25.0, 25.0, 16.0, 16.0]}),
            num_rows="dynamic", key="bars",
        )

    st.subheader("Materials")
    c1, c2 = st.columns(2)
    cls = c1.selectbox("Concrete class", list(CLASSES), index=5)
    acc = c1.number_input("αcc", 0.7, 1.0, 0.85)
    gamma_c = c1.number_input("γc", 1.0, 2.0, 1.5)
    fyk = c2.number_input("fyk (MPa)", 200.0, 700.0, 500.0)
    gamma_s = c2.number_input("γs", 1.0, 1.5, 1.15)
    es = c2.number_input("Es (MPa)", 100e3, 250e3, 200e3)
    eud = c2.number_input("Steel ultimate strain εud", 0.01, 0.1, 0.0225, format="%.4f")

    st.subheader("Design actions")
    c1, c2 = st.columns(2)
    n_kn = c1.number_input("N_Ed (kN, compression +ve)", -5000.0, 50000.0, 0.0)
    m_ed = c2.number_input("M_Ed (kNm)", 0.0, 50000.0, 150.0)
    m_ser = st.number_input("Service moment M_ser (kNm)", 0.0, 50000.0, 100.0)

    run = st.form_submit_button("Calculate")

if run:
    p = ec2_concrete(CLASSES[cls], acc, gamma_c)
    fyd = fyk / gamma_s

    concrete = Concrete(
        name=cls, density=2.4e-6,
        stress_strain_profile=ConcreteLinear(elastic_modulus=p["ecm"]),
        ultimate_stress_strain_profile=RectangularStressBlock(
            compressive_strength=p["fcd"],   # fcd already includes αcc
            alpha=p["eta"], gamma=p["lam"], ultimate_strain=p["ecu3"],
        ),
        flexural_tensile_strength=p["fctm"], colour="lightgrey",
    )
    steel = SteelBar(
        name="B500", density=7.85e-6,
        stress_strain_profile=SteelElasticPlastic(
            yield_strength=fyd, elastic_modulus=es, fracture_strain=eud),
        colour="grey",
    )

    # ---------- build geometry ----------
    if shape == "Rectangular":
        geom = concrete_rectangular_section(
            d=h, b=b,
            dia_top=dia_top, area_top=area(dia_top), n_top=n_top, c_top=cover,
            dia_bot=dia_bot, area_bot=area(dia_bot), n_bot=n_bot, c_bot=cover,
            n_circle=4, conc_mat=concrete, steel_mat=steel,
        )
    elif shape == "T-section":
        if hf >= h:
            st.error("Flange depth must be less than the overall depth.")
            st.stop()
        geom = concrete_tee_section(
            d=h, b=b, d_f=hf, b_f=bf,
            dia_top=dia_top, area_top=area(dia_top), n_top=n_top, c_top=cover,
            dia_bot=dia_bot, area_bot=area(dia_bot), n_bot=n_bot, c_bot=cover,
            n_circle=4, conc_mat=concrete, steel_mat=steel,
        )
    elif shape == "Circular":
        geom = concrete_circular_section(
            d=diam, area_conc=area(diam), n_conc=64,
            dia_bar=dia_bar, area_bar=area(dia_bar), n_bar=n_bar, cover=cover,
            n_circle=4, conc_mat=concrete, steel_mat=steel,
        )
    else:
        pts = list(poly_df.dropna().itertuples(index=False, name=None))
        bars = bar_df.dropna()
        poly = Polygon(pts) if len(pts) >= 3 else None
        if poly is None or not poly.is_valid or poly.area <= 0:
            st.error("The outline needs at least 3 vertices and must not "
                     "cross itself.")
            st.stop()
        if bars.empty:
            st.error("Add at least one reinforcing bar.")
            st.stop()
        geom = Geometry(geom=poly, material=concrete)
        for row in bars.itertuples(index=False):
            if row.dia <= 0 or not poly.contains(Point(row.x, row.y)):
                st.error(f"Bar at ({row.x}, {row.y}) is outside the outline or "
                         "has an invalid diameter.")
                st.stop()
            geom = add_bar(geom, area=area(row.dia), material=steel,
                           x=row.x, y=row.y, n=4)

    sec = ConcreteSection(geom)
    n_ed = n_kn * 1e3  # compression +ve

    # cracked properties (sagging: compression at top), reused by tabs 4 and 5
    cr = sec.calculate_cracked_properties(theta=0)
    cr.calculate_transformed_properties(elastic_modulus=p["ecm"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Section", "Capacity", "M-N diagram", "Stiffness", "Stresses"]
    )

    with tab1:
        st.pyplot(sec.plot_section(render=False).get_figure())
        st.caption(f"fcd = {p['fcd']:.1f} MPa, fyd = {fyd:.0f} MPa, "
                   f"λ = {p['lam']:.3f}, η = {p['eta']:.3f}, "
                   f"εcu3 = {p['ecu3']:.4f}, Ecm = {p['ecm']/1e3:.1f} GPa")

    with tab2:
        res = sec.ultimate_bending_capacity(theta=0, n=n_ed)
        m_rd = abs(res.m_xy) / 1e6
        c1, c2, c3 = st.columns(3)
        c1.metric("M_Rd (kNm)", f"{m_rd:,.1f}")
        c2.metric("M_Ed (kNm)", f"{m_ed:,.1f}")
        c3.metric("Utilisation", f"{m_ed / m_rd:.2f}" if m_rd > 0 else "n/a")

    with tab3:
        mi = sec.moment_interaction_diagram(theta=0, progress_bar=False)
        n_list, m_list = mi.get_results_lists(moment="m_x")
        m_kn = [abs(m) / 1e6 for m in m_list]
        n_plot = [n / 1e3 for n in n_list]       # compression +ve
        fig, ax = plt.subplots()
        ax.plot(m_kn, n_plot, "-b", label="Design resistance")
        ax.plot(m_ed, n_kn, "ro", label="(M_Ed, N_Ed)")
        ax.set_xlabel("M (kNm)")
        ax.set_ylabel("N (kN), compression +ve")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    with tab4:
        tg = sec.get_transformed_gross_properties(elastic_modulus=p["ecm"])
        i_hom = tg.ixx_c
        i_cr = cr.iuu_cr
        c1, c2, c3 = st.columns(3)
        c1.metric("I homogeneous (×10⁹ mm⁴)", f"{i_hom / 1e9:.3f}")
        c2.metric("I cracked (×10⁹ mm⁴)", f"{i_cr / 1e9:.3f}")
        c3.metric("I cracked / I homogeneous", f"{i_cr / i_hom:.2f}")
        c1, c2 = st.columns(2)
        c1.metric("M_cr (kNm)", f"{cr.m_cr / 1e6:,.1f}")
        c2.metric("Neutral axis depth, cracked (mm)", f"{cr.d_nc:,.0f}")
        st.caption("Transformed to concrete using Ecm (short-term). Sagging "
                   "bending, compression at top. Cracking uses fctm.")

    with tab5:
        sres = sec.calculate_cracked_stress(cracked_results=cr, m=m_ser * 1e6)
        lo, hi = sres.get_concrete_stress_limits()
        sig_c = max(abs(lo), abs(hi))
        sig_s = max(abs(s) for s in sres.lumped_reinforcement_stresses)
        fck = CLASSES[cls]
        c1, c2 = st.columns(2)
        c1.metric("Max concrete stress (MPa)", f"{sig_c:.1f}")
        c1.caption(f"EC2 7.2(2) limit 0.6·fck = {0.6 * fck:.1f} MPa → "
                   f"ratio {sig_c / (0.6 * fck):.2f}")
        c2.metric("Max steel stress (MPa)", f"{sig_s:.0f}")
        c2.caption(f"EC2 7.2(5) limit 0.8·fyk = {0.8 * fyk:.0f} MPa → "
                   f"ratio {sig_s / (0.8 * fyk):.2f}")
        st.pyplot(sres.plot_stress(render=False).get_figure())
        st.caption("Linear-elastic cracked section, sagging bending, "
                   "short-term Ecm, bending only (axial load not included).")
