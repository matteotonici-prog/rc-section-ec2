import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Polygon, Point
from concreteproperties import Concrete, ConcreteSection, SteelBar
from concreteproperties.stress_strain_profile import (
    BilinearStressStrain, ConcreteLinear, EurocodeParabolicUltimate,
    RectangularStressBlock, SteelElasticPlastic, SteelHardening,
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
    """EN 1992-1-1 Table 3.1 / cl. 3.1.7 parameters (MPa, strains as ratios)."""
    fcm = fck + 8
    fcd = acc * fck / gamma_c
    if fck <= 50:
        lam, eta = 0.8, 1.0
        ecu = 0.0035                  # εcu2 = εcu3
        ec2, n_exp, ec3 = 0.002, 2.0, 0.00175
        fctm = 0.30 * fck ** (2 / 3)
    else:
        r = ((90 - fck) / 100) ** 4
        lam = 0.8 - (fck - 50) / 400
        eta = 1.0 - (fck - 50) / 200
        ecu = (2.6 + 35 * r) / 1000   # εcu2 = εcu3
        ec2 = (2.0 + 0.085 * (fck - 50) ** 0.53) / 1000
        n_exp = 1.4 + 23.4 * r
        ec3 = (1.75 + 0.55 * (fck - 50) / 40) / 1000
        fctm = 2.12 * math.log(1 + fcm / 10)
    ecm = 22000 * (fcm / 10) ** 0.3
    return dict(fcd=fcd, lam=lam, eta=eta, ecu=ecu, ec2=ec2, n_exp=n_exp,
                ec3=ec3, fctm=fctm, ecm=ecm)


def area(dia):
    return math.pi * dia**2 / 4


PLOT_WIDTH = 0.8   # fraction of the page width used by each plot (1.0 = full width)


def show(fig, width=PLOT_WIDTH):
    if width >= 1.0:
        st.pyplot(fig)
    else:
        col, _ = st.columns([width, 1 - width])
        with col:
            st.pyplot(fig)
    plt.close(fig)


def bar_stress_table(sres):
    """One row per reinforcing bar: position, diameter, stress, force."""
    rows = []
    for i, (g, s) in enumerate(
        zip(sres.lumped_reinforcement_geometries,
            sres.lumped_reinforcement_stresses), start=1
    ):
        cx, cy = g.calculate_centroid()
        a = g.calculate_area()
        s = float(s)
        rows.append({
            "Bar": i,
            "x (mm)": round(cx, 1),
            "y (mm)": round(cy, 1),
            "Dia (mm)": round(math.sqrt(4 * a / math.pi), 1),
            "Stress (MPa)": round(s, 1),
            "Force (kN)": round(s * a / 1e3, 1),
            "State": "compression" if s > 1e-6 else ("tension" if s < -1e-6 else "unstressed"),
        })
    return pd.DataFrame(rows)


def plot_bar_stresses(sec, df, title):
    """Section outline with every bar drawn to scale, coloured and labelled by stress."""
    ax = sec.plot_section(render=False)
    fig = ax.get_figure()
    vmax = max(float(df["Stress (MPa)"].abs().max()), 1e-6)
    norm = plt.Normalize(-vmax, vmax)
    cmap = plt.cm.bwr
    for _, r in df.iterrows():
        ax.add_patch(Circle((r["x (mm)"], r["y (mm)"]), r["Dia (mm)"] / 2,
                            facecolor=cmap(norm(r["Stress (MPa)"])),
                            edgecolor="k", zorder=5))
        ax.annotate(f'{int(r["Bar"])}: {r["Stress (MPa)"]:.0f}',
                    (r["x (mm)"], r["y (mm)"]), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=7,
                    zorder=6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="Bar stress (MPa), +ve compression")
    ax.set_title(title)
    return fig


st.title("RC section check - Eurocode 2")

# Outside the form so the inputs below update as soon as the type changes
shape = st.selectbox(
    "Section type",
    ["Rectangular", "T-section", "Circular", "Custom (coordinates)"],
)

with st.form("inputs"):
    # ---------------- geometry ----------------
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
        st.caption("Concrete outline: vertices in order (x, y) in mm, y pointing "
                   "up. Add or delete rows as needed.")
        poly_df = st.data_editor(
            pd.DataFrame({"x": [0.0, 300.0, 300.0, 0.0],
                          "y": [0.0, 0.0, 600.0, 600.0]}),
            num_rows="dynamic", key="poly",
        )

    # ---------------- reinforcement ----------------
    st.subheader("Reinforcement")
    if shape in ("Rectangular", "T-section"):
        c1, c2 = st.columns(2)
        label_top = ("Top bars (no.)" if shape == "Rectangular"
                     else "Top bars in flange (no.)")
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

    # ---------------- materials ----------------
    st.subheader("Materials")
    c1, c2 = st.columns(2)
    cls = c1.selectbox("Concrete class", list(CLASSES), index=5)
    acc = c1.number_input("αcc", 0.7, 1.0, 0.85)
    gamma_c = c1.number_input("γc", 1.0, 2.0, 1.5)
    fyk = c2.number_input("fyk (MPa)", 200.0, 700.0, 500.0)
    gamma_s = c2.number_input("γs", 1.0, 1.5, 1.15)
    es = c2.number_input("Es (MPa)", 100e3, 250e3, 200e3)
    eud = c2.number_input("Steel design ultimate strain εud", 0.01, 0.1, 0.0225,
                          format="%.4f")

    ult_model = st.radio(
        "Concrete ultimate stress-strain model (EC2 3.1.7)",
        ["Rectangular stress block", "Parabola-rectangle", "Bilinear"],
        horizontal=True,
    )
    steel_model = st.radio(
        "Steel stress-strain model (EC2 3.2.7)",
        ["Elastic-plastic (horizontal top branch)",
         "Strain hardening (inclined top branch)"],
        horizontal=True,
    )
    k_steel = st.number_input("k = ft/fy (used only with strain hardening)",
                              1.0, 1.5, 1.05)
    st.caption("Reinforcement class A: k ≥ 1.05, εuk ≥ 2.5%. Class B: k ≥ 1.08, "
               "εuk ≥ 5%. Class C: k = 1.15 to 1.35, εuk ≥ 7.5%. "
               "EC2 recommends εud = 0.9·εuk.")

    # ---------------- stiffness basis ----------------
    st.subheader("Stiffness and service stresses")
    c1, c2 = st.columns(2)
    basis = c1.radio(
        "Concrete modulus to use",
        ["Short-term (Ecm)", "Long-term (modular ratio n)"],
    )
    n_long = c2.number_input("Long-term modular ratio n = Es / Ec,eff",
                             5.0, 30.0, 15.0)

    # ---------------- actions ----------------
    st.subheader("Actions")
    st.caption("Ultimate limit state")
    c1, c2 = st.columns(2)
    n_kn = c1.number_input("N_Ed (kN, compression +ve)", -5000.0, 50000.0, 0.0)
    m_ed = c2.number_input("M_Ed (kNm, enter as positive)", 0.0, 50000.0, 150.0)
    st.caption("Service combination")
    c1, c2 = st.columns(2)
    n_ser = c1.number_input("N_ser (kN, compression +ve)", -5000.0, 50000.0, 0.0)
    m_ser = c2.number_input("M_ser (kNm, enter as positive)", 0.0, 50000.0, 100.0)
    direction = st.radio(
        "Bending direction",
        ["Sagging (compression at top)", "Hogging (compression at bottom)"],
        horizontal=True,
    )

    run = st.form_submit_button("Calculate")

if run:
    p = ec2_concrete(CLASSES[cls], acc, gamma_c)
    fyd = fyk / gamma_s
    sagging = direction.startswith("Sagging")
    theta = 0.0 if sagging else math.pi
    dir_txt = "sagging" if sagging else "hogging"
    comp_face = "top" if sagging else "bottom"

    # Concrete modulus for stiffness / service stresses (ULS is not affected)
    long_term = basis.startswith("Long")
    e_service = es / n_long if long_term else p["ecm"]
    basis_txt = "Long-term" if long_term else "Short-term"

    # ---------------- concrete ultimate profile ----------------
    if ult_model == "Rectangular stress block":
        ult_profile = RectangularStressBlock(
            compressive_strength=p["fcd"],   # fcd already includes αcc
            alpha=p["eta"], gamma=p["lam"], ultimate_strain=p["ecu"],
        )
    elif ult_model == "Parabola-rectangle":
        ult_profile = EurocodeParabolicUltimate(
            compressive_strength=p["fcd"], compressive_strain=p["ec2"],
            ultimate_strain=p["ecu"], n=p["n_exp"],
        )
    else:
        ult_profile = BilinearStressStrain(
            compressive_strength=p["fcd"], compressive_strain=p["ec3"],
            ultimate_strain=p["ecu"],
        )

    concrete = Concrete(
        name=cls, density=2.4e-6,
        stress_strain_profile=ConcreteLinear(elastic_modulus=e_service),
        ultimate_stress_strain_profile=ult_profile,
        flexural_tensile_strength=p["fctm"], colour="lightgrey",
    )

    # ---------------- steel profile ----------------
    if steel_model.startswith("Elastic"):
        steel_profile = SteelElasticPlastic(
            yield_strength=fyd, elastic_modulus=es, fracture_strain=eud)
    else:
        steel_profile = SteelHardening(
            yield_strength=fyd, elastic_modulus=es, fracture_strain=eud,
            ultimate_strength=k_steel * fyd)

    steel = SteelBar(
        name="B500", density=7.85e-6,
        stress_strain_profile=steel_profile, colour="grey",
    )

    # ---------------- build geometry ----------------
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
                st.error(f"Bar at ({row.x}, {row.y}) is outside the outline "
                         "or has an invalid diameter.")
                st.stop()
            geom = add_bar(geom, area=area(row.dia), material=steel,
                           x=row.x, y=row.y, n=4)

    sec = ConcreteSection(geom)
    n_ed = n_kn * 1e3  # compression +ve

    # Elastic cracked properties (bending only), used by the stiffness tab
    cr = sec.calculate_cracked_properties(theta=theta)
    cr.calculate_transformed_properties(elastic_modulus=e_service)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Section", "Stiffness", "Service stresses", "ULS capacity", "M-N diagram"]
    )

    # ---------------- tab 1: section ----------------
    with tab1:
        show(sec.plot_section(render=False).get_figure())
        st.caption(f"fcd = {p['fcd']:.1f} MPa, fyd = {fyd:.0f} MPa, "
                   f"λ = {p['lam']:.3f}, η = {p['eta']:.3f}, "
                   f"εcu3 = {p['ecu']:.4f}, Ecm = {p['ecm']/1e3:.1f} GPa")

    # ---------------- tab 2: stiffness ----------------
    with tab2:
        tg = sec.get_transformed_gross_properties(elastic_modulus=e_service)
        i_hom = tg.ixx_c
        i_cr = cr.iuu_cr
        st.markdown(f"**{basis_txt} stiffness**: Ec = {e_service / 1e3:.1f} GPa, "
                    f"modular ratio n = Es/Ec = {es / e_service:.1f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("I homogeneous (×10⁹ mm⁴)", f"{i_hom / 1e9:.3f}")
        c2.metric("I cracked (×10⁹ mm⁴)", f"{i_cr / 1e9:.3f}")
        c3.metric("I cracked / I homogeneous", f"{i_cr / i_hom:.2f}")
        c1, c2 = st.columns(2)
        c1.metric("EI homogeneous (kNm²)", f"{e_service * i_hom * 1e-9:,.0f}")
        c2.metric("EI cracked (kNm²)", f"{e_service * i_cr * 1e-9:,.0f}")
        c1, c2 = st.columns(2)
        c1.metric("M_cr (kNm)", f"{cr.m_cr / 1e6:,.1f}")
        c2.metric("Neutral axis depth, cracked (mm)", f"{cr.d_nc:,.0f}")
        st.caption(f"Section transformed to concrete. {dir_txt.capitalize()} "
                   f"bending, compression at {comp_face}. Cracked properties "
                   "are for bending only (no axial force). Cracking uses fctm.")

    # ---------------- tab 3: service stresses (with axial force) ----------------
    with tab3:
        st.markdown(f"**{basis_txt} basis**: Ec = {e_service / 1e3:.1f} GPa "
                    f"(n = {es / e_service:.1f}), M_ser = {m_ser:,.0f} kNm, "
                    f"N_ser = {n_ser:,.0f} kN, {dir_txt}")
        with st.spinner("Running service analysis..."):
            try:
                mk = sec.moment_curvature_analysis(
                    theta=theta, n=n_ser * 1e3, progress_bar=False
                )
                sres = sec.calculate_service_stress(
                    moment_curvature_results=mk, m=m_ser * 1e6
                )
                method = ("Moment-curvature analysis including N_ser "
                          "(cracking taken into account)")
            except Exception as exc:
                st.warning(f"The service analysis with axial force failed "
                           f"({exc}). Showing the elastic cracked bending-only "
                           "result instead; N_ser is NOT included.")
                sres = sec.calculate_cracked_stress(
                    cracked_results=cr, m=m_ser * 1e6
                )
                method = "Elastic cracked section, bending only (N_ser ignored)"

        lo, hi = sres.get_concrete_stress_limits()
        sig_c = max(abs(lo), abs(hi))
        df_ser = bar_stress_table(sres)
        worst = df_ser.loc[df_ser["Stress (MPa)"].abs().idxmax()]
        sig_s = abs(float(worst["Stress (MPa)"]))
        fck = CLASSES[cls]

        c1, c2 = st.columns(2)
        c1.metric("Max concrete stress (MPa)", f"{sig_c:.1f}")
        c1.caption(f"EC2 7.2(2): 0.6·fck = {0.6 * fck:.1f} MPa "
                   f"(ratio {sig_c / (0.6 * fck):.2f}), characteristic. "
                   f"EC2 7.2(3): 0.45·fck = {0.45 * fck:.1f} MPa "
                   f"(ratio {sig_c / (0.45 * fck):.2f}), quasi-permanent.")
        c2.metric("Max steel stress (MPa)", f"{sig_s:.0f}")
        c2.caption(f"Bar {int(worst['Bar'])}. EC2 7.2(5): 0.8·fyk = "
                   f"{0.8 * fyk:.0f} MPa (ratio {sig_s / (0.8 * fyk):.2f}).")

        show(sres.plot_stress(render=False).get_figure())
        show(plot_bar_stresses(sec, df_ser, "Reinforcement stresses (service)"))
        st.dataframe(df_ser, hide_index=True)
        st.caption(f"Method: {method}. Compression is positive, so bars in "
                   "tension show negative stress. M and N are taken about the "
                   "gross centroid of the section.")

    # ---------------- tab 4: ULS capacity ----------------
    with tab4:
        res = sec.ultimate_bending_capacity(theta=theta, n=n_ed)
        m_rd = abs(res.m_xy) / 1e6
        c1, c2, c3 = st.columns(3)
        c1.metric("M_Rd (kNm)", f"{m_rd:,.1f}")
        c2.metric("M_Ed (kNm)", f"{m_ed:,.1f}")
        c3.metric("Utilisation", f"{m_ed / m_rd:.2f}" if m_rd > 0 else "n/a")
        d_n = getattr(res, "d_n", None)
        extra = f", neutral axis depth d_n = {d_n:,.0f} mm" if d_n else ""
        st.caption(f"{dir_txt.capitalize()} bending, compression at {comp_face}, "
                   f"with N_Ed = {n_kn:,.0f} kN{extra}. Concrete: {ult_model}. "
                   f"Steel: {steel_model}.")

        uls = sec.calculate_ultimate_stress(ultimate_results=res)
        st.subheader("Stresses at the ultimate limit state")
        show(uls.plot_stress(render=False).get_figure())
        df_uls = bar_stress_table(uls)
        show(plot_bar_stresses(sec, df_uls, "Reinforcement stresses at ULS"))
        st.dataframe(df_uls, hide_index=True)
        st.caption("The stresses shown are those of the section at its "
                   "ultimate capacity M_Rd for the applied axial force, not "
                   "at M_Ed. Compression is positive.")

        with st.expander("Stress-strain curves used"):
            try:
                show(ult_profile.plot_stress_strain(
                    title=f"Concrete ultimate: {ult_model}", render=False
                ).get_figure())
                show(steel_profile.plot_stress_strain(
                    title=f"Steel: {steel_model}", render=False
                ).get_figure())
            except Exception as exc:
                st.info(f"The curves could not be plotted ({exc}).")

    # ---------------- tab 5: M-N diagram ----------------
    with tab5:
        fig, ax = plt.subplots()
        for th, lab, style in [(0.0, "Sagging", "-b"), (math.pi, "Hogging", "-g")]:
            mi = sec.moment_interaction_diagram(
                theta=th, n_points=48, progress_bar=False
            )
            n_list, m_list = mi.get_results_lists(moment="m_x")
            ms = [m / 1e6 for m in m_list]          # keep the real sign
            if (th == 0.0 and sum(ms) < 0) or (th != 0.0 and sum(ms) > 0):
                ms = [-m for m in ms]
            ax.plot(ms, [n / 1e3 for n in n_list], style, label=lab)
        applied_sign = 1 if sagging else -1
        ax.plot(applied_sign * m_ed, n_kn, "ro", label="(M_Ed, N_Ed)")
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("M (kNm): sagging +ve, hogging -ve")
        ax.set_ylabel("N (kN), compression +ve")
        ax.grid(True)
        ax.legend()
        show(fig)
