import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Polygon, Point, box
from concreteproperties import Concrete, ConcreteSection, SteelBar
from concreteproperties.stress_strain_profile import (
    ConcreteLinear, ConcreteLinearNoTension, EurocodeParabolicUltimate,
    RectangularStressBlock, SteelElasticPlastic,
)
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library import concrete_circular_section
from sectionproperties.pre.library.concrete_sections import add_bars

MAX_LAYERS = 50
PLOT_WIDTH = 0.8   # fraction of the page width used by each plot (1.0 = full width)

CLASSES = {"C20/25": 20, "C25/30": 25, "C30/37": 30, "C32/40": 32, "C35/45": 35,
           "C40/50": 40, "C45/55": 45, "C50/60": 50, "C55/67": 55, "C60/75": 60,
           "C70/85": 70, "C80/95": 80, "C90/105": 90}

# EN 1992-1-1 Table 7.3N: max bar spacing (mm) for wk = 0.4 / 0.3 / 0.2 mm
TABLE_73N = {160: (300, 300, 200), 200: (300, 250, 150), 240: (250, 200, 100),
             280: (200, 150, 50), 320: (150, 100, None), 360: (100, 50, None)}

LAYER_COLS = {
    "n": st.column_config.NumberColumn("n bars", min_value=1, step=1, format="%d"),
    "dia": st.column_config.NumberColumn("dia (mm)", min_value=6.0),
    "gap": st.column_config.NumberColumn("gap to previous layer (mm)", min_value=0.0),
}


# =============================================================== helpers
def ec2_concrete(fck, acc, gamma_c):
    """EN 1992-1-1 Table 3.1 / cl. 3.1.7 parameters (MPa, strains as ratios)."""
    fcm = fck + 8
    fcd = acc * fck / gamma_c
    if fck <= 50:
        lam, eta = 0.8, 1.0
        ecu = 0.0035                   # εcu2 = εcu3
        ec2, n_exp = 0.002, 2.0
        fctm = 0.30 * fck ** (2 / 3)
    else:
        r = ((90 - fck) / 100) ** 4
        lam = 0.8 - (fck - 50) / 400
        eta = 1.0 - (fck - 50) / 200
        ecu = (2.6 + 35 * r) / 1000    # εcu2 = εcu3
        ec2 = (2.0 + 0.085 * (fck - 50) ** 0.53) / 1000
        n_exp = 1.4 + 23.4 * r
        fctm = 2.12 * math.log(1 + fcm / 10)
    ecm = 22000 * (fcm / 10) ** 0.3
    return dict(fcd=fcd, lam=lam, eta=eta, ecu=ecu, ec2=ec2, n_exp=n_exp,
                fctm=fctm, ecm=ecm)


def area(dia):
    return math.pi * dia**2 / 4


def fmt(v, nd=0):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


def show(fig, width=PLOT_WIDTH):
    if width >= 1.0:
        st.pyplot(fig)
    else:
        col, _ = st.columns([width, 1 - width])
        with col:
            st.pyplot(fig)
    plt.close(fig)


def max_spacing_73n(sigma_s, wmax):
    """Table 7.3N: conservative lookup (next tabulated stress, stricter wk column)."""
    col = 0 if wmax >= 0.4 else (1 if wmax >= 0.3 else (2 if wmax >= 0.2 else None))
    if col is None:
        return None
    for s in sorted(TABLE_73N):
        if sigma_s <= s:
            return TABLE_73N[s][col]
    return None


# ----------------------------------------------------------- spacing checks
def make_row(name, n, dia, spacing, dg, max_lim):
    """One line of the spacing check. spacing = c/c within a layer (None if 1 bar)."""
    req = max(dia, dg + 5.0, 20.0)          # EC2 8.2(2): max(k1·φ, dg + k2, 20 mm)
    problems = []
    clear = None
    if spacing is not None:
        clear = spacing - dia
        if clear < req - 1e-6:
            problems.append(f"clear spacing < {req:.0f}")
        if max_lim is not None and spacing > max_lim + 1e-6:
            problems.append(f"c/c > {max_lim:.0f}")
    return {
        "Check": name, "n": str(n), "dia (mm)": fmt(dia), "c/c (mm)": fmt(spacing),
        "clear (mm)": fmt(clear),
        "min clear (mm)": fmt(req) if spacing is not None else "-",
        "max c/c (mm)": fmt(max_lim),
        "Status": "✅ OK" if not problems else "⚠️ " + "; ".join(problems),
    }


def build_layers(kind, b, h, bf, hf, bottom_df, top_df, cover_tot, dg, max_lim):
    """Rectangular / T-section: bars from bottom and top layer tables."""
    bars, rows, errors = [], [], []

    def x_range(y):
        if kind == "T-section":
            w = bf if y > h - hf else b
            return -w / 2, w / 2
        return 0.0, b

    for label, df, from_top in (("Bottom", bottom_df, False), ("Top", top_df, True)):
        prev_dist, prev_dia = None, None
        for k, r in enumerate(df.itertuples(index=False), start=1):
            n, d, gap = int(round(r.n)), float(r.dia), float(r.gap)
            name = f"{label} {k}"
            if n < 1 or d <= 0 or gap < 0:
                errors.append(f"{name}: needs n ≥ 1, a positive diameter and a gap ≥ 0.")
                continue
            if prev_dist is None:
                dist = cover_tot + d / 2              # first layer: cover + link + φ/2
            else:
                dist = prev_dist + prev_dia / 2 + gap + d / 2
                req = max(d, prev_dia, dg + 5.0, 20.0)
                ok = gap >= req - 1e-6
                rows.append({
                    "Check": f"{name}: vertical gap to previous layer", "n": "-",
                    "dia (mm)": "-", "c/c (mm)": "-", "clear (mm)": fmt(gap),
                    "min clear (mm)": fmt(req), "max c/c (mm)": "-",
                    "Status": "✅ OK" if ok else f"⚠️ gap < {req:.0f}",
                })
            prev_dist, prev_dia = dist, d
            y = h - dist if from_top else dist
            xmin, xmax = x_range(y)
            xa, xb = xmin + cover_tot + d / 2, xmax - cover_tot - d / 2
            if n == 1:
                xs, spacing = [(xmin + xmax) / 2], None
            else:
                if xb <= xa:
                    errors.append(f"{name}: the covers leave no room for the bars.")
                    continue
                spacing = (xb - xa) / (n - 1)
                if spacing < d - 1e-6:
                    errors.append(f"{name}: {n} bars of {d:.0f} mm do not fit "
                                  "(bars would overlap).")
                    continue
                xs = [xa + j * spacing for j in range(n)]
            for x in xs:
                bars.append((x, y, d, name))
            rows.append(make_row(name, n, d, spacing, dg, max_lim))
    return bars, rows, errors


def build_wall(t, length, in_plane, faces, cover_tot, dg, d_min):
    """Wall: bars given as diameter @ spacing on each face, along the length."""
    bars, rows, errors = [], [], []
    v_pos = []
    lim = min(3 * t, 400.0)                           # EC2 9.6.2(1)
    for k, (name, d, s) in enumerate(faces):
        v = cover_tot + d / 2 if k == 0 else t - cover_tot - d / 2
        v_pos.append(v)
        u0 = cover_tot + d / 2
        avail = length - 2 * u0
        if s <= 0 or avail < 0:
            errors.append(f"{name}: invalid spacing or the wall is too short.")
            continue
        if s < d - 1e-6:
            errors.append(f"{name}: spacing {s:.0f} mm is smaller than the bar diameter.")
            continue
        n = int(math.floor(avail / s + 1e-9)) + 1
        start = u0 + (avail - (n - 1) * s) / 2        # group centred along the wall
        for j in range(n):
            u = start + j * s
            x, y = (v, u) if in_plane else (u, v)
            bars.append((x, y, d, name))
        rows.append(make_row(f"{name}: H{d:.0f}@{s:.0f} ({n} bars)", n, d, s, dg, lim))
    if len(v_pos) == 2:
        dist = v_pos[1] - v_pos[0]
        rows.append({
            "Check": f"Distance between the two layers (c/c), min {d_min:.0f}",
            "n": "-", "dia (mm)": "-", "c/c (mm)": fmt(dist), "clear (mm)": "-",
            "min clear (mm)": "-", "max c/c (mm)": "-",
            "Status": "✅ OK" if dist >= d_min else f"⚠️ < {d_min:.0f} mm between layers",
        })
    return bars, rows, errors


def custom_row(bars, dg):
    worst = None
    for i in range(len(bars)):
        xi, yi, di, _ = bars[i]
        for j in range(i + 1, len(bars)):
            xj, yj, dj, _ = bars[j]
            clear = math.hypot(xi - xj, yi - yj) - (di + dj) / 2
            req = max(di, dj, dg + 5.0, 20.0)
            margin = clear - req
            if worst is None or margin < worst[0]:
                worst = (margin, clear, req, max(di, dj))
    if worst is None:
        return None
    margin, clear, req, dia = worst
    return {
        "Check": "Closest pair of bars", "n": "-", "dia (mm)": fmt(dia),
        "c/c (mm)": "-", "clear (mm)": fmt(clear), "min clear (mm)": fmt(req),
        "max c/c (mm)": "-",
        "Status": "✅ OK" if margin >= -1e-6 else f"⚠️ clear spacing < {req:.0f}",
    }


def check_bars(poly, bars):
    """Blocking checks: bars inside the outline and not overlapping."""
    for x, y, d, lab in bars:
        if d <= 0:
            return [f"{lab}: invalid bar diameter."]
        if not poly.contains(Point(x, y).buffer(d / 2)):
            return [f"{lab}: the bar at ({x:.0f}, {y:.0f}) mm is not fully "
                    "inside the concrete outline."]
    for i in range(len(bars)):
        xi, yi, di, li = bars[i]
        for j in range(i + 1, len(bars)):
            xj, yj, dj, lj = bars[j]
            if math.hypot(xi - xj, yi - yj) < (di + dj) / 2 - 1e-6:
                return [f"Bars overlap: {li} and {lj}."]
    return []


# ---------------------------------------------------------------- plots
def plot_preview(poly, bars, title, band=None):
    fig, ax = plt.subplots()
    for g in getattr(poly, "geoms", [poly]):
        xs, ys = g.exterior.xy
        ax.fill(xs, ys, facecolor="lightgrey", edgecolor="k", zorder=1)
    if band is not None and not band.is_empty:
        for g in getattr(band, "geoms", [band]):
            if g.geom_type == "Polygon":
                xs, ys = g.exterior.xy
                ax.fill(xs, ys, facecolor="gold", alpha=0.45, edgecolor="none",
                        zorder=2)
    for x, y, d, lab in bars:
        ax.add_patch(Circle((x, y), d / 2,
                            facecolor="tab:red" if lab == "T" else "dimgray",
                            edgecolor="k", zorder=5))
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.05)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=9)
    return fig


def bar_stress_table(sres):
    """One row per reinforcing bar (numeric, unrounded)."""
    rows = []
    for i, (g, s) in enumerate(
        zip(sres.lumped_reinforcement_geometries,
            sres.lumped_reinforcement_stresses), start=1
    ):
        cx, cy = g.calculate_centroid()
        a = g.calculate_area()
        s = float(s)
        rows.append({
            "Bar": i, "x (mm)": cx, "y (mm)": cy,
            "Dia (mm)": math.sqrt(4 * a / math.pi),
            "Stress (MPa)": s, "Force (kN)": s * a / 1e3,
            "State": "compression" if s > 1e-6 else ("tension" if s < -1e-6 else "unstressed"),
        })
    return pd.DataFrame(rows)


def plot_bar_stresses(
    sec,
    df,
    title,
    fig_width=10,
    fig_height=6,
    annotation_font=5,
    axis_font=7,
    title_font=8,
):
    """
    Draws the reinforcement bars to scale and colours them according
    to their calculated stress.

    Positive stress = compression.
    Negative stress = tension.
    """

    ax = sec.plot_section(render=False)
    fig = ax.get_figure()

    # Smaller Matplotlib figure
    fig.set_size_inches(fig_width, fig_height)

    if df.empty:
        ax.set_title(title, fontsize=title_font)
        return fig

    vmax = max(
        float(df["Stress (MPa)"].abs().max()),
        1e-6,
    )

    norm = plt.Normalize(-vmax, vmax)
    cmap = plt.cm.bwr

    for _, r in df.iterrows():
        stress = float(r["Stress (MPa)"])
        bar_number = int(r["Bar"])

        ax.add_patch(
            Circle(
                (r["x (mm)"], r["y (mm)"]),
                r["Dia (mm)"] / 2,
                facecolor=cmap(norm(stress)),
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
            )
        )

        ax.annotate(
            f"{bar_number}: {stress:.0f}",
            (r["x (mm)"], r["y (mm)"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=annotation_font,
            zorder=6,
        )

    sm = plt.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    sm.set_array([])

    colour_bar = fig.colorbar(
        sm,
        ax=ax,
        fraction=0.035,
        pad=0.03,
    )

    colour_bar.set_label(
        "Bar stress (MPa), +ve compression",
        fontsize=axis_font,
    )

    colour_bar.ax.tick_params(
        labelsize=max(axis_font - 1, 1),
    )

    ax.set_title(
        title,
        fontsize=title_font,
    )

    ax.set_xlabel(
        "x (mm)",
        fontsize=axis_font,
    )

    ax.set_ylabel(
        "y (mm)",
        fontsize=axis_font,
    )

    ax.tick_params(
        axis="both",
        labelsize=max(axis_font - 1, 1),
    )

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    ax.margins(0.08)

    fig.tight_layout()

    return fig


# ------------------------------------------------- crack width, EC2 7.3.4
def ec2_crack(df, poly, sagging, es, ecm, fct_eff, kt, k1, k3, k4, to_face):
    """Crack width for the tension layers, from the bar stresses of a cracked analysis."""
    minx, y_min, maxx, y_max = poly.bounds
    h_tot = y_max - y_min
    y_t, y_c = (y_min, y_max) if sagging else (y_max, y_min)

    y = df["y (mm)"].to_numpy(float)
    eps = df["Stress (MPa)"].to_numpy(float) / es       # compression +ve
    if np.ptp(y) < 1.0:
        return {"status": "error", "msg": "All bars are at one level, so the strain "
                "profile and neutral axis cannot be found. Add a second layer."}
    slope, icpt = np.polyfit(y, eps, 1)
    eps_t = icpt + slope * y_t
    eps_c = icpt + slope * y_c
    if eps_t >= -1e-9:
        return {"status": "uncracked"}
    x = 0.0 if eps_c <= 0 else h_tot * eps_c / (eps_c - eps_t)   # NA depth from comp. face
    e1 = -eps_t
    e2 = max(0.0, -eps_c)
    k2 = (e1 + e2) / (2 * e1)                                    # 0.5 bending, 1.0 tension

    d_ = df.copy()
    d_["area"] = np.pi * d_["Dia (mm)"] ** 2 / 4
    d_["dist"] = (d_["y (mm)"] - y_t).abs()
    ten = d_[d_["Stress (MPa)"] < -0.05].copy()
    if ten.empty:
        return {"status": "error", "msg": "No bars are in tension."}
    d_t = float((ten["dist"] * ten["area"]).sum() / ten["area"].sum())   # h - d
    hc_eff = min(2.5 * d_t, (h_tot - x) / 3.0, h_tot / 2.0)
    if sagging:
        bx = box(minx - 1, y_min, maxx + 1, y_min + hc_eff)
    else:
        bx = box(minx - 1, y_max - hc_eff, maxx + 1, y_max)
    band = poly.intersection(bx)
    ac_eff = band.area
    if ac_eff <= 0:
        return {"status": "error", "msg": "Ac,eff is zero (check the section)."}
    inband = ten[ten["dist"] <= hc_eff + 1e-6]
    if inband.empty:
        inband = ten[ten["dist"] <= ten["dist"].min() + 1e-6]
    rho = float(inband["area"].sum() / ac_eff)
    phi_eq = float((inband["Dia (mm)"] ** 2).sum() / inband["Dia (mm)"].sum())
    alpha_e = es / ecm

    ten = ten.assign(layer=ten["y (mm)"].round(0))
    rows = []
    for _, sub in sorted(ten.groupby("layer"), key=lambda kv: kv[1]["dist"].min()):
        sig = float(-sub["Stress (MPa)"].mean())
        phi_l = float(sub["Dia (mm)"].max())
        c_l = float((sub["dist"] - sub["Dia (mm)"] / 2).min())
        xs = np.sort(sub["x (mm)"].to_numpy(float))
        spacing = float(np.max(np.diff(xs))) if len(xs) > 1 else float("inf")
        wide = spacing > 5 * (c_l + phi_l / 2)
        sr = 1.3 * (h_tot - x) if wide else k3 * c_l + k1 * k2 * k4 * phi_eq / rho
        diff = max((sig - kt * fct_eff / rho * (1 + alpha_e * rho)) / es,
                   0.6 * sig / es)
        eps_l = sig / es
        factor = (e1 / eps_l) if (to_face and eps_l > 0) else 1.0
        rows.append(dict(
            y=float(sub["y (mm)"].mean()), n=len(sub), sigma=sig, c=c_l, phi=phi_l,
            spacing=spacing, wide=wide, sr=sr, diff=diff, wk=sr * diff * factor,
        ))
    ten_bars = [(r["x (mm)"], r["y (mm)"], r["Dia (mm)"], "T")
                for _, r in ten.iterrows()]
    return dict(status="ok", rows=rows, x=x, h_tot=h_tot, hc_eff=hc_eff,
                ac_eff=ac_eff, rho=rho, phi_eq=phi_eq, k2=k2, alpha_e=alpha_e,
                band=band, d_t=d_t, ten_bars=ten_bars)


# ================================================================= INPUTS
st.title("RC section check - Eurocode 2")

shape = st.selectbox(
    "Section type",
    ["Rectangular", "T-section", "Rectangular wall", "Circular",
     "Custom (coordinates)"],
    key="shape",
)

# ---------------------------------------------------------------- geometry
st.subheader("Geometry (mm)")
if shape == "Rectangular":
    c1, c2 = st.columns(2)
    b = c1.number_input("Width b", 100.0, 3000.0, 300.0, key="rect_b")
    h = c2.number_input("Overall depth h", 100.0, 3000.0, 500.0, key="rect_h")
elif shape == "T-section":
    c1, c2 = st.columns(2)
    b = c1.number_input("Web width b", 100.0, 3000.0, 300.0, key="tee_b")
    h = c2.number_input("Overall depth h", 100.0, 3000.0, 600.0, key="tee_h")
    bf = c1.number_input("Flange width bf", 100.0, 6000.0, 1200.0, key="tee_bf")
    hf = c2.number_input("Flange depth hf", 50.0, 1000.0, 150.0, key="tee_hf")
elif shape == "Rectangular wall":
    c1, c2 = st.columns(2)
    t_w = c1.number_input("Wall thickness t", 100.0, 1500.0, 200.0, key="wall_t")
    l_w = c2.number_input("Wall length L", 500.0, 20000.0, 2000.0, key="wall_l")
    orient = st.radio(
        "Wall orientation in the section",
        ["In-plane: length vertical (bending about the strong axis)",
         "Out-of-plane: length horizontal (bending about the weak axis)"],
        key="wall_orient",
    )
elif shape == "Circular":
    diam = st.number_input("Diameter D", 200.0, 3000.0, 500.0, key="circ_d")
else:
    st.caption("Concrete outline: vertices in order (x, y) in mm, y pointing up. "
               "Add or delete rows as needed.")
    poly_df = st.data_editor(
        pd.DataFrame({"x": [0.0, 300.0, 300.0, 0.0],
                      "y": [0.0, 0.0, 600.0, 600.0]}),
        num_rows="dynamic", key="poly",
    )

# ----------------------------------------------------------- reinforcement
st.subheader("Reinforcement")
if shape != "Custom (coordinates)":
    c1, c2, c3 = st.columns(3)
    c_nom = c1.number_input("Nominal cover to links (mm)", 10.0, 100.0, 30.0, key="c_nom")
    link = c2.number_input("Link / horizontal bar dia (mm)", 0.0, 25.0, 10.0, key="link")
    dg = c3.number_input("Max aggregate size dg (mm)", 8.0, 40.0, 20.0, key="dg")
    cover_tot = c_nom + link          # cover to the surface of the longitudinal bars
else:
    dg = st.number_input("Max aggregate size dg (mm)", 8.0, 40.0, 20.0, key="dg_c")
    cover_tot = 0.0

if shape in ("Rectangular", "T-section"):
    elem = "Beam"
    if shape == "Rectangular":
        elem = st.radio("Element type for the maximum spacing check",
                        ["Beam", "Slab (one-way, main bars)"], horizontal=True,
                        key="elem")
    st.caption(
        "Bottom layers are counted from the bottom face upwards, top layers from "
        "the top face downwards. The first layer of each group sits at cover + "
        "link dia + bar dia/2 from the face; 'gap' is the clear vertical distance "
        "to the previous layer (ignored for the first layer). Outer bars of every "
        f"layer sit at cover + link dia from the sides. Up to {MAX_LAYERS} layers in total."
        + (" In a T-section, a layer whose centre is inside the flange spreads "
           "across the flange width, otherwise across the web."
           if shape == "T-section" else "")
    )
    st.markdown("**Bottom layers**")
    bottom_df = st.data_editor(
        pd.DataFrame({"n": [4], "dia": [25.0], "gap": [25.0]}),
        num_rows="dynamic", key=f"bot_{shape}", column_config=LAYER_COLS)
    st.markdown("**Top layers**")
    top_df = st.data_editor(
        pd.DataFrame({"n": [3 if shape == "Rectangular" else 6],
                      "dia": [16.0], "gap": [25.0]}),
        num_rows="dynamic", key=f"top_{shape}", column_config=LAYER_COLS)
elif shape == "Rectangular wall":
    st.caption("Vertical bars on each face, given as diameter @ spacing along the "
               "wall length. Number of bars per face = floor((L − 2·(cover + link "
               "+ φ/2)) / spacing) + 1, centred along the wall.")
    c1, c2 = st.columns(2)
    f1_dia = c1.number_input("Face 1 bar dia (mm)", 6, 40, 12, key="f1d")
    f1_s = c1.number_input("Face 1 spacing (mm)", 50, 600, 200, key="f1s")
    f2_dia = c2.number_input("Face 2 bar dia (mm)", 6, 40, 12, key="f2d")
    f2_s = c2.number_input("Face 2 spacing (mm)", 50, 600, 200, key="f2s")
    d_min = st.number_input("Minimum distance between the two layers, centre to "
                            "centre (mm)", 50, 1000, 200, key="dmin")
elif shape == "Circular":
    c1, c2 = st.columns(2)
    n_bar = c1.number_input("Number of bars (equally spaced)", 2, 60, 8, key="nbar")
    dia_bar = c2.number_input("Bar dia (mm)", 6, 40, 25, key="diabar")
else:
    st.caption("Bars: centre coordinates (x, y) and diameter, in mm.")
    bar_df = st.data_editor(
        pd.DataFrame({"x": [50.0, 150.0, 250.0, 50.0, 250.0],
                      "y": [50.0, 50.0, 50.0, 550.0, 550.0],
                      "dia": [25.0, 25.0, 25.0, 16.0, 16.0]}),
        num_rows="dynamic", key="bars",
    )

# ---------------------------------------------------- layout + validation
poly, bars, rows, errors = None, [], [], []

if shape in ("Rectangular", "T-section"):
    if shape == "T-section" and (hf >= h or bf <= b):
        errors.append("T-section: the flange depth must be less than the overall "
                      "depth and the flange width greater than the web width.")
    else:
        if shape == "Rectangular":
            poly = Polygon([(0, 0), (b, 0), (b, h), (0, h)])
        else:
            poly = Polygon([(-b / 2, 0), (b / 2, 0), (b / 2, h - hf),
                            (bf / 2, h - hf), (bf / 2, h), (-bf / 2, h),
                            (-bf / 2, h - hf), (-b / 2, h - hf)])
        bot, top = bottom_df.dropna(), top_df.dropna()
        if len(bot) + len(top) > MAX_LAYERS:
            errors.append(f"A maximum of {MAX_LAYERS} layers is allowed.")
        else:
            max_lim = (min(3 * h, 400.0)
                       if (shape == "Rectangular" and elem.startswith("Slab")) else None)
            bars, rows, errs = build_layers(shape, b, h,
                                            bf if shape == "T-section" else 0.0,
                                            hf if shape == "T-section" else 0.0,
                                            bot, top, cover_tot, dg, max_lim)
            errors += errs
elif shape == "Rectangular wall":
    in_plane = orient.startswith("In-plane")
    if in_plane:
        poly = Polygon([(0, 0), (t_w, 0), (t_w, l_w), (0, l_w)])
    else:
        poly = Polygon([(0, 0), (l_w, 0), (l_w, t_w), (0, t_w)])
    bars, rows, errors = build_wall(
        t_w, l_w, in_plane,
        [("Face 1", float(f1_dia), float(f1_s)), ("Face 2", float(f2_dia), float(f2_s))],
        cover_tot, dg, float(d_min))
elif shape == "Circular":
    poly = Point(0, 0).buffer(diam / 2, 64)
    r_bars = diam / 2 - cover_tot - dia_bar / 2
    if r_bars <= 0:
        errors.append("The cover leaves no room for the bars.")
        poly = None
    else:
        bars = [(r_bars * math.cos(2 * math.pi * k / n_bar),
                 r_bars * math.sin(2 * math.pi * k / n_bar), float(dia_bar), "Ring")
                for k in range(int(n_bar))]
        chord = 2 * r_bars * math.sin(math.pi / n_bar)
        rows.append(make_row("Ring of bars (chord spacing)", int(n_bar),
                             float(dia_bar), chord, dg, None))
else:
    pts = list(poly_df.dropna().itertuples(index=False, name=None))
    cand = Polygon(pts) if len(pts) >= 3 else None
    if cand is None or not cand.is_valid or cand.area <= 0:
        errors.append("The outline needs at least 3 vertices and must not cross itself.")
    else:
        poly = cand
        for k, r in enumerate(bar_df.dropna().itertuples(index=False), 1):
            bars.append((float(r.x), float(r.y), float(r.dia), f"Bar {k}"))
        if not bars:
            errors.append("Add at least one reinforcing bar.")
        else:
            cr_row = custom_row(bars, dg)
            if cr_row:
                rows.append(cr_row)

if poly is not None and bars and not errors:
    errors += check_bars(poly, bars)

if poly is not None and not bars and not errors:
    errors.append("Add at least one reinforcing bar.")

geom_ok = poly is not None and bool(bars) and not errors
n_bars_total = len(bars)
as_total = sum(area(d) for _, _, d, _ in bars)

by_dia = {}
if geom_ok and shape != "Circular":
    for x, y, d, _ in bars:
        by_dia.setdefault(d, ([], []))
        by_dia[d][0].append(x)
        by_dia[d][1].append(y)

# ------------------------------------------------- live preview and checks
st.subheader("Section preview and bar spacing checks")
for e in errors:
    st.error(e)
if poly is not None and bars:
    show(plot_preview(poly, bars,
                      f"{shape}: {n_bars_total} bars, As = {as_total:,.0f} mm²"),
         width=1)
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if any(r["Status"].startswith("⚠️") for r in rows):
        st.warning("Some bar spacings do not meet the EC2 limits below.")
    lim_txt = ("EC2 8.2(2): clear spacing ≥ max(φ, dg + 5 mm, 20 mm). ")
    if shape == "Rectangular wall":
        lim_txt += "EC2 9.6.2(1): vertical bars at most min(3t, 400 mm) apart. "
    elif shape == "Rectangular" and elem.startswith("Slab"):
        lim_txt += "EC2 9.3.1.1(3): main bars at most min(3h, 400 mm) apart. "
    elif shape in ("Rectangular", "T-section"):
        lim_txt += ("Beams: the maximum spacing is governed by crack control "
                    "(EC2 7.3.3), which is checked in the Crack width tab. ")
    st.caption(lim_txt + "Recommended values; check the National Annex.")

# =========================================================== CALCULATION FORM
with st.form("calc"):
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
        ["Rectangular stress block", "Parabola-rectangle"], horizontal=True)

    st.subheader("Stiffness and service stresses")
    c1, c2 = st.columns(2)
    basis = c1.radio("Concrete modulus to use",
                     ["Short-term (Ecm)", "Long-term (modular ratio n)"])
    n_long = c2.number_input("Long-term modular ratio n = Es / Ec,eff", 5.0, 30.0, 15.0)
    crack_basis = st.radio(
        "Tensile strength for the cracked / uncracked decision, fct,eff",
        ["fctm", "fctm,fl = max((1.6 - h/1000)·fctm, fctm)"], horizontal=True)

    st.subheader("Actions")
    st.caption("Ultimate limit state")
    c1, c2 = st.columns(2)
    n_kn = c1.number_input("N_Ed (kN, compression +ve)", -5000.0, 50000.0, 0.0)
    m_ed = c2.number_input("M_Ed (kNm, enter as positive)", 0.0, 50000.0, 150.0)
    st.caption("Service combination (stress limits)")
    c1, c2 = st.columns(2)
    n_ser = c1.number_input("N_ser (kN, compression +ve)", -5000.0, 50000.0, 0.0)
    m_ser = c2.number_input("M_ser (kNm, enter as positive)", 0.0, 50000.0, 100.0)
    st.caption("Quasi-permanent combination (crack width, long term)")
    c1, c2 = st.columns(2)
    n_qp = c1.number_input("N_qp (kN, compression +ve)", -5000.0, 50000.0, 0.0)
    m_qp = c2.number_input("M_qp (kNm, enter as positive)", 0.0, 50000.0, 80.0)
    direction = st.radio(
        "Bending direction",
        ["Sagging (compression at top)", "Hogging (compression at bottom)"],
        horizontal=True)

    st.subheader("Crack width parameters (EC2 7.3.4)")
    c1, c2 = st.columns(2)
    wmax = c1.number_input("Limiting crack width wmax (mm)", 0.1, 0.5, 0.3)
    kt_choice = c1.radio("Load duration",
                         ["Long term (kt = 0.4)", "Short term (kt = 0.6)"])
    k1 = c2.number_input("k1 (0.8 high bond, 1.6 plain bars)", 0.8, 1.6, 0.8)
    k3 = c2.number_input("k3", 1.0, 5.0, 3.4)
    k4 = c2.number_input("k4", 0.1, 1.0, 0.425)
    to_face = st.checkbox("Extrapolate the crack width from the bar level to the "
                          "tension face (strain ratio)", value=False)

    run = st.form_submit_button("Calculate", disabled=not geom_ok)

# ================================================================ RESULTS
if run and geom_ok:
    p = ec2_concrete(CLASSES[cls], acc, gamma_c)
    fyd = fyk / gamma_s
    sagging = direction.startswith("Sagging")
    theta = 0.0 if sagging else math.pi
    dir_txt = "sagging" if sagging else "hogging"
    comp_face = "top" if sagging else "bottom"
    kt = 0.4 if kt_choice.startswith("Long") else 0.6

    long_term = basis.startswith("Long")
    e_lt = es / n_long
    e_service = e_lt if long_term else p["ecm"]
    basis_txt = "Long-term" if long_term else "Short-term"
    h_depth = diam if shape == "Circular" else poly.bounds[3] - poly.bounds[1]

    if crack_basis.startswith("fctm,fl"):
        fct_eff = max((1.6 - h_depth / 1000) * p["fctm"], p["fctm"])
    else:
        fct_eff = p["fctm"]

    # ---------------- materials ----------------
    if ult_model == "Rectangular stress block":
        ult_profile = RectangularStressBlock(
            compressive_strength=p["fcd"],      # fcd already includes αcc
            alpha=p["eta"], gamma=p["lam"], ultimate_strain=p["ecu"])
    else:
        ult_profile = EurocodeParabolicUltimate(
            compressive_strength=p["fcd"], compressive_strain=p["ec2"],
            ultimate_strain=p["ecu"], n=p["n_exp"])

    steel_profile = SteelElasticPlastic(
        yield_strength=fyd, elastic_modulus=es, fracture_strain=eud)
    steel = SteelBar(name="B500", density=7.85e-6,
                     stress_strain_profile=steel_profile, colour="grey")

    def make_section(service_profile):
        """Same geometry, with the chosen concrete service stress-strain profile."""
        conc = Concrete(
            name=cls, density=2.4e-6, stress_strain_profile=service_profile,
            ultimate_stress_strain_profile=ult_profile,
            flexural_tensile_strength=p["fctm"], colour="lightgrey")
        if shape == "Circular":
            g = concrete_circular_section(
                d=diam, area_conc=area(diam), n_conc=64,
                dia_bar=dia_bar, area_bar=area(dia_bar), n_bar=n_bar,
                cover=cover_tot, n_circle=4, conc_mat=conc, steel_mat=steel)
        else:
            g = Geometry(geom=poly, material=conc)
            for d, (xs, ys) in by_dia.items():
                g = add_bars(g, area=area(d), material=steel, x=xs, y=ys, n=4)
        return ConcreteSection(g)

    def run_service(sec_u, e_mod, n_s, m_s):
        """Uncracked analysis first; if the tension exceeds fct,eff, re-run cracked."""
        mk_u = sec_u.moment_curvature_analysis(theta=theta, n=n_s, progress_bar=False)
        s_unc = sec_u.calculate_service_stress(moment_curvature_results=mk_u, m=m_s)
        lo_u, _ = s_unc.get_concrete_stress_limits()
        t_unc = max(0.0, -lo_u)                      # tension is negative
        if t_unc <= fct_eff:
            return s_unc, "UNCRACKED", t_unc
        sec_c = make_section(ConcreteLinearNoTension(elastic_modulus=e_mod))
        mk_c = sec_c.moment_curvature_analysis(theta=theta, n=n_s, progress_bar=False)
        s_cr = sec_c.calculate_service_stress(moment_curvature_results=mk_c, m=m_s)
        return s_cr, "CRACKED", t_unc

    sec = make_section(ConcreteLinear(elastic_modulus=e_service))
    n_ed = n_kn * 1e3  # compression +ve

    try:
        cr = sec.calculate_cracked_properties(theta=theta)
        cr.calculate_transformed_properties(elastic_modulus=e_service)
    except Exception as exc:
        cr = None
        st.warning(f"Elastic cracked properties could not be calculated ({exc!r}).")

    st.caption(f"Reinforcement: {n_bars_total} bars, total As = {as_total:,.0f} mm²")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Section", "Stiffness", "Service stresses", "ULS capacity",
         "M-N diagram", "Crack width"])

    # ---------------- tab 1: section ----------------
    with tab1:
        show(sec.plot_section(render=False).get_figure())
        st.caption(f"fcd = {p['fcd']:.1f} MPa, fyd = {fyd:.0f} MPa, "
                   f"λ = {p['lam']:.3f}, η = {p['eta']:.3f}, "
                   f"εcu3 = {p['ecu']:.4f}, Ecm = {p['ecm']/1e3:.1f} GPa")

    # ---------------- tab 2: stiffness ----------------
    with tab2:

        # Gross transformed section properties.
        # Concrete is active in both compression and tension.
        tg = sec.get_transformed_gross_properties(
            elastic_modulus=e_service
        )
    
        i_hom = float(tg.ixx_c)
        ei_hom = e_service * i_hom
    
        st.markdown(
            f"**{basis_txt} stiffness basis**: "
            f"Ec = {e_service / 1e3:.1f} GPa, "
            f"modular ratio n = Es/Ec = {es / e_service:.1f}"
        )
    
        # ---------------------------------------------------------
        # Standard transformed-section properties
        # ---------------------------------------------------------
        st.subheader("Transformed-section properties")
    
        if cr is None:
            c1, c2 = st.columns(2)
    
            c1.metric(
                "I homogeneous (x10^9 mm4)",
                f"{i_hom / 1e9:.3f}"
            )
    
            c2.metric(
                "EI homogeneous (kNm2)",
                f"{ei_hom * 1e-9:,.0f}"
            )
    
            st.info(
                "The fully cracked pure-bending properties are not "
                "available for this section."
            )
    
        else:
            i_cr = float(cr.iuu_cr)
            ei_cr = e_service * i_cr
    
            c1, c2, c3 = st.columns(3)
    
            c1.metric(
                "I homogeneous (x10^9 mm4)",
                f"{i_hom / 1e9:.3f}"
            )
    
            c2.metric(
                "I cracked, pure bending (x10^9 mm4)",
                f"{i_cr / 1e9:.3f}"
            )
    
            c3.metric(
                "Icr / Ihom",
                f"{i_cr / i_hom:.3f}"
            )
    
            c1, c2, c3 = st.columns(3)
    
            c1.metric(
                "EI homogeneous (kNm2)",
                f"{ei_hom * 1e-9:,.0f}"
            )
    
            c2.metric(
                "EI cracked, pure bending (kNm2)",
                f"{ei_cr * 1e-9:,.0f}"
            )
    
            c3.metric(
                "Mcr (kNm)",
                f"{cr.m_cr / 1e6:,.1f}"
            )
    
            st.metric(
                "Neutral-axis depth, fully cracked pure bending (mm)",
                f"{cr.d_nc:,.1f}"
            )
    
            st.caption(
                "I homogeneous is calculated with the reinforcement "
                "transformed to concrete and the complete concrete section active. "
                "I cracked is the fully cracked transformed-section property for "
                f"{dir_txt} pure bending. Concrete in tension is ignored. "
                "The pure-bending cracked property does not include axial force."
            )
    
        # ---------------------------------------------------------
        # Load-dependent service stiffness
        # ---------------------------------------------------------
        st.divider()
        st.subheader("Effective stiffness at the entered service load")
    
        n_s_eff = n_ser * 1e3
        m_s_eff = m_ser * 1e6
    
        st.caption(
            f"Service load used: Nser = {n_ser:,.1f} kN compression-positive, "
            f"Mser = {m_ser:,.1f} kNm, {dir_txt} bending."
        )
    
        try:
            # First determine whether the section is cracked at the entered
            # service load. This uses concrete with tensile stiffness.
            mk_unc_eff = sec.moment_curvature_analysis(
                theta=theta,
                n=n_s_eff,
                progress_bar=False
            )
    
            s_unc_eff = sec.calculate_service_stress(
                moment_curvature_results=mk_unc_eff,
                m=m_s_eff
            )
    
            lo_unc_eff, _ = s_unc_eff.get_concrete_stress_limits()
            concrete_tension_eff = max(0.0, -float(lo_unc_eff))
    
            service_is_cracked = concrete_tension_eff > fct_eff
    
            if abs(m_s_eff) <= 1e-9:
                st.info(
                    "Mser is zero, so a secant bending stiffness M/kappa "
                    "cannot be calculated."
                )
    
                c1, c2 = st.columns(2)
    
                c1.metric(
                    "Concrete tension from uncracked analysis (MPa)",
                    f"{concrete_tension_eff:.3f}"
                )
    
                c2.metric(
                    "fct,eff (MPa)",
                    f"{fct_eff:.3f}"
                )
    
            elif not service_is_cracked:
                # For the linear uncracked model, the effective bending
                # inertia is the transformed homogeneous inertia.
                i_eff = i_hom
                ei_eff = ei_hom
    
                c1, c2, c3 = st.columns(3)
    
                c1.metric(
                    "I effective @ service (x10^9 mm4)",
                    f"{i_eff / 1e9:.3f}"
                )
    
                c2.metric(
                    "EI effective @ service (kNm2)",
                    f"{ei_eff * 1e-9:,.0f}"
                )
    
                c3.metric(
                    "Ieff / Ihom",
                    f"{i_eff / i_hom:.3f}"
                )
    
                c1, c2 = st.columns(2)
    
                c1.metric(
                    "Concrete tension (MPa)",
                    f"{concrete_tension_eff:.3f}"
                )
    
                c2.metric(
                    "fct,eff (MPa)",
                    f"{fct_eff:.3f}"
                )
    
                st.success(
                    "The section is uncracked at the entered service load. "
                    "Therefore Ieff = Ihom for the selected linear-elastic model."
                )
    
            else:
                # Re-run with no concrete tensile resistance.
                sec_eff = make_section(
                    ConcreteLinearNoTension(
                        elastic_modulus=e_service
                    )
                )
    
                mk_eff = sec_eff.moment_curvature_analysis(
                    theta=theta,
                    n=n_s_eff,
                    progress_bar=False
                )
    
                moments_eff = np.asarray(mk_eff.m_xy, dtype=float)
    
                if moments_eff.size == 0:
                    raise ValueError(
                        "The moment-curvature analysis returned no results."
                    )
    
                m_min = float(np.min(moments_eff))
                m_max = float(np.max(moments_eff))
    
                if m_s_eff < m_min - 1e-6 or m_s_eff > m_max + 1e-6:
                    raise ValueError(
                        f"Mser = {m_ser:,.1f} kNm is outside the generated "
                        f"moment-curvature range of "
                        f"{m_min / 1e6:,.1f} to {m_max / 1e6:,.1f} kNm."
                    )
    
                # Use the library interpolation method instead of selecting
                # the nearest analysis point.
                kappa_eff = float(
                    mk_eff.get_curvature(moment=m_s_eff)
                )
    
                if not np.isfinite(kappa_eff):
                    raise ValueError(
                        "The interpolated service curvature is not finite."
                    )
    
                if abs(kappa_eff) <= 1e-16:
                    raise ValueError(
                        "The calculated service curvature is too close to zero "
                        "to determine M/kappa."
                    )
    
                # Secant stiffness at the entered service load:
                #
                # EIeff = M / kappa
                # Ieff  = EIeff / Ec
                #
                # Absolute values are used because the sign is controlled by
                # the selected bending direction.
                ei_eff = abs(m_s_eff / kappa_eff)
                i_eff = ei_eff / e_service
    
                c1, c2, c3 = st.columns(3)
    
                c1.metric(
                    "I effective @ service (x10^9 mm4)",
                    f"{i_eff / 1e9:.3f}"
                )
    
                c2.metric(
                    "EI effective @ service (kNm2)",
                    f"{ei_eff * 1e-9:,.0f}"
                )
    
                c3.metric(
                    "Ieff / Ihom",
                    f"{i_eff / i_hom:.3f}"
                )
    
                c1, c2, c3 = st.columns(3)
    
                c1.metric(
                    "Curvature @ service (1/mm)",
                    f"{abs(kappa_eff):.4e}"
                )
    
                c2.metric(
                    "Concrete tension before cracking (MPa)",
                    f"{concrete_tension_eff:.3f}"
                )
    
                c3.metric(
                    "fct,eff (MPa)",
                    f"{fct_eff:.3f}"
                )
    
                # -------------------------------------------------
                # Determine zero-strain axis from reinforcement strain
                # -------------------------------------------------
                try:
                    s_eff = sec_eff.calculate_service_stress(
                        moment_curvature_results=mk_eff,
                        m=m_s_eff
                    )
    
                    df_eff = bar_stress_table(s_eff)
    
                    y_bar = df_eff["y (mm)"].to_numpy(dtype=float)
                    steel_strain = (
                        df_eff["Stress (MPa)"].to_numpy(dtype=float) / es
                    )
    
                    y_range = float(np.ptp(y_bar))
    
                    if y_range <= 1e-6:
                        st.caption(
                            "The service zero-strain axis cannot be determined "
                            "because all reinforcement bars are at the same "
                            "vertical level."
                        )
    
                    else:
                        strain_slope, strain_intercept = np.polyfit(
                            y_bar,
                            steel_strain,
                            1
                        )
    
                        if abs(strain_slope) <= 1e-16:
                            st.caption(
                                "The service zero-strain axis cannot be determined "
                                "because the fitted strain gradient is too small."
                            )
    
                        else:
                            y_na_eff = -strain_intercept / strain_slope
    
                            y_bottom = float(poly.bounds[1])
                            y_top = float(poly.bounds[3])
    
                            if sagging:
                                na_depth_eff = y_top - y_na_eff
                            else:
                                na_depth_eff = y_na_eff - y_bottom
    
                            st.metric(
                                "Zero-strain-axis depth @ service "
                                f"from {comp_face} face (mm)",
                                f"{na_depth_eff:,.1f}"
                            )
    
                            if y_na_eff < y_bottom or y_na_eff > y_top:
                                st.caption(
                                    "The zero-strain axis lies outside the concrete "
                                    "section. This can occur when axial compression "
                                    "or tension causes the entire section to have "
                                    "the same strain sign."
                                )
                            else:
                                st.caption(
                                    "The service zero-strain-axis position is "
                                    "obtained from a linear fit through the calculated "
                                    "reinforcement strains. It is load-dependent and "
                                    "is not the same as the fully cracked pure-bending "
                                    "neutral axis."
                                )
    
                except Exception as na_exc:
                    st.caption(
                        "The effective stiffness was calculated, but the service "
                        "zero-strain axis could not be extracted: "
                        f"{na_exc!r}"
                    )
    
                st.success(
                    "The section is cracked at the entered service load. "
                    "Ieff is the secant stiffness Mser/kappa from the cracked "
                    "no-tension moment-curvature analysis, including Nser."
                )
    
                if cr is not None:
                    st.caption(
                        f"For comparison: Icr/Ihom = {i_cr / i_hom:.3f} "
                        f"for fully cracked pure bending, while "
                        f"Ieff/Ihom = {i_eff / i_hom:.3f} at the entered "
                        "service axial force and moment."
                    )
    
        except Exception as exc:
            st.warning(
                "The load-dependent service stiffness could not be calculated. "
                f"Reason: {exc!r}"
            )

    # ---------------- tab 3: service stresses ----------------
    with tab3:
        st.markdown(f"**{basis_txt} basis**: Ec = {e_service / 1e3:.1f} GPa "
                    f"(n = {es / e_service:.1f}), M_ser = {m_ser:,.0f} kNm, "
                    f"N_ser = {n_ser:,.0f} kN, {dir_txt}")
        n_s, m_s = n_ser * 1e3, m_ser * 1e6
        t_unc, sres = None, None
        with st.spinner("Running service analysis..."):
            try:
                sres, state, t_unc = run_service(sec, e_service, n_s, m_s)
                method = ("Uncracked section (concrete carries tension)"
                          if state == "UNCRACKED"
                          else "Cracked section (concrete resists no tension)")
                method += ", including N_ser"
            except Exception as exc:
                if cr is not None:
                    st.warning(f"The service analysis with axial force failed "
                               f"({exc!r}). Showing the elastic cracked bending-only "
                               "result instead; N_ser is NOT included.")
                    sres = sec.calculate_cracked_stress(cracked_results=cr, m=m_s)
                    state = "CRACKED (assumed)"
                    method = "Elastic cracked section, bending only (N_ser ignored)"
                else:
                    st.error(f"The service analysis failed ({exc!r}).")
        if sres is not None:
            if t_unc is not None:
                st.markdown(f"**Section state at service: {state}.** Largest concrete "
                            f"tension in the uncracked analysis = {t_unc:.2f} MPa, "
                            f"fct,eff = {fct_eff:.2f} MPa.")
            else:
                st.markdown(f"**Section state at service: {state}.**")
            lo, hi = sres.get_concrete_stress_limits()
            sig_c, t_conc = max(hi, 0.0), max(-lo, 0.0)
            df_ser = bar_stress_table(sres)
            worst = df_ser.loc[df_ser["Stress (MPa)"].abs().idxmax()]
            sig_s = abs(float(worst["Stress (MPa)"]))
            fck = CLASSES[cls]
            c1, c2, c3 = st.columns(3)
            c1.metric("Max concrete compression (MPa)", f"{sig_c:.1f}")
            c1.caption(f"EC2 7.2(2): 0.6·fck = {0.6 * fck:.1f} MPa "
                       f"(ratio {sig_c / (0.6 * fck):.2f}), characteristic. "
                       f"EC2 7.2(3): 0.45·fck = {0.45 * fck:.1f} MPa "
                       f"(ratio {sig_c / (0.45 * fck):.2f}), quasi-permanent.")
            c2.metric("Max concrete tension (MPa)", f"{t_conc:.2f}")
            c3.metric("Max steel stress (MPa)", f"{sig_s:.0f}")
            c3.caption(f"Bar {int(worst['Bar'])}. EC2 7.2(5): 0.8·fyk = "
                       f"{0.8 * fyk:.0f} MPa (ratio {sig_s / (0.8 * fyk):.2f}).")
            show(sres.plot_stress(render=False).get_figure())
            show(
                plot_bar_stresses(
                    sec,
                    df_ser,
                    "Reinforcement stresses (service)",
                    fig_width=10,
                    fig_height=5,
                    annotation_font=5,
                    axis_font=7,
                    title_font=8,
                ),
                width=0.60,
            )

    # ---------------- tab 4: ULS capacity ----------------
    with tab4:
        res = None
        try:
            res = sec.ultimate_bending_capacity(theta=theta, n=n_ed)
        except Exception as exc:
            st.error(f"The ultimate capacity could not be calculated ({exc!r}).")
        if res is not None:
            m_rd = abs(res.m_xy) / 1e6
            c1, c2, c3 = st.columns(3)
            c1.metric("M_Rd (kNm)", f"{m_rd:,.1f}")
            c2.metric("M_Ed (kNm)", f"{m_ed:,.1f}")
            c3.metric("Utilisation", f"{m_ed / m_rd:.2f}" if m_rd > 0 else "n/a")
            d_n = getattr(res, "d_n", None)
            extra = f", neutral axis depth d_n = {d_n:,.0f} mm" if d_n else ""
            st.caption(f"{dir_txt.capitalize()} bending, compression at {comp_face}, "
                       f"with N_Ed = {n_kn:,.0f} kN{extra}. Concrete: {ult_model}. "
                       "Steel: elastic-plastic.")
            uls = sec.calculate_ultimate_stress(ultimate_results=res)
            st.subheader("Stresses at the ultimate limit state")
            show(uls.plot_stress(render=False).get_figure())
            df_uls = bar_stress_table(uls)
            show(
                plot_bar_stresses(
                    sec,
                    df_uls,
                    "Reinforcement stresses at ULS",
                    fig_width=4.5,
                    fig_height=3.5,
                    annotation_font=5,
                    axis_font=7,
                    title_font=8,
                ),
                width=0.60,
            )
            st.dataframe(df_uls.round(1), hide_index=True)
            st.caption("The stresses shown are those of the section at its "
                       "ultimate capacity M_Rd for the applied axial force, not "
                       "at M_Ed. Compression is positive.")
            with st.expander("Stress-strain curves used"):
                try:
                    show(ult_profile.plot_stress_strain(
                        title=f"Concrete ultimate: {ult_model}", render=False
                    ).get_figure())
                    show(steel_profile.plot_stress_strain(
                        title="Steel: elastic-plastic", render=False).get_figure())
                except Exception as exc:
                    st.info(f"The curves could not be plotted ({exc!r}).")

    # ---------------- tab 5: M-N diagram ----------------
    with tab5:
        fig, ax = plt.subplots()
        ends = {}
        for th, lab, style in [(0.0, "Sagging", "-b"), (math.pi, "Hogging", "-g")]:
            mi = None
            try:
                try:
                    mi = sec.moment_interaction_diagram(
                        theta=th, n_points=48, progress_bar=False)
                except Exception:
                    # retry without the extremely thin compression zone at the
                    # tension end of the default limits
                    d_lim = max(1.0, 0.005 * h_depth)
                    mi = sec.moment_interaction_diagram(
                        theta=th, n_points=48,
                        limits=[("D", 1.0), ("d_n", d_lim)], progress_bar=False)
            except Exception as exc:
                st.warning(f"The {lab.lower()} M-N curve could not be calculated "
                           f"({exc!r}). If this keeps happening, choose Python 3.12 "
                           "in the app's advanced settings on Streamlit Cloud.")
            if mi is None:
                continue
            n_list, m_list = mi.get_results_lists(moment="m_x")
            ms = [m / 1e6 for m in m_list]          # keep the real sign
            if (th == 0.0 and sum(ms) < 0) or (th != 0.0 and sum(ms) > 0):
                ms = [-m for m in ms]
            ns = [n / 1e3 for n in n_list]
            ax.plot(ms, ns, style, label=lab)
            ends[lab] = ((ms[0], ns[0]), (ms[-1], ns[-1]))
        if len(ends) == 2:                           # close the diagram
            for k in (0, 1):
                ax.plot([ends["Sagging"][k][0], ends["Hogging"][k][0]],
                        [ends["Sagging"][k][1], ends["Hogging"][k][1]],
                        "k--", lw=0.8)
        applied_sign = 1 if sagging else -1
        ax.plot(applied_sign * m_ed, n_kn, "ro", label="(M_Ed, N_Ed)")
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("M (kNm): sagging +ve, hogging -ve")
        ax.set_ylabel("N (kN), compression +ve")
        ax.grid(True)
        ax.legend()
        show(fig)

    # ---------------- tab 6: crack width ----------------
    with tab6:
        st.markdown("**Crack width to EN 1992-1-1 7.3.4: long-term, "
                    "quasi-permanent combination**")
        if shape == "Circular":
            st.info("The crack width check is not provided for circular sections.")
        else:
            cres, t_q = None, None
            with st.spinner("Running the quasi-permanent service analysis..."):
                try:
                    sec_lt = (sec if abs(e_service - e_lt) < 1e-6
                              else make_section(ConcreteLinear(elastic_modulus=e_lt)))
                    s_q, state_q, t_q = run_service(
                        sec_lt, e_lt, n_qp * 1e3, m_qp * 1e6)
                    if state_q == "UNCRACKED":
                        cres = {"status": "uncracked"}
                    else:
                        df_q = bar_stress_table(s_q)
                        cres = ec2_crack(df_q, poly, sagging, es, p["ecm"],
                                         p["fctm"], kt, k1, k3, k4, to_face)
                except Exception as exc:
                    st.error(f"The crack width analysis failed ({exc!r}).")
            if cres is not None:
                if cres["status"] == "uncracked":
                    st.success(f"Uncracked under the quasi-permanent combination "
                               f"(concrete tension {t_q:.2f} MPa ≤ fct,eff = "
                               f"{fct_eff:.2f} MPa). No crack width check is needed.")
                elif cres["status"] == "error":
                    st.error(cres["msg"])
                else:
                    rws = cres["rows"]
                    wk_gov = max(r["wk"] for r in rws)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Governing crack width wk (mm)", f"{wk_gov:.3f}")
                    c2.metric("Limit wmax (mm)", f"{wmax:.2f}")
                    c3.metric("wk / wmax", f"{wk_gov / wmax:.2f}")
                    if wk_gov <= wmax:
                        st.success("Crack width limit satisfied.")
                    else:
                        st.error("Crack width limit EXCEEDED.")
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
                        f"x = {cres['x']:.0f} mm, h − d = {cres['d_t']:.0f} mm, "
                        f"hc,eff = {cres['hc_eff']:.0f} mm, Ac,eff = "
                        f"{cres['ac_eff']:,.0f} mm², ρp,eff = {cres['rho']:.4f}, "
                        f"φeq = {cres['phi_eq']:.1f} mm, k2 = {cres['k2']:.2f}, "
                        f"αe = Es/Ecm = {cres['alpha_e']:.2f}, kt = {kt}, "
                        f"fct,eff = {p['fctm']:.2f} MPa, k1 = {k1}, k3 = {k3}, "
                        f"k4 = {k4}. σs from the cracked section analysis at the "
                        f"long-term modular ratio n = {n_long:.1f}. Layers are listed "
                        "from the tension face inwards.")
                    ext = rws[0]
                    lim73 = max_spacing_73n(ext["sigma"], wmax)
                    if math.isinf(ext["spacing"]):
                        st.caption("EC2 7.3.3 (Table 7.3N): not applicable, the "
                                   "outer tension layer has a single bar.")
                    elif lim73 is None:
                        st.caption(f"EC2 7.3.3 (Table 7.3N): σs = {ext['sigma']:.0f} MPa "
                                   f"and wmax = {wmax} mm are outside the table.")
                    else:
                        ok73 = ext["spacing"] <= lim73 + 1e-6
                        st.caption(f"EC2 7.3.3 (Table 7.3N), outer tension layer: "
                                   f"spacing {ext['spacing']:.0f} mm vs maximum "
                                   f"{lim73} mm at σs = {ext['sigma']:.0f} MPa → "
                                   + ("✅ OK" if ok73 else "⚠️ exceeds the table value")
                                   + " (table assumes c = 25 mm and fct,eff = 2.9 MPa).")
                    show(plot_preview(
                        poly, cres["ten_bars"], "Ac,eff (yellow) and tension bars (red)",
                        band=cres["band"]))
elif run and not geom_ok:
    st.error("Fix the section errors above before calculating.")
