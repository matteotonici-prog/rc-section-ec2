import io

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

PLOT_WIDTH = 0.8   # fraction of the page width used by each plot (1.0 = full width)

# Soft palette: blue = tension, red = compression, green = OK
TENSION = "#6FA3D8"
COMPRESSION = "#E58A8A"
GREEN = "#7BC8A4"
GOLD = "#F6D98B"
GREY = "#9AA5B1"
CONCRETE = "#E3E8EE"
EDGE = "#7C8896"
INK = "#24364B"
BAR = "#8FA3B8"

SOFT_CMAP = LinearSegmentedColormap.from_list(
    "soft_stress", [TENSION, "#F7F9FC", COMPRESSION])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#FBFCFE",
    "axes.edgecolor": "#C9D3DF", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "grid.color": "#DDE5EE", "legend.frameon": False,
})


def apply_theme():
    st.markdown("""
<style>
[data-testid="stMetric"]{background:#F4F8FC;border:1px solid #DCE6F2;
  border-radius:12px;padding:10px 14px;}
[data-testid="stMetricLabel"] p{font-size:0.78rem;color:#4A5D73;}
.stTabs [data-baseweb="tab-list"]{gap:6px;}
.stTabs [data-baseweb="tab"]{background:#EAF1FA;border-radius:10px 10px 0 0;
  padding:6px 14px;}
.stTabs [aria-selected="true"]{background:#CFE0F5;}
h1,h2,h3{color:#24364B;}
</style>""", unsafe_allow_html=True)


def to_png(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def show_png(png, width=PLOT_WIDTH):
    if width >= 1.0:
        st.image(png)
    else:
        col, _ = st.columns([width, 1 - width])
        with col:
            st.image(png)


def show_fig(fig, width=PLOT_WIDTH):
    show_png(to_png(fig), width)


def soften(fig):
    """Best effort: apply the soft colour map to the library's stress plots."""
    for ax in fig.axes:
        for c in list(ax.collections):
            try:
                if c.get_array() is not None:
                    c.set_cmap(SOFT_CMAP)
            except Exception:
                pass
    return fig


def plot_preview(poly, bars, title, band=None):
    fig, ax = plt.subplots()
    for g in getattr(poly, "geoms", [poly]):
        xs, ys = g.exterior.xy
        ax.fill(xs, ys, facecolor=CONCRETE, edgecolor=EDGE, zorder=1)
    if band is not None and not band.is_empty:
        for g in getattr(band, "geoms", [band]):
            if g.geom_type == "Polygon":
                xs, ys = g.exterior.xy
                ax.fill(xs, ys, facecolor=GOLD, alpha=0.6, edgecolor="none", zorder=2)
    for x, y, d, lab in bars:
        ax.add_patch(Circle((x, y), d / 2,
                            facecolor=TENSION if lab == "T" else BAR,
                            edgecolor=INK, linewidth=0.5, zorder=5))
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.05)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=9)
    return fig


def plot_bar_stresses(sec, df, title, fig_width=12, fig_height=12,
                      annotation_font=18, axis_font=18, title_font=18):
    """Bars drawn to scale and coloured by stress (blue = tension, red = compression)."""
    ax = sec.plot_section(render=False)
    fig = ax.get_figure()
    fig.set_size_inches(fig_width, fig_height)
    if df.empty:
        ax.set_title(title, fontsize=title_font)
        return fig
    vmax = max(float(df["Stress (MPa)"].abs().max()), 1e-6)
    norm = plt.Normalize(-vmax, vmax)
    for _, r in df.iterrows():
        stress = float(r["Stress (MPa)"])
        ax.add_patch(Circle((r["x (mm)"], r["y (mm)"]), r["Dia (mm)"] / 2,
                            facecolor=SOFT_CMAP(norm(stress)), edgecolor=INK,
                            linewidth=0.4, zorder=5))
        ax.annotate(f'{int(r["Bar"])}: {stress:.0f}', (r["x (mm)"], r["y (mm)"]),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    va="bottom", fontsize=annotation_font, zorder=6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SOFT_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Bar stress (MPa), +ve compression", fontsize=axis_font)
    cb.ax.tick_params(labelsize=max(axis_font - 1, 1))
    ax.set_title(title, fontsize=title_font)
    ax.set_xlabel("x (mm)", fontsize=axis_font)
    ax.set_ylabel("y (mm)", fontsize=axis_font)
    ax.tick_params(axis="both", labelsize=max(axis_font - 1, 1))
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.08)
    fig.tight_layout()
    return fig


def plot_mn(curves, m_ed, n_kn, sagging):
    fig, ax = plt.subplots()
    colours = {"Sagging": TENSION, "Hogging": GREEN}
    for lab, (ms, ns) in curves.items():
        ax.plot(ms, ns, color=colours[lab], lw=1.8, label=lab)
    if len(curves) == 2:                        # close the diagram
        a, b = curves["Sagging"], curves["Hogging"]
        for k in (0, -1):
            ax.plot([a[0][k], b[0][k]], [a[1][k], b[1][k]],
                    color=GREY, ls="--", lw=0.8)
    sign = 1 if sagging else -1
    ax.plot(sign * m_ed, n_kn, "o", color=COMPRESSION, ms=7, label="(M_Ed, N_Ed)")
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("M (kNm): sagging +ve, hogging -ve")
    ax.set_ylabel("N (kN), compression +ve")
    ax.grid(True)
    ax.legend()
    return fig
    
def plot_biaxial(mxs, mys, dx, dy, cap_pt, n_kn):
    """Biaxial capacity diagram at N_Ed with the design point (kNm)."""
    fig, ax = plt.subplots()
    xs, ys = list(mxs) + [mxs[0]], list(mys) + [mys[0]]
    ax.fill(xs, ys, color=TENSION, alpha=0.18)
    ax.plot(xs, ys, color=TENSION, lw=1.8, label=f"Capacity at N = {n_kn:,.0f} kN")
    ax.plot([0, cap_pt[0]], [0, cap_pt[1]], color=GREY, ls="--", lw=0.9)
    ax.plot(dx, dy, "o", color=COMPRESSION, ms=7, label="(M_Ed,x, M_Ed,y)")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("M_x (kNm)")
    ax.set_ylabel("M_y (kNm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend()
    return fig
