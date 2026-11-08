import math
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from concreteproperties import Concrete, ConcreteSection
from concreteproperties.stress_strain_profile import (
    ConcreteLinear, ConcreteLinearNoTension,
)
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library import concrete_circular_section
from sectionproperties.pre.library.concrete_sections import add_bars
from shapely.geometry import box

from core.geometry import area
from core.materials import CLASSES, ec2_concrete, make_profiles
from core.plots import (
    plot_bar_stresses, plot_biaxial, plot_mn, plot_preview, soften, to_png,
)
from core.shear import auto_shear_dims, shear_check

# EN 1992-1-1 Table 7.3N: max bar spacing (mm) for wk = 0.4 / 0.3 / 0.2 mm
TABLE_73N = {160: (300, 300, 200), 200: (300, 250, 150), 240: (250, 200, 100),
             280: (200, 150, 50), 320: (150, 100, None), 360: (100, 50, None)}


def max_spacing_73n(sigma_s, wmax):
    """Table 7.3N: conservative lookup (next tabulated stress, stricter wk column)."""
    col = 0 if wmax >= 0.4 else (1 if wmax >= 0.3 else (2 if wmax >= 0.2 else None))
    if col is None:
        return None
    for s in sorted(TABLE_73N):
        if sigma_s <= s:
            return TABLE_73N[s][col]
    return None


@contextmanager
def timed(R, name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        R["time"][name] = R["time"].get(name, 0.0) + time.perf_counter() - t0


def make_section_factory(lay, g, ult_profile, steel, p, cls):
    """Returns make_section(service_profile) building the ConcreteSection."""
    def make_section(service_profile):
        conc = Concrete(
            name=cls, density=2.4e-6, stress_strain_profile=service_profile,
            ultimate_stress_strain_profile=ult_profile,
            flexural_tensile_strength=p["fctm"], colour="lightgrey")
        if g["shape"] == "Circular":
            gm = concrete_circular_section(
                d=g["diam"], area_conc=area(g["diam"]), n_conc=64,
                dia_bar=float(g["dia_bar"]), area_bar=area(float(g["dia_bar"])),
                n_bar=int(g["n_bar"]), cover=g["cover_tot"], n_circle=4,
                conc_mat=conc, steel_mat=steel)
        else:
            gm = Geometry(geom=lay["poly"], material=conc)
            for d, (xs, ys) in lay["by_dia"].items():
                gm = add_bars(gm, area=area(d), material=steel, x=xs, y=ys, n=4)
        return ConcreteSection(gm)
    return make_section


def build_ctx(lay, g, mat):
    """Everything that depends only on geometry and materials."""
    p = ec2_concrete(CLASSES[mat["cls"]], mat["acc"], mat["gamma_c"])
    es = float(mat["es"])
    fyd = mat["fyk"] / mat["gamma_s"]
    e_lt = es / mat["n_long"]
    e_service = e_lt if mat["basis_long"] else p["ecm"]
    fct_eff = (max((1.6 - lay["h_depth"] / 1000) * p["fctm"], p["fctm"])
               if mat["crack_basis_fl"] else p["fctm"])
    ult_profile, steel_profile, steel = make_profiles(
        p, fyd, es, mat["eud"], mat["ult_model"])
    make_section = make_section_factory(lay, g, ult_profile, steel, p, mat["cls"])
    sec = make_section(ConcreteLinear(elastic_modulus=e_service))
    return dict(p=p, es=es, fyd=fyd, e_lt=e_lt, e_service=e_service,
                fct_eff=fct_eff, ult_profile=ult_profile,
                steel_profile=steel_profile, steel=steel,
                make_section=make_section, sec=sec, fck=CLASSES[mat["cls"]],
                mat=mat, h_depth=lay["h_depth"],
                basis_txt="Long-term" if mat["basis_long"] else "Short-term")


def bar_stress_table(sres):
    """One row per reinforcing bar (numeric, unrounded)."""
    rows = []
    for i, (gm, s) in enumerate(
        zip(sres.lumped_reinforcement_geometries,
            sres.lumped_reinforcement_stresses), start=1
    ):
        cx, cy = gm.calculate_centroid()
        a = gm.calculate_area()
        s = float(s)
        rows.append({
            "Bar": i, "x (mm)": cx, "y (mm)": cy,
            "Dia (mm)": math.sqrt(4 * a / math.pi),
            "Stress (MPa)": s, "Force (kN)": s * a / 1e3,
            "State": "compression" if s > 1e-6 else ("tension" if s < -1e-6 else "unstressed"),
        })
    return pd.DataFrame(rows)


def homogeneous_props(sec, lay, e_service, es):
    """Transformed, uncracked section: area, centroid and inertia."""
    tg = sec.get_transformed_gross_properties(elastic_modulus=e_service)
    alpha = es / e_service
    a_s = sum(area(d) for _, _, d, _ in lay["bars"])
    s_y = sum(area(d) * y for _, y, d, _ in lay["bars"])
    a_c = lay["conc_area"]
    a_hom = a_c + (alpha - 1.0) * a_s
    y_hom = (a_c * lay["conc_cy"] + (alpha - 1.0) * s_y) / a_hom
    i_hom = float(tg.ixx_c)
    return dict(alpha=alpha, a_c=a_c, a_s=a_s, a_hom=a_hom,
                y_hom=y_hom - lay["y_min"], i_hom=i_hom, ei_hom=e_service * i_hom)


def cracked_props(sec, e_service, theta):
    cr = sec.calculate_cracked_properties(theta=theta)
    cr.calculate_transformed_properties(elastic_modulus=e_service)
    i_cr = float(cr.iuu_cr)
    return cr, dict(i_cr=i_cr, ei_cr=e_service * i_cr,
                    m_cr=float(cr.m_cr) / 1e6, d_nc=float(cr.d_nc))


def geo_results(ctx, lay):
    """Automatic section results: homogeneous and cracked (pure bending)."""
    out = {"err": {}}
    sec, es, e = ctx["sec"], ctx["es"], ctx["e_service"]
    try:
        out["hom"] = homogeneous_props(sec, lay, e, es)
    except Exception as exc:
        out["err"]["hom"] = repr(exc)
    for key, th in (("sag", 0.0), ("hog", math.pi)):
        try:
            out[key] = cracked_props(sec, e, th)[1]
        except Exception as exc:
            out["err"][key] = repr(exc)
    return out


# ------------------------------------------------------------ service
class ServiceRunner:
    """Runs each moment-curvature analysis and service stress only once."""

    def __init__(self, make_section, theta, sec_main, e_main, log):
        self.make_section, self.theta = make_section, theta
        self.sec_main, self.e_main, self.log = sec_main, e_main, log
        self._mk, self._st = {}, {}

    def mk(self, cracked, e_mod, n_s):
        key = (bool(cracked), round(float(e_mod), 1), round(float(n_s), 0))
        if key not in self._mk:
            if cracked:
                s = self.make_section(ConcreteLinearNoTension(elastic_modulus=e_mod))
            elif abs(e_mod - self.e_main) < 1e-6:
                s = self.sec_main
            else:
                s = self.make_section(ConcreteLinear(elastic_modulus=e_mod))
            t0 = time.perf_counter()
            res = s.moment_curvature_analysis(theta=self.theta, n=n_s,
                                              progress_bar=False)
            self.log.append(
                f"{'cracked' if cracked else 'uncracked'} moment-curvature, "
                f"N = {n_s / 1e3:,.0f} kN, Ec = {e_mod / 1e3:.1f} GPa: "
                f"{time.perf_counter() - t0:.1f} s")
            self._mk[key] = (s, res)
        return self._mk[key]

    def stress(self, cracked, e_mod, n_s, m_s):
        key = (bool(cracked), round(float(e_mod), 1), round(float(n_s), 0),
               round(float(m_s), 0))
        if key not in self._st:
            s, res = self.mk(cracked, e_mod, n_s)
            self._st[key] = s.calculate_service_stress(
                moment_curvature_results=res, m=m_s)
        return self._st[key]


def run_service(runner, fct_eff, e_mod, n_s, m_s):
    """Uncracked first; if the tension exceeds fct,eff use the cracked analysis."""
    s_unc = runner.stress(False, e_mod, n_s, m_s)
    lo_u, _ = s_unc.get_concrete_stress_limits()
    t_unc = max(0.0, -lo_u)                          # tension is negative
    if t_unc <= fct_eff:
        return s_unc, "UNCRACKED", t_unc
    return runner.stress(True, e_mod, n_s, m_s), "CRACKED", t_unc


def effective_stiffness(runner, e_service, i_hom, fct_eff, es, poly, sagging,
                        n_s, m_s):
    """Secant stiffness M/kappa at the service load, including the axial force."""
    out = {}
    if abs(m_s) <= 1e-9:
        out.update(kind="zero", t_conc=0.0)
        return out
    s_u = runner.stress(False, e_service, n_s, m_s)
    lo_u, _ = s_u.get_concrete_stress_limits()
    out["t_conc"] = max(0.0, -float(lo_u))
    if out["t_conc"] <= fct_eff:
        out.update(kind="uncracked", i_eff=i_hom, ei_eff=e_service * i_hom)
        return out

    _, mk = runner.mk(True, e_service, n_s)
    moments = np.asarray(mk.m_xy, dtype=float)
    if moments.size == 0:
        raise ValueError("The moment-curvature analysis returned no results.")
    m_min, m_max = float(moments.min()), float(moments.max())
    if m_s < m_min - 1e-6 or m_s > m_max + 1e-6:
        raise ValueError(f"M_ser is outside the generated moment-curvature range "
                         f"({m_min / 1e6:,.1f} to {m_max / 1e6:,.1f} kNm).")
    kappa = float(mk.get_curvature(moment=m_s))
    if not np.isfinite(kappa) or abs(kappa) <= 1e-16:
        raise ValueError("The service curvature is not valid.")
    ei = abs(m_s / kappa)
    out.update(kind="cracked", i_eff=ei / e_service, ei_eff=ei, kappa=abs(kappa),
               na_depth=None, na_outside=False, na_msg=None)
    try:                                  # zero-strain axis from the bar strains
        s_eff = runner.stress(True, e_service, n_s, m_s)
        df = bar_stress_table(s_eff)
        y = df["y (mm)"].to_numpy(float)
        eps = df["Stress (MPa)"].to_numpy(float) / es
        if np.ptp(y) <= 1e-6:
            out["na_msg"] = "All bars are at one level: the neutral axis cannot be found."
        else:
            slope, icpt = np.polyfit(y, eps, 1)
            if abs(slope) <= 1e-16:
                out["na_msg"] = "The strain gradient is too small to find the neutral axis."
            else:
                y_na = -icpt / slope
                y_bot, y_top = poly.bounds[1], poly.bounds[3]
                out["na_depth"] = (y_top - y_na) if sagging else (y_na - y_bot)
                out["na_outside"] = bool(y_na < y_bot or y_na > y_top)
    except Exception as exc:
        out["na_msg"] = f"Neutral axis not extracted: {exc!r}"
    return out


def ec2_crack(df, poly, sagging, es, alpha_e, fct_eff, kt, k1, k3, k4, to_face):
    """EC2 7.3.4 crack width for the tension layers, from cracked-analysis bar stresses."""
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
            spacing=spacing, wide=wide, sr=sr, diff=diff, wk=sr * diff * factor))
    ten_bars = [(r["x (mm)"], r["y (mm)"], r["Dia (mm)"], "T")
                for _, r in ten.iterrows()]
    return dict(status="ok", rows=rows, x=x, h_tot=h_tot, hc_eff=hc_eff,
                ac_eff=ac_eff, rho=rho, phi_eq=phi_eq, k2=k2, alpha_e=alpha_e,
                band=band, d_t=d_t, ten_bars=ten_bars)


def run_sls(ctx, lay, g, act):
    """SLS check: stiffness at service load, stresses and crack width."""
    mat = ctx["mat"]
    sec, p, es = ctx["sec"], ctx["p"], ctx["es"]
    e_service, e_lt, fct_eff = ctx["e_service"], ctx["e_lt"], ctx["fct_eff"]
    sagging = bool(act["sagging"])
    theta = 0.0 if sagging else math.pi
    R = {"err": {}, "img": {}, "time": {}, "mk_log": [], "act": act, "mat": mat,
         "p": p, "fck": ctx["fck"], "es": es, "e_service": e_service,
         "fct_eff": fct_eff, "basis_txt": ctx["basis_txt"], "sagging": sagging}
    runner = ServiceRunner(ctx["make_section"], theta, sec, e_service, R["mk_log"])
    n_s, m_s = act["n_ser"] * 1e3, act["m_ser"] * 1e6

    cr = None
    try:
        with timed(R, "homogeneous properties"):
            R["hom"] = homogeneous_props(sec, lay, e_service, es)
    except Exception as exc:
        R["err"]["hom"] = repr(exc)
    try:
        with timed(R, "cracked properties (pure bending)"):
            cr, R["cr"] = cracked_props(sec, e_service, theta)
    except Exception as exc:
        cr = None
        R["err"]["cr"] = repr(exc)
    if "hom" in R:
        try:
            with timed(R, "stiffness at service load"):
                R["eff"] = effective_stiffness(
                    runner, e_service, R["hom"]["i_hom"], fct_eff, es,
                    lay["poly"], sagging, n_s, m_s)
        except Exception as exc:
            R["err"]["eff"] = repr(exc)

    try:
        with timed(R, "service stresses"):
            sres, state, t_unc, warn = None, None, None, None
            try:
                sres, state, t_unc = run_service(runner, fct_eff, e_service, n_s, m_s)
                method = ("Uncracked section (concrete carries tension)"
                          if state == "UNCRACKED"
                          else "Cracked section (concrete resists no tension)")
                method += ", including N_ser"
            except Exception as exc:
                if cr is None:
                    raise
                warn = (f"The service analysis with axial force failed ({exc!r}). "
                        "Showing the elastic cracked bending-only result; N_ser is "
                        "NOT included.")
                sres = sec.calculate_cracked_stress(cracked_results=cr, m=m_s)
                state = "CRACKED (assumed)"
                method = "Elastic cracked section, bending only (N_ser ignored)"
            lo, hi = sres.get_concrete_stress_limits()
            df = bar_stress_table(sres)
            worst = df.loc[df["Stress (MPa)"].abs().idxmax()]
            R["svc"] = dict(state=state, t_unc=t_unc, method=method, warn=warn,
                            sig_c=max(float(hi), 0.0), t_conc=max(-float(lo), 0.0),
                            sig_s=abs(float(worst["Stress (MPa)"])),
                            worst_bar=int(worst["Bar"]), df=df)
            R["img"]["svc_conc"] = to_png(
                soften(sres.plot_stress(render=False).get_figure()))
            R["img"]["svc_bars"] = to_png(
                plot_bar_stresses(sec, df, "Reinforcement stresses (service)"))
    except Exception as exc:
        R["err"]["svc"] = repr(exc)

    if g["shape"] == "Circular":
        R["crack"] = {"status": "circular"}
    else:
        try:
            with timed(R, "crack width"):
                crack_long = bool(mat["crack_long"])
                e_crack = e_lt if crack_long else p["ecm"]
                alpha_e = es / e_crack                  # = n when long term
                kt = 0.4 if crack_long else 0.6
                s_q, state_q, t_q = run_service(
                    runner, fct_eff, e_crack, act["n_qp"] * 1e3, act["m_qp"] * 1e6)
                if state_q == "UNCRACKED":
                    c = {"status": "uncracked", "t_q": t_q}
                else:
                    c = ec2_crack(bar_stress_table(s_q), lay["poly"], sagging, es,
                                  alpha_e, p["fctm"], kt, mat["k1"], mat["k3"],
                                  mat["k4"], mat["to_face"])
                    if c["status"] == "ok":
                        R["img"]["crack"] = to_png(plot_preview(
                            lay["poly"], c["ten_bars"],
                            "Ac,eff (yellow) and tension bars (blue)",
                            band=c["band"]))
                c.update(e_crack=e_crack, alpha_e=alpha_e, kt=kt, long=crack_long)
                R["crack"] = c
        except Exception as exc:
            R["err"]["crack"] = repr(exc)
    return R


# ---------------------------------------------------------------- ULS
def radial_capacity(mxs, mys, dx, dy):
    """Distance from the origin to the diagram boundary along the direction (dx, dy)."""
    d = math.hypot(dx, dy)
    if d <= 1e-12:
        return None
    ux, uy = dx / d, dy / d
    pts = list(zip(mxs, mys))
    best = None
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        ex, ey = x2 - x1, y2 - y1
        den = ux * ey - uy * ex
        if abs(den) < 1e-12:
            continue
        t = (x1 * ey - y1 * ex) / den
        s = (x1 * uy - y1 * ux) / den
        if t > 0 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t < best):
            best = t
    return best


def _bending(ctx, c, act, cache):
    """Bending check of one load combination (uniaxial or biaxial)."""
    sec = ctx["sec"]
    n_ed = c["n"] * 1e3
    mx, my = c["mx"] * 1e6, c["my"] * 1e6
    if abs(c["my"]) <= 1e-9:
        theta = 0.0 if c["mx"] >= 0 else math.pi
        res = sec.ultimate_bending_capacity(theta=theta, n=n_ed)
        m_rd = abs(res.m_xy) / 1e6
        return dict(kind="uniaxial", res=res, m_rd=m_rd, m_ed=abs(c["mx"]),
                    d_n=getattr(res, "d_n", None), sagging=(theta == 0.0),
                    util=(abs(c["mx"]) / m_rd if m_rd > 0 else None))

    key = round(n_ed, 0)                       # one biaxial diagram per axial force
    if key not in cache:
        bb = sec.biaxial_bending_diagram(
            n=n_ed, n_points=int(act["bb_points"]), progress_bar=False)
        mxs, mys = bb.get_results_lists()
        my_side = None                         # which face is compressed for M_y > 0
        try:
            probe = min(bb.results, key=lambda r: abs(r.theta - math.pi / 2))
            dfp = bar_stress_table(
                sec.calculate_ultimate_stress(ultimate_results=probe))
            comp = dfp[dfp["Stress (MPa)"] > 1e-6]
            ten = dfp[dfp["Stress (MPa)"] < -1e-6]
            if not comp.empty and not ten.empty and abs(probe.m_y) > 1e-6:
                plus_x = comp["x (mm)"].mean() > ten["x (mm)"].mean()
                right = plus_x if probe.m_y > 0 else (not plus_x)
                my_side = "right (+x)" if right else "left (-x)"
        except Exception:
            pass
        cache[key] = dict(bb=bb, mxs=mxs, mys=mys, my_side=my_side)
    d = cache[key]
    bb, mxs, mys = d["bb"], d["mxs"], d["mys"]
    demand = math.hypot(mx, my)
    cap = radial_capacity(mxs, mys, mx, my)
    try:
        inside = bool(bb.point_in_diagram(mx, my))
    except Exception:
        inside = None
    best = max(bb.results, key=lambda r: (r.m_x * mx + r.m_y * my)
               / (math.hypot(r.m_x, r.m_y) or 1.0))
    cap_pt = ((cap * mx / demand / 1e6, cap * my / demand / 1e6) if cap else (0.0, 0.0))
    return dict(kind="biaxial", res=best, mx_ed=c["mx"], my_ed=c["my"],
                m_res=demand / 1e6, cap=(cap / 1e6 if cap else None),
                util=(demand / cap if cap else None), inside=inside,
                my_side=d["my_side"], theta_best=best.theta,
                diag=([m / 1e6 for m in mxs], [m / 1e6 for m in mys], cap_pt))


def _shear(ctx, lay, g, act, c):
    """Shear checks of one load combination, along y (major) and x (minor)."""
    mat = ctx["mat"]
    dims = auto_shear_dims(lay, c["mx"] >= 0)
    use = dict(
        bw_y=act["bw_y"] if act["bw_y"] > 0 else dims["bw_y"],
        d_y=act["d_y"] if act["d_y"] > 0 else dims["d_y"],
        bw_x=act["bw_x"] if act["bw_x"] > 0 else dims["bw_x"],
        d_x=act["d_x"] if act["d_x"] > 0 else dims["d_x"],
        asl_y=dims["asl_y"], asl_x=dims["asl_x"])

    def one(v, bw, d, asl, legs):
        return shear_check(
            abs(v), bw, d, asl, lay["conc_area"], c["n"], ctx["fck"],
            mat["gamma_c"], mat["acc_shear"], mat["fywk"], mat["gamma_s"],
            g["link"], legs, act["s_link"], act["cot_mode"], act["cot_manual"],
            g.get("c_nom"))

    return dict(
        y=one(c["vy"], use["bw_y"], use["d_y"], use["asl_y"], act["legs_y"]),
        x=one(c["vx"], use["bw_x"], use["d_x"], use["asl_x"], act["legs_x"]),
        dims=use)


def run_uls(ctx, lay, g, act):
    """ULS check of every load combination in act['combos']."""
    mat = ctx["mat"]
    R = {"err": {}, "img": {}, "time": {}, "act": act, "mat": mat, "p": ctx["p"],
         "fck": ctx["fck"], "fyd": ctx["fyd"], "combos": []}
    cache = {}
    with timed(R, "ULS combinations"):
        for i, c in enumerate(act["combos"]):
            out = dict(name=f"{i + 1}: {c['name']}", n=c["n"], mx=c["mx"], my=c["my"],
                       vy=c["vy"], vx=c["vx"], img={})
            try:
                out["bend"] = _bending(ctx, c, act, cache)
            except Exception as exc:
                out["err_bend"] = repr(exc)
            try:
                out["shear"] = _shear(ctx, lay, g, act, c)
            except Exception as exc:
                out["err_shear"] = repr(exc)
            R["combos"].append(out)
    try:
        R["img"]["ss_conc"] = to_png(ctx["ult_profile"].plot_stress_strain(
            title=f"Concrete ultimate: {mat['ult_model']}", render=False).get_figure())
        R["img"]["ss_steel"] = to_png(ctx["steel_profile"].plot_stress_strain(
            title="Steel: elastic-plastic", render=False).get_figure())
    except Exception:
        pass
    return R


def ensure_combo_images(ctx, combo):
    """Stress plots of one combination, built on first use and then kept."""
    b = combo.get("bend")
    if b is None or "uls_bars" in combo["img"]:
        return
    sec = ctx["sec"]
    uls = sec.calculate_ultimate_stress(ultimate_results=b["res"])
    b["df"] = bar_stress_table(uls)
    combo["img"]["uls_conc"] = to_png(soften(uls.plot_stress(render=False).get_figure()))
    combo["img"]["uls_bars"] = to_png(
        plot_bar_stresses(sec, b["df"], "Reinforcement stresses at ULS"))
    if b["kind"] == "biaxial":
        mxs_k, mys_k, cap_pt = b["diag"]
        combo["img"]["biax"] = to_png(plot_biaxial(
            mxs_k, mys_k, combo["mx"], combo["my"], cap_pt, combo["n"]))


def compute_mn(ctx, combo, both):
    """Uniaxial M-N diagram about the x axis for one combination, on demand."""
    sec, h_depth = ctx["sec"], ctx["h_depth"]
    sagging = combo["mx"] >= 0
    dirs = [(0.0, "Sagging"), (math.pi, "Hogging")]
    if not both:
        dirs = [dirs[0]] if sagging else [dirs[1]]
    curves, errs = {}, []
    for th, lab in dirs:
        mi = None
        try:
            try:
                mi = sec.moment_interaction_diagram(
                    theta=th, n_points=24, progress_bar=False)
            except Exception:
                d_lim = max(1.0, 0.005 * h_depth)       # avoid micron-thin zones
                mi = sec.moment_interaction_diagram(
                    theta=th, n_points=24, limits=[("D", 1.0), ("d_n", d_lim)],
                    progress_bar=False)
        except Exception as exc:
            errs.append(f"{lab}: {exc!r}")
        if mi is None:
            continue
        n_list, m_list = mi.get_results_lists(moment="m_x")
        ms = [m / 1e6 for m in m_list]                  # keep the real sign
        if (th == 0.0 and sum(ms) < 0) or (th != 0.0 and sum(ms) > 0):
            ms = [-m for m in ms]
        curves[lab] = (ms, [n / 1e3 for n in n_list])
    combo["mn_err"] = errs
    combo["img"].pop("mn", None)
    if curves:
        combo["img"]["mn"] = to_png(plot_mn(curves, abs(combo["mx"]), combo["n"], sagging))
