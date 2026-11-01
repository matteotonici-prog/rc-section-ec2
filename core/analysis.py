import math

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
from core.plots import plot_bar_stresses, plot_mn, plot_preview, soften, to_png

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


def run_service(make_section, theta, fct_eff, sec_u, e_mod, n_s, m_s):
    """Uncracked analysis first; if the tension exceeds fct,eff, re-run cracked."""
    mk_u = sec_u.moment_curvature_analysis(theta=theta, n=n_s, progress_bar=False)
    s_unc = sec_u.calculate_service_stress(moment_curvature_results=mk_u, m=m_s)
    lo_u, _ = s_unc.get_concrete_stress_limits()
    t_unc = max(0.0, -lo_u)                          # tension is negative
    if t_unc <= fct_eff:
        return s_unc, "UNCRACKED", t_unc
    sec_c = make_section(ConcreteLinearNoTension(elastic_modulus=e_mod))
    mk_c = sec_c.moment_curvature_analysis(theta=theta, n=n_s, progress_bar=False)
    s_cr = sec_c.calculate_service_stress(moment_curvature_results=mk_c, m=m_s)
    return s_cr, "CRACKED", t_unc


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


def effective_stiffness(sec, make_section, theta, e_service, i_hom, fct_eff, es,
                        poly, sagging, n_s, m_s):
    """Secant stiffness M/kappa at the service load, including the axial force."""
    out = {}
    mk_u = sec.moment_curvature_analysis(theta=theta, n=n_s, progress_bar=False)
    s_u = sec.calculate_service_stress(moment_curvature_results=mk_u, m=m_s)
    lo_u, _ = s_u.get_concrete_stress_limits()
    out["t_conc"] = max(0.0, -float(lo_u))
    if abs(m_s) <= 1e-9:
        out["kind"] = "zero"
        return out
    if out["t_conc"] <= fct_eff:
        out.update(kind="uncracked", i_eff=i_hom, ei_eff=e_service * i_hom)
        return out

    sec_c = make_section(ConcreteLinearNoTension(elastic_modulus=e_service))
    mk = sec_c.moment_curvature_analysis(theta=theta, n=n_s, progress_bar=False)
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
        s_eff = sec_c.calculate_service_stress(moment_curvature_results=mk, m=m_s)
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


def run_all(lay, g, inp):
    """Runs every analysis; returns a dict of numbers, tables and PNG images."""
    R = {"err": {}, "img": {}}
    p = ec2_concrete(CLASSES[inp["cls"]], inp["acc"], inp["gamma_c"])
    es = float(inp["es"])
    fyd = inp["fyk"] / inp["gamma_s"]
    sagging = bool(inp["sagging"])
    theta = 0.0 if sagging else math.pi
    e_lt = es / inp["n_long"]
    long_basis = bool(inp["basis_long"])
    e_service = e_lt if long_basis else p["ecm"]
    fct_eff = (max((1.6 - lay["h_depth"] / 1000) * p["fctm"], p["fctm"])
               if inp["crack_basis_fl"] else p["fctm"])
    ult_profile, steel_profile, steel = make_profiles(
        p, fyd, es, inp["eud"], inp["ult_model"])
    make_section = make_section_factory(lay, g, ult_profile, steel, p, inp["cls"])
    sec = make_section(ConcreteLinear(elastic_modulus=e_service))
    n_ed = inp["n_ed"] * 1e3
    n_s, m_s = inp["n_ser"] * 1e3, inp["m_ser"] * 1e6
    R.update(p=p, fck=CLASSES[inp["cls"]], fyd=fyd, es=es, theta=theta,
             sagging=sagging, e_service=e_service, e_lt=e_lt, fct_eff=fct_eff,
             inp=inp, basis_txt="Long-term" if long_basis else "Short-term")

    # ---- section figure
    try:
        R["img"]["section"] = to_png(sec.plot_section(render=False).get_figure())
    except Exception as exc:
        R["err"]["section"] = repr(exc)

    # ---- homogeneous, cracked (pure bending), effective at service
    cr = None
    try:
        R["hom"] = homogeneous_props(sec, lay, e_service, es)
    except Exception as exc:
        R["err"]["hom"] = repr(exc)
    try:
        cr = sec.calculate_cracked_properties(theta=theta)
        cr.calculate_transformed_properties(elastic_modulus=e_service)
        i_cr = float(cr.iuu_cr)
        R["cr"] = dict(i_cr=i_cr, ei_cr=e_service * i_cr, m_cr=float(cr.m_cr) / 1e6,
                       d_nc=float(cr.d_nc))
    except Exception as exc:
        cr = None
        R["err"]["cr"] = repr(exc)
    if "hom" in R:
        try:
            R["eff"] = effective_stiffness(
                sec, make_section, theta, e_service, R["hom"]["i_hom"], fct_eff, es,
                lay["poly"], sagging, n_s, m_s)
        except Exception as exc:
            R["err"]["eff"] = repr(exc)

    # ---- service stresses
    try:
        sres, state, t_unc, warn = None, None, None, None
        try:
            sres, state, t_unc = run_service(make_section, theta, fct_eff, sec,
                                             e_service, n_s, m_s)
            method = ("Uncracked section (concrete carries tension)"
                      if state == "UNCRACKED"
                      else "Cracked section (concrete resists no tension)")
            method += ", including N_ser"
        except Exception as exc:
            if cr is None:
                raise
            warn = (f"The service analysis with axial force failed ({exc!r}). "
                    "Showing the elastic cracked bending-only result; N_ser is NOT "
                    "included.")
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
        R["img"]["svc_conc"] = to_png(soften(sres.plot_stress(render=False).get_figure()))
        R["img"]["svc_bars"] = to_png(
            plot_bar_stresses(sec, df, "Reinforcement stresses (service)"))
    except Exception as exc:
        R["err"]["svc"] = repr(exc)

    # ---- ULS bending
    try:
        res = sec.ultimate_bending_capacity(theta=theta, n=n_ed)
        uls = sec.calculate_ultimate_stress(ultimate_results=res)
        df_u = bar_stress_table(uls)
        R["uls"] = dict(m_rd=abs(res.m_xy) / 1e6, d_n=getattr(res, "d_n", None),
                        df=df_u)
        R["img"]["uls_conc"] = to_png(soften(uls.plot_stress(render=False).get_figure()))
        R["img"]["uls_bars"] = to_png(
            plot_bar_stresses(sec, df_u, "Reinforcement stresses at ULS"))
        try:
            R["img"]["ss_conc"] = to_png(ult_profile.plot_stress_strain(
                title=f"Concrete ultimate: {inp['ult_model']}", render=False).get_figure())
            R["img"]["ss_steel"] = to_png(steel_profile.plot_stress_strain(
                title="Steel: elastic-plastic", render=False).get_figure())
        except Exception:
            pass
    except Exception as exc:
        R["err"]["uls"] = repr(exc)

    # ---- M-N diagram (both directions)
    curves, mn_err = {}, []
    for th, lab in ((0.0, "Sagging"), (math.pi, "Hogging")):
        mi = None
        try:
            try:
                mi = sec.moment_interaction_diagram(
                    theta=th, n_points=48, progress_bar=False)
            except Exception:
                d_lim = max(1.0, 0.005 * lay["h_depth"])   # avoid micron-thin zones
                mi = sec.moment_interaction_diagram(
                    theta=th, n_points=48, limits=[("D", 1.0), ("d_n", d_lim)],
                    progress_bar=False)
        except Exception as exc:
            mn_err.append(f"{lab}: {exc!r}")
        if mi is None:
            continue
        n_list, m_list = mi.get_results_lists(moment="m_x")
        ms = [m / 1e6 for m in m_list]                  # keep the real sign
        if (th == 0.0 and sum(ms) < 0) or (th != 0.0 and sum(ms) > 0):
            ms = [-m for m in ms]
        curves[lab] = (ms, [n / 1e3 for n in n_list])
    R["mn_err"] = mn_err
    if curves:
        R["img"]["mn"] = to_png(plot_mn(curves, inp["m_ed"], inp["n_ed"], sagging))

    # ---- crack width (EC2 7.3.4), short or long term
    if g["shape"] == "Circular":
        R["crack"] = {"status": "circular"}
    else:
        try:
            crack_long = bool(inp["crack_long"])
            e_crack = e_lt if crack_long else p["ecm"]
            alpha_e = es / e_crack                  # = n when long term
            kt = 0.4 if crack_long else 0.6
            sec_c = (sec if abs(e_service - e_crack) < 1e-6
                     else make_section(ConcreteLinear(elastic_modulus=e_crack)))
            s_q, state_q, t_q = run_service(
                make_section, theta, fct_eff, sec_c, e_crack,
                inp["n_qp"] * 1e3, inp["m_qp"] * 1e6)
            if state_q == "UNCRACKED":
                c = {"status": "uncracked", "t_q": t_q}
            else:
                c = ec2_crack(bar_stress_table(s_q), lay["poly"], sagging, es,
                              alpha_e, p["fctm"], kt, inp["k1"], inp["k3"],
                              inp["k4"], inp["to_face"])
                if c["status"] == "ok":
                    R["img"]["crack"] = to_png(plot_preview(
                        lay["poly"], c["ten_bars"],
                        "Ac,eff (yellow) and tension bars (blue)", band=c["band"]))
            c.update(e_crack=e_crack, alpha_e=alpha_e, kt=kt, long=crack_long)
            R["crack"] = c
        except Exception as exc:
            R["err"]["crack"] = repr(exc)
    return R
