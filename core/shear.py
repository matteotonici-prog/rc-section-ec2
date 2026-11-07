import math

import numpy as np
from shapely.geometry import LineString


def _ab(d):
    return math.pi * d * d / 4


def _min_chord(poly, horizontal, lo, hi, n=25):
    """Smallest chord length of the section along sampled horizontal/vertical lines."""
    minx, miny, maxx, maxy = poly.bounds
    vals = []
    for pos in np.linspace(lo, hi, n):
        if horizontal:
            line = LineString([(minx - 1.0, pos), (maxx + 1.0, pos)])
        else:
            line = LineString([(pos, miny - 1.0), (pos, maxy + 1.0)])
        length = poly.intersection(line).length
        if length > 1e-6:
            vals.append(length)
    return float(min(vals)) if vals else 0.0


def auto_shear_dims(lay, sagging_y):
    """Default bw, d and tension steel area for shear along y (major) and x (minor)."""
    poly, bars = lay["poly"], lay["bars"]
    minx, miny, maxx, maxy = poly.bounds
    h, w = maxy - miny, maxx - minx

    # --- shear along y: tension side from the sign of M_x
    bw_y = _min_chord(poly, True, miny + 0.1 * h, maxy - 0.1 * h)
    if sagging_y:
        half = [(y - miny, d) for _, y, d, _ in bars if y <= miny + h / 2]
    else:
        half = [(maxy - y, d) for _, y, d, _ in bars if y >= maxy - h / 2]
    if half:
        a_tot = sum(_ab(d) for _, d in half)
        c_t = sum(_ab(d) * dist for dist, d in half) / a_tot
        d_y, asl_y = h - c_t, a_tot
    else:
        d_y, asl_y = 0.9 * h, 0.0

    # --- shear along x: most unfavourable side
    bw_x = _min_chord(poly, False, minx + 0.1 * w, maxx - 0.1 * w)
    cand = []
    for side in ("left", "right"):
        if side == "left":
            half = [(x - minx, d) for x, _, d, _ in bars if x <= minx + w / 2]
        else:
            half = [(maxx - x, d) for x, _, d, _ in bars if x >= maxx - w / 2]
        if half:
            a_tot = sum(_ab(d) for _, d in half)
            c_t = sum(_ab(d) * dist for dist, d in half) / a_tot
            cand.append((w - c_t, a_tot))
    if cand:
        d_x, asl_x = min(c[0] for c in cand), min(c[1] for c in cand)
    else:
        d_x, asl_x = 0.9 * w, 0.0
    return dict(bw_y=bw_y, d_y=d_y, asl_y=asl_y, bw_x=bw_x, d_x=d_x, asl_x=asl_x,
                h=h, w=w)


def shear_check(v_ed, bw, d, asl, ac, n_ed, fck, gamma_c, acc_shear, fywk, gamma_s,
                dia_link, legs, s, cot_mode, cot_manual, c_nom=None):
    """EN 1992-1-1 6.2.2 (VRd,c) and 6.2.3 (VRd,s, VRd,max), vertical links.
    Forces in kN, lengths in mm, stresses in MPa."""
    if bw <= 0 or d <= 0 or ac <= 0:
        return {"error": "bw, d and the concrete area must be positive."}
    fcd_s = acc_shear * fck / gamma_c
    sig_cp = n_ed * 1e3 / ac                         # MPa, compression positive
    sig_cp_c = min(sig_cp, 0.2 * fcd_s)

    # ---- 6.2.2(1): concrete without shear reinforcement
    k = min(1.0 + math.sqrt(200.0 / d), 2.0)
    rho_l = min(asl / (bw * d), 0.02)
    crdc = 0.18 / gamma_c
    v_min = 0.035 * k ** 1.5 * math.sqrt(fck)
    vrdc_a = (crdc * k * (100.0 * rho_l * fck) ** (1.0 / 3.0)
              + 0.15 * sig_cp_c) * bw * d / 1e3
    vrdc_b = (v_min + 0.15 * sig_cp_c) * bw * d / 1e3
    vrdc = max(vrdc_a, vrdc_b, 0.0)

    # ---- 6.2.3: vertical links, variable strut inclination
    asw = legs * math.pi * dia_link ** 2 / 4.0
    has_links = asw > 0 and s > 0
    fywd = fywk / gamma_s
    nu1 = 0.6 * (1.0 - fck / 250.0)
    z = 0.9 * d
    if sig_cp <= 0:
        acw = 1.0
    elif sig_cp <= 0.25 * fcd_s:
        acw = 1.0 + sig_cp / fcd_s
    elif sig_cp <= 0.5 * fcd_s:
        acw = 1.25
    else:
        acw = max(2.5 * (1.0 - sig_cp / fcd_s), 0.0)

    cot = vrd_s = vrd_max = None
    if has_links:
        kk = acw * bw * nu1 * fcd_s * s / (asw * fywd)
        cot_opt = min(max(math.sqrt(max(kk - 1.0, 0.0)), 1.0), 2.5)
        cot = cot_opt if cot_mode == "optimum" else min(max(cot_manual, 1.0), 2.5)
        vrd_s = asw / s * z * fywd * cot / 1e3
        vrd_max = acw * bw * z * nu1 * fcd_s / (cot + 1.0 / cot) / 1e3
        vrd = min(vrd_s, vrd_max)
        governs = ("links (VRd,s)" if vrd_s <= vrd_max
                   else "concrete struts (VRd,max)")
    else:
        vrd, governs = vrdc, "concrete only (VRd,c): no links entered"

    checks = []
    st_val = st_max = rho_w = rho_w_min = None
    if has_links:
        rho_w = asw / (s * bw)
        rho_w_min = 0.08 * math.sqrt(fck) / fywk
        checks.append((f"ρw = {rho_w:.4f} ≥ ρw,min = {rho_w_min:.4f} (EC2 9.5N)",
                       rho_w >= rho_w_min))
        checks.append((f"s = {s:.0f} mm ≤ 0.75d = {0.75 * d:.0f} mm (EC2 9.6N)",
                       s <= 0.75 * d))
        if c_nom is not None and legs >= 2:
            st_val = (bw - 2 * (c_nom + dia_link / 2)) / (legs - 1)
            st_max = min(0.75 * d, 600.0)
            checks.append((f"transverse leg spacing ≈ {st_val:.0f} mm ≤ "
                           f"{st_max:.0f} mm (EC2 9.8N)", st_val <= st_max))

    return dict(
        v_ed=v_ed, bw=bw, d=d, z=z, asl=asl, rho_l=rho_l, k=k, crdc=crdc,
        v_min=v_min, sig_cp=sig_cp, vrdc=vrdc, asw=asw, fywd=fywd, nu1=nu1,
        acw=acw, cot=cot, theta=(math.degrees(math.atan(1.0 / cot)) if cot else None),
        vrd_s=vrd_s, vrd_max=vrd_max, vrd=vrd, governs=governs,
        util=(v_ed / vrd if vrd > 0 else float("inf")),
        links_not_needed=(v_ed <= vrdc), has_links=has_links, checks=checks)
