import math

from shapely.geometry import Point, Polygon

MAX_LAYERS = 50


def area(dia):
    return math.pi * dia**2 / 4


def fmt(v, nd=0):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


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
    """Rectangular / T-section: bars from the bottom and top layer tables."""
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


def build_layout(g):
    """Concrete outline, bar list, spacing checks and validation from the inputs g."""
    shape, dg, cover_tot = g["shape"], g["dg"], g["cover_tot"]
    poly, bars, rows, errors = None, [], [], []

    if shape in ("Rectangular", "T-section"):
        b, h = g["b"], g["h"]
        bf, hf = g.get("bf", 0.0), g.get("hf", 0.0)
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
            bot, top = g["bottom_df"].dropna(), g["top_df"].dropna()
            if len(bot) + len(top) > MAX_LAYERS:
                errors.append(f"A maximum of {MAX_LAYERS} layers is allowed.")
            else:
                max_lim = (min(3 * h, 400.0) if (shape == "Rectangular"
                           and g.get("elem", "Beam").startswith("Slab")) else None)
                bars, rows, errs = build_layers(shape, b, h, bf, hf, bot, top,
                                                cover_tot, dg, max_lim)
                errors += errs
    elif shape == "Rectangular wall":
        t_w, l_w = g["t_w"], g["l_w"]
        in_plane = g["orient"].startswith("In-plane")
        if in_plane:
            poly = Polygon([(0, 0), (t_w, 0), (t_w, l_w), (0, l_w)])
        else:
            poly = Polygon([(0, 0), (l_w, 0), (l_w, t_w), (0, t_w)])
        bars, rows, errors = build_wall(
            t_w, l_w, in_plane,
            [("Face 1", float(g["f1_dia"]), float(g["f1_s"])),
             ("Face 2", float(g["f2_dia"]), float(g["f2_s"]))],
            cover_tot, dg, float(g["d_min"]))
    elif shape == "Circular":
        diam, n_bar, dia_bar = g["diam"], int(g["n_bar"]), float(g["dia_bar"])
        poly = Point(0, 0).buffer(diam / 2, 64)
        r_bars = diam / 2 - cover_tot - dia_bar / 2
        if r_bars <= 0:
            errors.append("The cover leaves no room for the bars.")
            poly = None
        else:
            bars = [(r_bars * math.cos(2 * math.pi * k / n_bar),
                     r_bars * math.sin(2 * math.pi * k / n_bar), dia_bar, "Ring")
                    for k in range(n_bar)]
            chord = 2 * r_bars * math.sin(math.pi / n_bar)
            rows.append(make_row("Ring of bars (chord spacing)", n_bar, dia_bar,
                                 chord, dg, None))
    else:
        pts = list(g["poly_df"].dropna().itertuples(index=False, name=None))
        cand = Polygon(pts) if len(pts) >= 3 else None
        if cand is None or not cand.is_valid or cand.area <= 0:
            errors.append("The outline needs at least 3 vertices and must not "
                          "cross itself.")
        else:
            poly = cand
            for k, r in enumerate(g["bar_df"].dropna().itertuples(index=False), 1):
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

    ok = poly is not None and bool(bars) and not errors
    by_dia = {}
    if ok and shape != "Circular":
        for x, y, d, _ in bars:
            by_dia.setdefault(d, ([], []))
            by_dia[d][0].append(x)
            by_dia[d][1].append(y)

    if shape == "Circular" and poly is not None:
        conc_area, conc_cy = area(g["diam"]), 0.0
        y_min, h_depth = -g["diam"] / 2, g["diam"]
    elif poly is not None:
        conc_area, conc_cy = poly.area, poly.centroid.y
        y_min, h_depth = poly.bounds[1], poly.bounds[3] - poly.bounds[1]
    else:
        conc_area = conc_cy = y_min = h_depth = 0.0

    return dict(poly=poly, bars=bars, rows=rows, errors=errors, ok=ok,
                n_bars=len(bars), as_total=sum(area(d) for _, _, d, _ in bars),
                by_dia=by_dia, conc_area=conc_area, conc_cy=conc_cy,
                y_min=y_min, h_depth=h_depth)
