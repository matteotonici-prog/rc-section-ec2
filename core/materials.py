import math

from concreteproperties import SteelBar
from concreteproperties.stress_strain_profile import (
    EurocodeParabolicUltimate, RectangularStressBlock, SteelElasticPlastic,
)

CLASSES = {"C20/25": 20, "C25/30": 25, "C30/37": 30, "C32/40": 32, "C35/45": 35,
           "C40/50": 40, "C45/55": 45, "C50/60": 50, "C55/67": 55, "C60/75": 60,
           "C70/85": 70, "C80/95": 80, "C90/105": 90}


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


def make_profiles(p, fyd, es, eud, ult_model):
    """Returns (ultimate concrete profile, steel profile, SteelBar material)."""
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
    return ult_profile, steel_profile, steel
