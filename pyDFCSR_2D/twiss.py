import numpy as np
import torch
from .interfaces import to_numpy


def _cov3_torch(a, b, c):
    """3x3 covariance matrix of three 1D tensors, using ddof=1 (matching np.cov default)."""
    n = a.shape[0]
    am, bm, cm = a.mean(), b.mean(), c.mean()
    da, db, dc = a - am, b - bm, c - cm
    inv = 1.0 / (n - 1)
    return torch.stack([
        torch.stack([(da * da).sum() * inv, (da * db).sum() * inv, (da * dc).sum() * inv]),
        torch.stack([(db * da).sum() * inv, (db * db).sum() * inv, (db * dc).sum() * inv]),
        torch.stack([(dc * da).sum() * inv, (dc * db).sum() * inv, (dc * dc).sum() * inv]),
    ])


def twiss_from_bmadx_particles(p):
    if isinstance(p.x, torch.Tensor):
        return _twiss_from_bmadx_particles_torch(p)

    x = to_numpy(p.x)
    px = to_numpy(p.px)
    y = to_numpy(p.y)
    py = to_numpy(p.py)
    pz = to_numpy(p.pz)

    twiss = twiss_dispersion_calc(np.cov([x, px, pz]))
    twiss['norm_emit'] = twiss['emit'] * p.p0c / p.mc2
    out = {}
    for k in twiss:
        out[k + '_x'] = twiss[k]

    twiss = twiss_dispersion_calc(np.cov([y, py, pz]))
    twiss['norm_emit'] = twiss['emit'] * p.p0c / p.mc2
    for k in twiss:
        out[k + '_y'] = twiss[k]

    return out


def _twiss_from_bmadx_particles_torch(p):
    sigma3_x = _cov3_torch(p.x, p.px, p.pz)
    twiss = _twiss_dispersion_calc_torch(sigma3_x)
    p0c = float(p.p0c)
    mc2 = float(p.mc2)
    twiss['norm_emit'] = twiss['emit'] * p0c / mc2
    out = {}
    for k in twiss:
        out[k + '_x'] = twiss[k]

    sigma3_y = _cov3_torch(p.y, p.py, p.pz)
    twiss = _twiss_dispersion_calc_torch(sigma3_y)
    twiss['norm_emit'] = twiss['emit'] * p0c / mc2
    for k in twiss:
        out[k + '_y'] = twiss[k]

    return out


def _twiss_dispersion_calc_torch(sigma3):
    """Torch version of twiss_dispersion_calc. Returns Python floats."""
    delta2 = sigma3[2, 2]
    xd = sigma3[0, 2]
    pd = sigma3[1, 2]

    eb = sigma3[0, 0] - xd ** 2 / delta2
    eg = sigma3[1, 1] - pd ** 2 / delta2
    ea = -sigma3[0, 1] + xd * pd / delta2

    emit = torch.sqrt(eb * eg - ea ** 2)

    d = {}
    d['alpha'] = (ea / emit).item()
    d['beta'] = (eb / emit).item()
    d['gamma'] = (eg / emit).item()
    d['emit'] = emit.item()
    d['eta'] = (xd / delta2).item()
    d['etap'] = (pd / delta2).item()
    return d


def twiss_dispersion_calc(sigma3):
    """
    Twiss and Dispersion calculation from a 3x3 sigma (covariance) matrix from particles
    x, p, delta

    From https://github.com/ChristopherMayes/openPMD-beamphysics/blob/master/pmd_beamphysics/statistics.py

    Formulas from:
        https://uspas.fnal.gov/materials/19Knoxville/g-2/creation-and-analysis-of-beam-distributions.html

    Returns a dict with:
        alpha
        beta
        gamma
        emit
        eta
        etap

    """

    # Collect terms

    delta2 = sigma3[2, 2]
    xd = sigma3[0, 2]
    pd = sigma3[1, 2]

    eb = sigma3[0, 0] - xd ** 2 / delta2
    eg = sigma3[1, 1] - pd ** 2 / delta2
    ea = -sigma3[0, 1] + xd * pd / delta2

    emit = np.sqrt(eb * eg - ea ** 2)

    # Form the output dict
    d = {}

    d['alpha'] = ea / emit
    d['beta'] = eb / emit
    d['gamma'] = eg / emit
    d['emit'] = emit
    d['eta'] = xd / delta2
    d['etap'] = pd / delta2

    return d