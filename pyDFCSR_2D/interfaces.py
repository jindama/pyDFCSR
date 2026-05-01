import numpy as np
from bmadx.constants import C_LIGHT, M_ELECTRON, E_CHARGE
from bmadx.structures import Particle
import torch
import sys


# Modified from https://github.com/bmad-sim/Bmad-X


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_torch(x, device='cpu', dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(x), device=device, dtype=dtype)


from pmd_beamphysics import ParticleGroup
def openpmd_to_bmadx_coords(
        pmd_particle: ParticleGroup,
        p0c
):
    """
    Transforms openPMD-beamphysics ParticleGroup to
    bmad phase-space coordinates.

        Parameters:
            pmd_particle (pmd_beamphysics.ParticleGroup): openPMD-beamphysics ParticleGroup
            p0c (float): reference momentum in eV

        Returns:
            bmad_coods (list): list of bmad coords (x, px, y, py, z, pz)
    """

    x = pmd_particle.x
    px = pmd_particle.px / p0c
    y = pmd_particle.y
    py = pmd_particle.py / p0c
    z =  pmd_particle.z
    pz = pmd_particle.p / p0c - 1.0

    bmad_coords = (x, px, y, py, z, pz)

    return bmad_coords
def openpmd_to_bmadx_particles(
        pmd_particle: ParticleGroup,
        p0c: float,
        s : float = 0.0,
        mc2 : float = M_ELECTRON,
        device: str = 'cpu'
        ):
    """
    Transforms openPMD-beamphysics ParticleGroup to
    bmad phase-space Particle named tuple.

        Parameters:
            pmd_particle (pmd_beamphysics.ParticleGroup): openPMD-beamphysics ParticleGroup
            p0c (float): reference momentum in eV
            device (str): torch device ('cpu' or 'cuda')

        Returns:
            Bmadx Particle
    """
    coords = openpmd_to_bmadx_coords(pmd_particle, p0c)
    if device != 'cpu':
        coords = tuple(to_torch(c, device=device) for c in coords)
    particle = Particle(
        *coords,
        s = s,
        p0c = p0c,
        mc2 = mc2)
    return particle


def bmadx_particles_to_openpmd(particle: Particle, charge):
    """
    Transforms bmadx Particle to openPMD-beamphysics ParticleGroup.

        Parameters
        ----------
        particle: bmax Particle
            particle to transform.

        Returns
        -------
        pmd_beamphysics.ParticleGroup
    """
    lib = sys.modules[type(particle.x).__module__]
    if lib == np:
        x = particle.x
        px = particle.px
        y = particle.y
        py = particle.py
        z = particle.z
        pz = particle.pz
    elif lib == torch:
        x = particle.x.detach().cpu().numpy()
        px = particle.px.detach().cpu().numpy()
        y = particle.y.detach().cpu().numpy()
        py = particle.py.detach().cpu().numpy()
        z = particle.z.detach().cpu().numpy()
        pz = particle.pz.detach().cpu().numpy()
    else:
        raise ValueError('Only numpy and torch Particles are supported as of now')

    p0c = float(particle.p0c)
    s = float(particle.s)
    mc2 = float(particle.mc2)

    dat = {}

    dat['x'] = x
    dat['px'] = px * p0c
    dat['y'] = y
    dat['py'] = py * p0c
    dat['z'] = z
    dat['pz'] = p0c * ((pz + 1.0) ** 2 - px ** 2 - py ** 2) ** 0.5

    p = (1 + pz) * p0c
    beta = (
                   (p / M_ELECTRON) ** 2 /
                   (1 + (p / M_ELECTRON) ** 2)
           ) ** 0.5

    dat['t'] = (s - z) / (C_LIGHT * beta)

    dat['status'] = np.ones_like(x, dtype=int)
    dat['weight'] = np.ones_like(x) * charge/len(x)

    if np.isclose(mc2, M_ELECTRON):
        dat['species'] = 'electron'
    else:
        raise ValueError('only electrons are supported as of now')

    return ParticleGroup(data=dat)

