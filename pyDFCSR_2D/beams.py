import numpy as np
import torch
from distgen import Generator
from .physical_constants import MC2
from scipy.interpolate import RegularGridInterpolator
from bmadx import Particle, M_ELECTRON
from .interfaces import openpmd_to_bmadx_particles, bmadx_particles_to_openpmd
from .interfaces import to_numpy, to_torch
from bmadx import track_element
from pmd_beamphysics import ParticleGroup
from .twiss import twiss_from_bmadx_particles


def _polyfit1_torch(z, x):
    """Degree-1 polyfit: returns [slope, intercept] matching np.polyfit convention."""
    n = z.shape[0]
    z_mean = z.mean()
    x_mean = x.mean()
    slope = ((z - z_mean) * (x - x_mean)).sum() / ((z - z_mean) ** 2).sum()
    intercept = x_mean - slope * z_mean
    return torch.stack([slope, intercept])


def _polyval1_torch(p, z):
    """Evaluate degree-1 polynomial p = [slope, intercept] at z."""
    return p[0] * z + p[1]


def _std_torch(x):
    return torch.std(x, correction=0)


class Beam():
    """
    Beam class to initialize, track and apply wakes
    """
    def __init__(self, input_beam, device='cpu'):

        self.device = device
        self.check_inputs(input_beam)
        self.input_beam_config = input_beam
        self.style = input_beam['style']

        if self.style == 'from_file':
            filename = input_beam['beamfile']

            ## Read bmadx coords
            coords = np.loadtxt(filename)
            assert coords.shape[1] == 6, f'Error: input beam must have 6 dimension, but get {coords.shape[1]} instead'

            self._charge = input_beam['charge']
            self._init_energy = input_beam['energy']

            if device == 'cpu':
                self.particle = Particle(*coords.T, 0, self._init_energy, MC2)
            else:
                coords_t = to_torch(coords, device=device)
                self.particle = Particle(*coords_t.T, 0, self._init_energy, MC2)

        elif self.style == 'distgen':
            filename = input_beam['distgen_input_file']
            gen = Generator(filename)
            gen.run()
            pg = gen.particles
            self._charge = pg['charge']
            self._init_energy = np.mean(pg['energy'])

            self.particle = openpmd_to_bmadx_particles(pg, self._init_energy, 0.0, MC2, device=device)

        else:
            ParticleGroup_h5 = input_beam['ParticleGroup_h5']
            pg = ParticleGroup(ParticleGroup_h5)

            self._charge = pg['charge']
            self._init_energy = np.mean(pg['energy'])

            self.particle = openpmd_to_bmadx_particles(pg, self._init_energy, 0.0, MC2, device=device)

        self._init_gamma = self._init_energy / MC2

        self.position = 0
        self.step = 0
        self._use_torch = isinstance(self.particle.x, torch.Tensor)

        self.update_status()


    def check_inputs(self, input_beam):
        assert 'style' in input_beam, 'ERROR: input_beam must have keyword <style>'
        if input_beam['style'] == 'from_file':
            self.required_inputs = ['style', 'beamfile', 'charge','energy']
        elif input_beam['style'] == 'distgen':
            self.required_inputs = ['style', 'distgen_input_file']
        elif input_beam['style'] == 'ParticleGroup':
            self.required_inputs = ['style', 'ParticleGroup_h5']
        else:
            raise Exception("input beam parsing Error: invalid input style")

        allowed_params = self.required_inputs + ['verbose']
        for input_param in input_beam:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input_beam, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def update_status(self):
        # Cache as plain floats/numpy so CSR code doesn't need torch conversion
        sx = self.sigma_x
        sz = self.sigma_z
        mx = self.mean_x
        mz = self.mean_z
        sl = self.slope
        if self._use_torch:
            self._sigma_x = sx.item()
            self._sigma_z = sz.item()
            self._mean_x = mx.item()
            self._mean_z = mz.item()
            self._slope = to_numpy(sl)
        else:
            self._sigma_x = sx
            self._sigma_z = sz
            self._mean_x = mx
            self._mean_z = mz
            self._slope = sl

    def track(self, element, step_size, update_step=True):
        self.particle = track_element(self.particle, element)
        self.position += step_size
        if update_step:
            self.step += 1
        self.update_status()

    def apply_wakes(self, dE_dct, x_kick, xrange, zrange, step_size, transverse_on):
        dE_E1 = step_size * dE_dct * 1e6 / self.init_energy
        if self._use_torch:
            from .torch_interp import interpolate2d_bilinear_torch
            # CSR arrays may be numpy — convert to torch on the right device
            dE_E1_t = to_torch(dE_E1, device=self.device)
            xrange_t = to_torch(xrange, device=self.device)
            zrange_t = to_torch(zrange, device=self.device)
            x_t = self.x_transform
            z_t = self.z
            dE_Es = interpolate2d_bilinear_torch(x_t, z_t, dE_E1_t, xrange_t, zrange_t)
            pz_new = self.particle.pz + dE_Es

            if transverse_on:
                dxp = step_size * x_kick * 1e6 / self.init_energy
                dxp_t = to_torch(dxp, device=self.device)
                dxps = interpolate2d_bilinear_torch(x_t, z_t, dxp_t, xrange_t, zrange_t)
                px_new = self.particle.px + dxps
            else:
                px_new = self.particle.px
        else:
            interp = RegularGridInterpolator((xrange, zrange), dE_E1, fill_value=0.0, bounds_error=False)
            dE_Es = interp(np.array([self.x_transform, self.z]).T)
            pz_new = self.particle.pz + dE_Es

            if transverse_on:
                dxp = step_size * x_kick * 1e6 / self.init_energy
                interp = RegularGridInterpolator((xrange, zrange), dxp, fill_value=0.0, bounds_error=False)
                dxps = interp(np.array([self.x_transform, self.z]).T)
                px_new = self.particle.px + dxps
            else:
                px_new = self.particle.px

        self.particle = Particle(self.particle.x, px_new,
                                 self.particle.y, self.particle.py,
                                 self.particle.z, pz_new,
                                 self.particle.s, self.particle.p0c, self.particle.mc2)

        self.update_status()

    def frog_leap(self):
        # Todo: track half step, apply kicks, track another half step
        pass

    @property
    def mean_x(self):
        if self._use_torch:
            return self.particle.x.mean()
        return np.mean(self.particle.x)

    @property
    def mean_y(self):
        if self._use_torch:
            return self.particle.y.mean()
        return np.mean(self.particle.y)

    @property
    def sigma_x(self):
        if self._use_torch:
            return _std_torch(self.particle.x)
        return np.std(self.particle.x)

    @property
    def sigma_z(self):
        if self._use_torch:
            return _std_torch(self.particle.z)
        return np.std(self.particle.z)

    @property
    def mean_z(self):
        if self._use_torch:
            return self.particle.z.mean()
        return np.mean(self.particle.z)

    @property
    def init_energy(self):
        return self._init_energy

    @property
    def init_gamma(self):
        return self._init_gamma

    @property
    def energy(self):
        return (self.particle.pz + 1) * self.particle.p0c

    @property
    def mean_energy(self):
        if self._use_torch:
            return self.energy.mean()
        return np.mean(self.energy)

    @property
    def gamma(self):
        return self.energy / MC2

    @property
    def sigma_energy(self):
        if self._use_torch:
            return _std_torch(self.energy)
        return np.std(self.energy)

    @property
    def x(self):
        return self.particle.x

    @property
    def px(self):
        return self.particle.px

    @property
    def z(self):
        return self.particle.z

    @property
    def pz(self):
        return self.particle.pz

    @property
    def slope(self):
        if self._use_torch:
            return _polyfit1_torch(self.z, self.x)
        return np.polyfit(self.z, self.x, deg=1)

    @property
    def x_transform(self):
        """x coordinates after removing the x-z chirp"""
        if self._use_torch:
            return self.x - _polyval1_torch(self.slope, self.z)
        return self.x - np.polyval(self.slope, self.z)

    @property
    def sigma_x_transform(self):
        if self._use_torch:
            return _std_torch(self.x_transform)
        return np.std(self.x_transform)

    @property
    def charge(self):
        return self._charge

    @property
    def twiss(self):
        return twiss_from_bmadx_particles(self.particle)

    @property
    def particle_group(self):
        pg = bmadx_particles_to_openpmd(self.particle, self.charge)
        return pg
