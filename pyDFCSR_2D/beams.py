import numpy as np
from distgen import Generator
from .physical_constants import MC2
from scipy.interpolate import RegularGridInterpolator
from bmadx import Particle, M_ELECTRON
from .interfaces import  openpmd_to_bmadx_particles, bmadx_particles_to_openpmd
from bmadx import track_element
from pmd_beamphysics import ParticleGroup
from .twiss import  twiss_from_bmadx_particles

class Beam():
    """
    Beam class to initialize, track, and apply wakes
    """
    def __init__(self, input_beam):
        """
        Initalizes instance of Beam using the settings defined in input_beam. Input beam may be in 3 different allowed
        formats. Regardless of the formatt, 3 class attributes are defined: _charge, _init_energy, and particle.
        Parameters:
            input_beam: dictionary of beam settings
        Returns:
            instance of Beam
        """

        # Verify that the input beam has the correct format
        self.check_inputs(input_beam)
        self.input_beam_config = input_beam

        # Indicates how the beam settings are stored
        self.style = input_beam['style']

        # Create a bmadx Particle instance using beam settings
        # There are 3 ways beam settings can be stored
        # 1: from a .dat file path inside the input_beam dictionary
        # 2: from a YAML distgen file path inside the input_beam dictionary
        # 3: from a h5 file in particlegroup format

        if self.style == 'from_file':
            filename = input_beam['beamfile']

            ## Read bmadx coords
            coords = np.loadtxt(filename)
            assert coords.shape[1] == 6, f'Error: input beam must have 6 dimension, but get {coords.shape[1]} instead'

            self._charge = input_beam['charge']
            self._init_energy = input_beam['energy']

            # Keep track of both BmadX particle format (for tracking) and Particle Group format (for calculating twiss).
            self.particle = Particle(*coords.T, 0, self._init_energy, MC2)   #BmadX Particle
            #self.particleGroup = bmadx_particles_to_openpmd(self.particle)  # Particle Group

        elif  self.style == 'distgen':
            filename = input_beam['distgen_input_file']
            # Generates a particle distribution based upon the settings in the distgen_input_file
            gen = Generator(filename)
            gen.run()
            pg = gen.particles
            self._charge = pg['charge']
            self._init_energy = np.mean(pg['energy'])

            self.particle = openpmd_to_bmadx_particles(pg, self._init_energy, 0.0, MC2)   #Bmad X particle
            #self.particleGroup = pg              # Particle Group

        else:
            ParticleGroup_h5 = input_beam['ParticleGroup_h5']
            pg = ParticleGroup(ParticleGroup_h5)

            self._charge = pg['charge']
            self._init_energy = np.mean(pg['energy'])

            self.particle = openpmd_to_bmadx_particles(pg, self._init_energy, 0.0, MC2)  # Bmad X particle
            #self.particleGroup = pg  # Particle Group

        # unchanged, initial energy and gamma
        self._init_gamma = self._init_energy/MC2
        #self._n_particle = self.particles.shape[0]

        # Used during CSR wake computations
        self.position = 0
        self.step = 0

        self.update_status()

    def check_inputs(self, input_beam):
        """
        Checks to make sure that the dictionary we are using for our inital beam settings has the correct format.
        Parameters:
            input_beam: the dictionary in question
        Returns:
            nothing if the dictionary has the correct format, if not asserts what is wrong
        """

        # The input_beam must have a style key, indicating in what format the beam parameters are stored in
        assert 'style' in input_beam, 'ERROR: input_beam must have keyword <style>'

        # The beam parameters can be stored either in another YAML file,
        if input_beam['style'] == 'from_file':
            self.required_inputs = ['style', 'beamfile', 'charge','energy']
        elif input_beam['style'] == 'distgen':
            self.required_inputs = ['style', 'distgen_input_file']
        elif input_beam['style'] == 'ParticleGroup':
            self.required_inputs = ['style', 'particleGroup_h5']
        else:
            raise Exception("input beam parsing Error: invalid input style")

        allowed_params = self.required_inputs + ['verbose']
        for input_param in input_beam:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input_beam, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

#    @profile
    def update_status(self):
        """
        Updates the internal status attributes of the object based on the current state of other related attributes
        """
        #self.particleGroup = bmadx_particles_to_openpmd(self.particle)
        self._sigma_x = self.sigma_x
        self._sigma_z = self.sigma_z
        self._slope = self.slope
        self._mean_x = self.mean_x
        self._mean_z = self.mean_z
        #self._twiss = self.twiss
        #self._sigma_energy = self.sigma_energy
        #self._mean_energy = self.mean_energy

 #   @profile
    def track(self, element, step_size, update_step=True):
        """
        Moves the beam through a step in the lattice
        Parameters:
            element: bmadx element obj
            step_size: the length of the step
        """

        # Use bmadx to move the particle object
        self.particle = track_element(self.particle, element)

        # Update our current position
        self.position += step_size

        # When a step contains 2 lattice elements, we need to call this function twice, in this case we should not uopdate the step count
        if update_step:
            self.step += 1
        self.update_status()

  #  @profile
    def apply_wakes(self, dE_dct, x_kick, xrange, zrange, step_size, transverse_on):
        """
        Apply the CSR wake to the current position of the beam
        Paramters:
            dE_dct, x_kick: array corresponding to the energy and momentum change of each csr mesh element
            xrange, zrange: flatted 2D mesh grid corresponding to the CSR mesh coordinates
            step_size: the distance between the slices for which CSR is computed
            transverse_only: booleans, indicates if the transverse wake should be applied
        """
        # TODO: add options for transverse or longitudinal kick only
        # Convert energy from J/m to eV/init_energy
        dE_E1 = step_size * dE_dct * 1e6 / self.init_energy  # self.energy in eV

        # Create an interpolator that will transfer the CSR wake from the CSR mesh to the DF mesh
        interp = RegularGridInterpolator((xrange, zrange), dE_E1, fill_value=0.0, bounds_error=False)

        # Apply the interpolator to populate the DF mesh
        dE_Es = interp(np.array([self.x_transform, self.z]).T)

        # Apply longitudinal kick, note that since the electrons are moving at near the speed of light,
        # change in momentum is roughly equal to change in energy
        pz_new = self.particle.pz + dE_Es

        # Use the same process as above to apply the transverse wake
        if transverse_on:
            dxp = step_size * x_kick * 1e6 / self.init_energy
            interp = RegularGridInterpolator((xrange, zrange), dxp, fill_value=0.0, bounds_error=False)
            dxps = interp(np.array([self.x_transform, self.z]).T)
            px_new = self.particle.px + dxps

        else:
            px_new = self.particle.px

        # Update the particle object with the new energy and momentum values
        self.particle = Particle(self.particle.x, px_new,
                                 self.particle.y, self.particle.py,
                                 self.particle.z, pz_new,
                                 self.particle.s, self.particle.p0c, self.particle.mc2)

        self.update_status()

    def frog_leap(self):
        # Todo: track half step, apply kicks, track another half step
        pass

    ### Various properties of the beam ###
    @property
    def mean_x(self):
        return np.mean(self.particle.x)

    @property
    def mean_y(self):
        return np.mean(self.particle.y)

    @property
    def sigma_x(self):
        return np.std(self.particle.x)

    @property
    def sigma_z(self):
        return np.std(self.particle.z)

    @property
    def mean_z(self):
        return np.mean(self.particle.z)

    @property
    def init_energy(self):
        return self._init_energy

    @property
    def init_gamma(self):
        return self._init_gamma

    @property
    def energy(self):
        return (self.particle.pz+1)*self.particle.p0c

    @property
    def mean_energy(self):
        return np.mean(self.energy)

    @property
    def gamma(self):
        return self.energy/MC2

    @property
    def sigma_energy(self):
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
        """
        Computers the line of best fit for (x,z) point distribution.
        Returns:
            p = [a,b] where f(x) = a*z + b
        """
        p = np.polyfit(self.z, self.x, deg=1)
        return p

    @property
    def x_transform(self):
        """
        :return: x coordinates after removing the x-z chirp (will make a tilted distribution virtical in x direction)
        """
        return self.x - np.polyval(self.slope, self.z)

    @property
    def sigma_x_transform(self):
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
        #pg.weight = np.abs(pg.weight)
        return pg
