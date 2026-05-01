import os
import time
from bmadx import  Drift, SBend, Quadrupole, Sextupole
from .tools import dict2hdf5
import h5py
import numpy as np
import torch
from mpi4py import MPI

from .beams import Beam
from .deposit import DF_tracker
from .interfaces import to_numpy
from .interp1D import interpolate1D
from .interp3D import interpolate3D
from .torch_interp import interpolate1d_torch, interpolate3d_torch, interpolate3d_multi_torch, interpolate1d_multi_torch
from .lattice import Lattice
from .params import Integration_params, CSR_params
from .r_gen6 import r_gen6
from .tools import full_path, isotime
from .twiss_R import twiss_R
from .yaml_parser import parse_yaml



@torch.compile
def _csr_integrand_math(r_x, r_y, xp_flat, rho_sp,
                        tau_s_x, tau_s_y, n_s_x, n_s_y, vx_obs_e,
                        tau_sp_x, tau_sp_y, n_sp_x, n_sp_y,
                        density_ret, density_x_ret, density_z_ret, vx_ret, vx_x_ret):
    """Pure arithmetic portion of the CSR integrand. Compiled to fuse element-wise kernels."""
    r = torch.sqrt(r_x ** 2 + r_y ** 2)
    scale = 1 + xp_flat * rho_sp

    vel_x = tau_s_x + vx_obs_e * n_s_x
    vel_y = tau_s_y + vx_obs_e * n_s_y
    vel_ret_x = tau_sp_x + vx_ret * n_sp_x
    vel_ret_y = tau_sp_y + vx_ret * n_sp_y

    nabla_rho_x = density_x_ret * n_sp_x + density_z_ret / scale * tau_sp_x
    nabla_rho_y = density_x_ret * n_sp_y + density_z_ret / scale * tau_sp_y

    dot_v_vr = vel_x * vel_ret_x + vel_y * vel_ret_y
    num1 = scale * ((vel_x - dot_v_vr * vel_ret_x) * nabla_rho_x +
                     (vel_y - dot_v_vr * vel_ret_y) * nabla_rho_y)
    num2 = -scale * dot_v_vr * density_ret * vx_x_ret
    iz = (num1 + num2) / r

    dn_x = n_s_x - n_sp_x
    dn_y = n_s_y - n_sp_y
    dot_dr_dn = r_x * dn_x + r_y * dn_y
    dot_n_tau = n_s_x * tau_sp_x + n_s_y * tau_sp_y

    partial_density = -(vel_ret_x * nabla_rho_x + vel_ret_y * nabla_rho_y) - density_ret * vx_x_ret

    W1 = scale * dot_dr_dn / (r ** 3) * density_ret
    W2 = scale * dot_dr_dn / (r ** 2) * partial_density
    W3 = -scale * dot_n_tau / r * partial_density
    ix = W1 + W2 + W3

    return iz, ix


def _batched_trapz2d(integrand, xp_1d, sp_1d):
    """Double trapezoid rule: ∫∫ f(xp, sp) dxp dsp for a batch of observers.
    integrand: (N, N_xp, N_sp), xp_1d: (N, N_xp), sp_1d: (N, N_sp) → (N,)"""
    dxp = torch.diff(xp_1d, dim=1)
    mid_xp = (integrand[:, :-1, :] + integrand[:, 1:, :]) / 2
    inner = (mid_xp * dxp.unsqueeze(-1)).sum(dim=1)
    dsp = torch.diff(sp_1d, dim=1)
    mid_sp = (inner[:, :-1] + inner[:, 1:]) / 2
    return (mid_sp * dsp).sum(dim=1)


class CSR2D:
    """
    The main class to calculate 2D CSR
    """

    def __init__(self, input_file=None, parallel = False, device='cpu'):

        self.device = device
        self.use_torch = (device != 'cpu')
        self.timestamp = isotime()
        if input_file:
            self.parse_input(input_file)
            self.input_file = input_file
        self.formation_length = None
        self.initialization()  # process the initial beam

        self.prefix = f'{self.CSR_params.write_name}-{self.timestamp}'

        if parallel:
            self.init_MPI()
        else:
            self.parallel = False

    def parse_input(self, input_file):
        input = parse_yaml(input_file)
        self.check_input_consistency(input)
        self.input = input
        self.beam = Beam(input['input_beam'], device=self.device)
        self.lattice = Lattice(input['input_lattice'], device=self.device)

        if 'particle_deposition' in input:
            self.DF_tracker = DF_tracker(input['particle_deposition'], device=self.device)
        else:
            self.DF_tracker = DF_tracker(device=self.device)

        if 'CSR_integration' in input:
            self.integration_params = Integration_params(input['CSR_integration'])
        else:
            self.integration_params = Integration_params()

        if 'CSR_computation' in input:
            self.CSR_params = CSR_params(input['CSR_computation'])
        else:
            self.CSR_params = CSR_params()

    def initialization(self):
        """
        deposit the initial beam
        :return:
        """
        self.DF_tracker.get_DF(x=self.beam.x, z=self.beam.z, px=self.beam.px, t=self.beam.position)
        self.DF_tracker.append_DF()
        self.DF_tracker.append_interpolant(formation_length=float('inf'),
                                           n_formation_length=self.integration_params.n_formation_length)
        #Todo: add more flexible unit conversion, for both charge and energy
        self.CSR_scaling = 8.98755e3 * self.beam.charge # charge in C (8.98755e-6 MeV/m for 1nC/m^2)
        self.init_statistics()
    def init_statistics(self):
        Nstep = self.lattice.total_steps
        self.statistics = {}
        self.statistics['twiss'] = {'alpha_x': np.zeros(Nstep),
                                    'beta_x': np.zeros(Nstep),
                                    'gamma_x': np.zeros(Nstep),
                                    'emit_x': np.zeros(Nstep),
                                    'eta_x': np.zeros(Nstep),
                                    'etap_x': np.zeros(Nstep),
                                    'norm_emit_x': np.zeros(Nstep),
                                    'alpha_y': np.zeros(Nstep),
                                    'beta_y': np.zeros(Nstep),
                                    'gamma_y': np.zeros(Nstep),
                                    'emit_y': np.zeros(Nstep),
                                    'eta_y': np.zeros(Nstep),
                                    'etap_y': np.zeros(Nstep),
                                    'norm_emit_y': np.zeros(Nstep)}

        self.statistics['slope'] = np.zeros((Nstep, 2))
        self.statistics['sigma_x'] = np.zeros(Nstep)
        self.statistics['sigma_z'] = np.zeros(Nstep)
        self.statistics['sigma_energy'] = np.zeros(Nstep)
        self.statistics['mean_x']  = np.zeros(Nstep)
        self.statistics['mean_z'] = np.zeros(Nstep)
        self.statistics['mean_energy'] = np.zeros(Nstep)

        self.update_statistics(step = 0)


        self.inbend = False
        self.afterbend = False
        self.R_rec = None
        self.phi_rec = None

    def init_MPI(self):
        self.parallel = True
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        work_size = self.CSR_params.xbins * self.CSR_params.zbins
        ave, res = divmod(work_size, mpi_size)
        self.count = [ave + 1 if p < res else ave for p in range(mpi_size)]
        displ = [sum(self.count[:p]) for p in range(mpi_size)]
        self.displ = np.array(displ)

    def check_input_consistency(self, input):
        # Todo: need modification if dipole_config.yaml format changed
        self.required_inputs = ['input_beam', 'input_lattice']

        allowed_params = self.required_inputs + ['particle_deposition', 'distribution_interpolation', 'CSR_integration',
                                                 'CSR_computation']
        for input_param in input:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def get_formation_length(self, R, sigma_z, phi = 0.0, inbend=True):
        sigma_z = self._to_float(sigma_z)
        if inbend:
            self.formation_length = (24 * (R ** 2) * sigma_z) ** (1 / 3)
        else:
            self.formation_length = (3*R**2*phi**4)/(4*(-6*sigma_z + R*phi**3))

    def get_bmadx_element(self, ele,  DL, entrance = False, exit = False):
        input_dic = self.lattice.lattice_config[ele].copy()
        input_dic.pop('nsep')
        L = input_dic.pop('L')
        type = input_dic.pop('type')


        if type == 'dipole':
            if 'angle' in input_dic.keys():
                angle = input_dic.pop('angle')
                G = angle / L

            if 'G' in input_dic.keys():
                G = input_dic.pop('G')

            if 'E1' in input_dic.keys():
                E1 = input_dic.pop('E1')
            else:
                E1 = 0

            if 'E2' in input_dic.keys():
                E2 = input_dic.pop('E2')
            else:
                E2 = 0

            if 'FRINGE_AT' in input_dic.keys():
                FRINGE_AT = input_dic.pop('FRINGE_AT')


            if entrance and exit:
                element = SBend(L = DL, P0C = self.beam.init_energy, G = G, E1 = E1, E2 = E2, FRINGE_AT = FRINGE_AT, **input_dic)

            elif entrance:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=E1, E2=0.0, FRINGE_AT = "entrance_end", **input_dic)


            elif exit:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=0.0, E2=E2, FRINGE_AT = "exit_end", **input_dic)

            else:
                element = SBend(L=DL, P0C=self.beam.init_energy, G=G, E1=0.0, E2=0.0, FRINGE_AT = "no_end", **input_dic)

        elif type == 'drift':
            element = Drift(L = DL)

        elif type == 'quad':
            K1 = input_dic.pop('K1')
            element = Quadrupole(L=DL, K1=K1, **input_dic)

        elif type == 'sextupole':
            K2 = input_dic.pop('K2')
            element = Sextupole(L=DL, K2=K2, **input_dic)
        print(element)
        return element

#    @profile
    def run(self, stop_time = None, debug = False):

        if (not self.parallel) or (self.rank == 0):
            print('Starting the DFCSR run')

        step_count = 1

        DL = self.lattice.step_size
        ele_count = 0
        skip_ele = False
        self.inbend = False
        self.afterbend = False
        self.formation_length = 0.0

        for ele in list(self.lattice.lattice_config.keys())[1:]:

            self.lattice.update(ele)
            # Todo: add sextupole, maybe Bmad Tracking?
            # -----------------------load current lattice params-----------------#
            # Pre-process the lattice params
            L = self.lattice.lattice_config[ele]['L']
            type = self.lattice.lattice_config[ele]['type']
            steps = self.lattice.steps_per_element[ele_count]
            R = float('inf')

            ####### A step over the boundary of the elements, deal with the part of the step in the previous element
            if (not skip_ele) and ele_count > 0:
                DL_1 = self.lattice.distance[ele_count - 1] - self.beam.position   # The remaining distance in last element
                #Todo: Bmadx seems to have some problems when DL is very
                if DL_1 > 1.0e-6:
                # calculate the part in the previous element
                    element = self.get_bmadx_element(ele=ele_prev, DL=DL_1, exit=True)
                    self.beam.track(element, DL_1, update_step=False)
                else:
                    DL_1 = 0.0
            # If no steps inside an element
            if steps == 0:    #If one step over the whole element
                skip_ele = True
                element = self.get_bmadx_element(ele=ele,  DL=L, exit=True, entrance = True)
                self.beam.track(element, L, update_step=False)




            if type == 'dipole':
                angle = self.lattice.lattice_config[ele]['angle']
                R = L / angle

                self.inbend = True

                self.afterbend = True
                self.R_rec = R
                self.phi_rec = angle

                self.get_formation_length(R=R, sigma_z=5*self.beam.sigma_z, inbend = True)


            else:  # If not in a bend
                self.inbend = False

                if self.afterbend:
                    #Todo: Verify the formation length in the drift
                    #self.get_formation_length(R=self.R_rec, sigma_z=5*self.beam.sigma_z, phi = self.phi_rec, inbend=False)
                    self.get_formation_length(R=self.R_rec, sigma_z=5 * self.beam.sigma_z, inbend=True)


                else:  # if it is the first drift in the lattice
                    self.formation_length += L



            distance_in_current_ele = 0.0
            # -----------------------tracking---------------------------------
            for step in range(steps):
                time0  = time.time()

                # Deal with boundary condition. A step over the boundary of two adjacent elements
                if (step == 0) and (ele_count > 0):
                    # If enter a new element, split the step

                    DL_2 = self.lattice._positions_record[step_count] - self.lattice.distance[ele_count - 1]

                    # calculate the part in the new element
                    element = self.get_bmadx_element(ele = ele,  DL = DL_2, entrance = True)
                    self.beam.track(element, DL_2)
                    distance_in_current_ele += DL_2
                    skip_ele = False    # Reset the flag

                else:
                    element = self.get_bmadx_element(ele = ele,  DL = DL)
                    # Propagate beam for one step
                    self.beam.track(element, DL)
                    distance_in_current_ele += DL


                if debug or self.CSR_params.compute_CSR:
                    self.DF_tracker.get_DF(x=self.beam.x, z=self.beam.z, px=self.beam.px, t=self.beam.position)
                    self.DF_tracker.append_DF()
                    self.DF_tracker.append_interpolant(formation_length=self.formation_length,
                                                       n_formation_length=self.integration_params.n_formation_length)
                    self.DF_tracker.build_interpolant()

                if self.CSR_params.compute_CSR:
                    if step % self.lattice.nsep[ele_count] == 0:
                        # calculate CSR mesh given beam shape
                        self.get_CSR_mesh()
                        # Calculate CSR on the mesh
                        if self.parallel:
                            self.calculate_2D_CSR_parallel()
                        elif self.use_torch:
                            self.calculate_2D_CSR_torch()
                        else:
                            self.calculate_2D_CSR()
                        # Apply CSR kick to the beam
                        if self.CSR_params.apply_CSR:
                            self.beam.apply_wakes(self.dE_dct, self.x_kick,
                                              self.CSR_xrange_transformed, self.CSR_zrange, DL*self.lattice.nsep[ele_count],
                                                  self.CSR_params.transverse_on)
                        if (self.CSR_params.write_beam == 'all' or
                                (isinstance(self.CSR_params.write_beam, list) and (step_count in self.CSR_params.write_beam))):
                            self.dump_beam(label = step_count)
                        if self.CSR_params.write_wakes:
                            self.write_wakes()

                # recording statistics at each step
                self.update_statistics(step = step_count)

                if not self.parallel or self.rank == 0:
                    print("Finish step {}, s = {},  in {} seconds".format(step_count, self.beam.position, time.time() - time0))

                step_count += 1

                if stop_time and self.beam.position > stop_time:
                    return

            ele_prev = ele
            type_prev = type

            ele_count += 1

        self.dump_beam(label='end')
        self.write_statistics()


    def get_CSR_mesh(self):
        """
        calculating the mesh of observation points by taking linear transformation
        (xmesh, zmesh) TWO 1D arrays representiong (x, z) coordinates on a linear transformed mesh
        :return:
        """
        if self.use_torch:
            return self._get_CSR_mesh_torch()

        x_transform = to_numpy(self.beam.x_transform)
        p = to_numpy(self.beam.slope)

        sig_x = np.std(x_transform)
        mean_x = np.mean(x_transform)
        sig_z = self._to_float(self.beam.sigma_z)
        mean_z = self._to_float(self.beam.mean_z)
        xlim = self.CSR_params.xlim
        zlim = self.CSR_params.zlim
        xbins = self.CSR_params.xbins
        zbins = self.CSR_params.zbins

        zrange = np.linspace(mean_z - zlim * sig_z, mean_z + zlim * sig_z, zbins)
        xrange = np.linspace(mean_x - xlim * sig_x, mean_x + xlim * sig_x, xbins)

        # Todo: check the order
        xmesh_transform, zmesh = np.meshgrid(xrange, zrange, indexing='ij')

        xmesh_transform = xmesh_transform.flatten()
        zmesh = zmesh.flatten()

        xmesh = xmesh_transform +  np.polyval(p, zmesh)

        self.CSR_xmesh = xmesh
        self.CSR_zmesh = zmesh
        self.CSR_zrange = zrange
        self.CSR_xrange_transformed = xrange

    def _get_CSR_mesh_torch(self):
        device = self.device
        x_transform = self.beam.x_transform
        p = self.beam.slope

        sig_x = torch.std(x_transform, correction=0).item()
        mean_x = x_transform.mean().item()
        sig_z = self._to_float(self.beam.sigma_z)
        mean_z = self._to_float(self.beam.mean_z)
        xlim = self.CSR_params.xlim
        zlim = self.CSR_params.zlim
        xbins = self.CSR_params.xbins
        zbins = self.CSR_params.zbins

        zrange = torch.linspace(mean_z - zlim * sig_z, mean_z + zlim * sig_z, zbins, device=device, dtype=torch.float64)
        xrange = torch.linspace(mean_x - xlim * sig_x, mean_x + xlim * sig_x, xbins, device=device, dtype=torch.float64)

        xmesh_transform, zmesh = torch.meshgrid(xrange, zrange, indexing='ij')
        xmesh_transform = xmesh_transform.flatten()
        zmesh = zmesh.flatten()

        xmesh = xmesh_transform + p[0] * zmesh + p[1]

        self.CSR_xmesh = xmesh
        self.CSR_zmesh = zmesh
        self.CSR_zrange = zrange
        self.CSR_xrange_transformed = xrange
    
#    @profile
    def calculate_2D_CSR(self):

        N = self.CSR_params.xbins*self.CSR_params.zbins
        self.dE_dct = np.zeros((N,))
        self.x_kick = np.zeros((N,))

        start_time = time.time()
        for i in range(N):

            #if i == 210:
            #    print(i)

            #if i%int(N//10) == 0:
            #    print('Complete', str(np.round(i/N*100,2)), '%')

            s = self.beam.position + self.CSR_zmesh[i]
            x = self.CSR_xmesh[i]

            self.dE_dct[i], self.x_kick[i] = self.get_CSR_wake(s,x)

        self.dE_dct = self.dE_dct.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        self.x_kick = self.x_kick.reshape((self.CSR_params.xbins, self.CSR_params.zbins))

    def calculate_2D_CSR_parallel(self):
        work_size= self.CSR_params.xbins * self.CSR_params.zbins
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        start = int(self.displ[rank])
        local_size = int(self.count[rank])

        self.dE_dct = np.zeros((work_size,))
        self.x_kick = np.zeros((work_size,))

        dE_dct_local = np.zeros((local_size,))
        x_kick_local = np.zeros(local_size, )

        start_time = time.time()
        for i in range(local_size):
            k  = start + i
            # if i == 210:
            #    print(i)

            # if i%int(N//10) == 0:
            #    print('Complete', str(np.round(i/N*100,2)), '%')

            s = self.beam.position + self.CSR_zmesh[k]
            x = self.CSR_xmesh[k]

            dE_dct_local[i], x_kick_local[i] = self.get_CSR_wake(s,x)

        comm.Allgatherv(dE_dct_local, [self.dE_dct, self.count, self.displ, MPI.DOUBLE])
        comm.Allgatherv(x_kick_local, [self.x_kick, self.count, self.displ, MPI.DOUBLE])

        self.dE_dct = self.dE_dct.reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        self.x_kick = self.x_kick.reshape((self.CSR_params.xbins, self.CSR_params.zbins))

    def calculate_2D_CSR_torch(self):
        """GPU-batched CSR wake calculation. All N observation points processed simultaneously."""
        device = self.device
        N = self.CSR_params.xbins * self.CSR_params.zbins

        s_obs = self.beam.position + self.CSR_zmesh
        x_obs = self.CSR_xmesh
        t = self.beam.position

        sigma_z = self.beam._sigma_z
        sigma_x = self.beam._sigma_x
        tan_theta = self.beam._slope[0]
        xmean = self.beam._mean_x

        x0 = (s_obs - t) * tan_theta

        n_fl = self.integration_params.n_formation_length
        fl = self.formation_length
        zbins_int = self.integration_params.zbins
        xbins_int = self.integration_params.xbins

        t01_z = torch.linspace(0, 1, zbins_int, device=device, dtype=torch.float64)
        t01_x = torch.linspace(0, 1, xbins_int, device=device, dtype=torch.float64)
        t01_x2 = torch.linspace(0, 1, 2 * xbins_int, device=device, dtype=torch.float64)

        self._prepare_csr_tensors()

        dE_dct = torch.zeros(N, device=device, dtype=torch.float64)
        x_kick = torch.zeros(N, device=device, dtype=torch.float64)

        chirp_band = abs(tan_theta) > 1

        if not chirp_band:
            s2 = s_obs - 500 * sigma_z
            s3 = s_obs - 20 * sigma_z
            s4 = s_obs + 5 * sigma_z
            s1 = torch.clamp(s2 - n_fl * fl, min=0)

            x1_w = x0 - 20 * sigma_x
            x2_w = x0 + 20 * sigma_x
            x1_n = x0 - 10 * sigma_x
            x2_n = x0 + 10 * sigma_x

            sp1 = s1.unsqueeze(1) + (s2 - s1).unsqueeze(1) * t01_z
            sp2 = s2.unsqueeze(1) + (s3 - s2).unsqueeze(1) * t01_z
            sp3 = s3.unsqueeze(1) + (s4 - s3).unsqueeze(1) * t01_z
            xp_w = x1_w.unsqueeze(1) + (x2_w - x1_w).unsqueeze(1) * t01_x2
            xp_n = x1_n.unsqueeze(1) + (x2_n - x1_n).unsqueeze(1) * t01_x

            regions = [(xp_w, sp1), (xp_n, sp2), (xp_n, sp3)]
        else:
            if tan_theta > 0:
                tan_alpha = -2 * tan_theta / (1 - tan_theta ** 2)
                d = (10 * sigma_x + xmean - x_obs) / tan_alpha
            else:
                tan_alpha = 2 * tan_theta / (1 - tan_theta ** 2)
                d = -(xmean - x_obs - 10 * sigma_x) / tan_alpha

            s4 = s_obs + 3 * sigma_z
            s3 = torch.clamp(s_obs - d, min=0)
            s2 = s3 - 200 * sigma_z
            s1 = torch.clamp(s2 - n_fl * fl, min=0)

            if tan_theta > 0:
                x1_l = x_obs + 0.1 * sigma_x
                x1_r = x_obs + 10 * sigma_x
                x2_l = x_obs - 3 * sigma_x
                x2_r = x1_l
            else:
                x1_l = x_obs - 10 * sigma_x
                x1_r = x_obs - 1 * sigma_x
                x2_l = x1_r
                x2_r = x_obs + 3 * sigma_x

            x3_l = x0 - 5 * sigma_x
            x3_r = x0 + 5 * sigma_x
            x4_l = x0 - 20 * sigma_x
            x4_r = x0 + 20 * sigma_x

            sp1 = s1.unsqueeze(1) + (s2 - s1).unsqueeze(1) * t01_z
            sp2 = s2.unsqueeze(1) + (s3 - s2).unsqueeze(1) * t01_z
            sp3 = s3.unsqueeze(1) + (s4 - s3).unsqueeze(1) * t01_z
            xp1 = x1_l.unsqueeze(1) + (x1_r - x1_l).unsqueeze(1) * t01_x
            xp2 = x2_l.unsqueeze(1) + (x2_r - x2_l).unsqueeze(1) * t01_x
            xp3 = x3_l.unsqueeze(1) + (x3_r - x3_l).unsqueeze(1) * t01_x
            xp4 = x4_l.unsqueeze(1) + (x4_r - x4_l).unsqueeze(1) * t01_x2

            regions = [(xp4, sp1), (xp3, sp2), (xp1, sp3), (xp2, sp3)]

        for xp_1d, sp_1d in regions:
            iz, ix = self._get_CSR_integrand_batched(s_obs, x_obs, t, xp_1d, sp_1d)
            dE_dct += -self.CSR_scaling * _batched_trapz2d(iz, xp_1d, sp_1d)
            x_kick += self.CSR_scaling * _batched_trapz2d(ix, xp_1d, sp_1d)

        self.dE_dct = dE_dct.cpu().numpy().reshape((self.CSR_params.xbins, self.CSR_params.zbins))
        self.x_kick = x_kick.cpu().numpy().reshape((self.CSR_params.xbins, self.CSR_params.zbins))

    def _prepare_csr_tensors(self):
        """Reference DF tracker tensors (already on GPU from build_interpolant) and lattice data."""
        df = self.DF_tracker
        self._df_density_t = df.data_density_interp
        self._df_density_x_t = df.data_density_x_interp
        self._df_density_z_t = df.data_density_z_interp
        self._df_vx_t = df.data_vx_interp
        self._df_vx_x_t = df.data_vx_x_interp
        if not hasattr(self, '_distance_t'):
            device = self.device
            self._distance_t = torch.tensor(self.lattice.distance, device=device, dtype=torch.float64)
            self._rho_t = torch.tensor(self.lattice.rho, device=device, dtype=torch.float64)

    def _get_CSR_integrand_batched(self, s_obs, x_obs, t, xp_1d, sp_1d):
        """
        Batched CSR integrand for all observers simultaneously.
        s_obs, x_obs: (N,)
        xp_1d: (N, N_xp) — per-observer source x coordinates
        sp_1d: (N, N_sp) — per-observer source s coordinates
        Returns (iz, ix) each (N, N_xp, N_sp)
        """
        device = self.device
        N = s_obs.shape[0]
        N_xp = xp_1d.shape[1]
        N_sp = sp_1d.shape[1]
        M = N_xp * N_sp

        # Build per-observer meshgrids and flatten source dims
        xp_flat = xp_1d.unsqueeze(2).expand(N, N_xp, N_sp).reshape(N, M)
        sp_flat = sp_1d.unsqueeze(1).expand(N, N_xp, N_sp).reshape(N, M)

        # Fully flatten for interpolation calls
        xp_ff = xp_flat.reshape(-1)
        sp_ff = sp_flat.reshape(-1)

        # --- Observer quantities (N,) ---

        t_obs = torch.full((N,), t, device=device, dtype=torch.float64)
        vx_obs = interpolate3d_torch(
            xval=t_obs, yval=x_obs, zval=s_obs - t,
            data=self._df_vx_t,
            min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y,
            min_z=self.DF_tracker.min_z, delta_x=self.DF_tracker.delta_x,
            delta_y=self.DF_tracker.delta_y, delta_z=self.DF_tracker.delta_z)

        lat_min_x = self.lattice.min_x
        lat_delta_x = self.lattice.delta_x
        coords_x = self.lattice.coords_t[:, 0]
        coords_y = self.lattice.coords_t[:, 1]
        nvec_x = self.lattice.n_vec_t[:, 0]
        nvec_y = self.lattice.n_vec_t[:, 1]
        tvec_x = self.lattice.tau_vec_t[:, 0]
        tvec_y = self.lattice.tau_vec_t[:, 1]

        # Fused observer geometry: 6 lookups at s_obs, indices computed once
        X0_s, Y0_s, n_s_x, n_s_y, tau_s_x, tau_s_y = interpolate1d_multi_torch(
            s_obs, [coords_x, coords_y, nvec_x, nvec_y, tvec_x, tvec_y], lat_min_x, lat_delta_x)

        # Fused source geometry: 6 lookups at sp_ff, indices computed once
        X0_sp, Y0_sp, n_sp_x, n_sp_y, tau_sp_x, tau_sp_y = interpolate1d_multi_torch(
            sp_ff, [coords_x, coords_y, nvec_x, nvec_y, tvec_x, tvec_y], lat_min_x, lat_delta_x)
        X0_sp = X0_sp.reshape(N, M)
        Y0_sp = Y0_sp.reshape(N, M)
        n_sp_x = n_sp_x.reshape(N, M)
        n_sp_y = n_sp_y.reshape(N, M)
        tau_sp_x = tau_sp_x.reshape(N, M)
        tau_sp_y = tau_sp_y.reshape(N, M)

        # Expand observer quantities for broadcasting: (N,) → (N, 1)
        X0_s = X0_s.unsqueeze(1)
        Y0_s = Y0_s.unsqueeze(1)
        n_s_x = n_s_x.unsqueeze(1)
        n_s_y = n_s_y.unsqueeze(1)
        tau_s_x = tau_s_x.unsqueeze(1)
        tau_s_y = tau_s_y.unsqueeze(1)
        x_obs_e = x_obs.unsqueeze(1)
        vx_obs_e = vx_obs.unsqueeze(1)

        # Separation vector: (N, M)
        r_x = X0_s - X0_sp + x_obs_e * n_s_x - xp_flat * n_sp_x
        r_y = Y0_s - Y0_sp + x_obs_e * n_s_y - xp_flat * n_sp_y

        # Curvature at source points: (N, M)
        rho_sp = self._rho_t[torch.searchsorted(self._distance_t, sp_flat).clamp(0, len(self._rho_t) - 1)]

        # Retarded-time: need r for t_ret, then DF lookups
        r = torch.sqrt(r_x ** 2 + r_y ** 2)
        t_ret = t - r
        t_ret_ff = t_ret.reshape(-1)
        xp_ff2 = xp_flat.reshape(-1)
        z_ret_ff = (sp_flat - t_ret).reshape(-1)

        df_min_x = self.DF_tracker.min_x
        df_min_y = self.DF_tracker.min_y
        df_min_z = self.DF_tracker.min_z
        df_dx = self.DF_tracker.delta_x
        df_dy = self.DF_tracker.delta_y
        df_dz = self.DF_tracker.delta_z

        # Fused DF lookups: 5 fields at same (t_ret, xp, z_ret), indices computed once
        density_ret, density_x_ret, density_z_ret, vx_ret, vx_x_ret = interpolate3d_multi_torch(
            t_ret_ff, xp_ff2, z_ret_ff,
            [self._df_density_t, self._df_density_x_t, self._df_density_z_t,
             self._df_vx_t, self._df_vx_x_t],
            df_min_x, df_min_y, df_min_z, df_dx, df_dy, df_dz)
        density_ret = density_ret.reshape(N, M)
        density_x_ret = density_x_ret.reshape(N, M)
        density_z_ret = density_z_ret.reshape(N, M)
        vx_ret = vx_ret.reshape(N, M)
        vx_x_ret = vx_x_ret.reshape(N, M)

        # Compiled arithmetic: fuses ~40 element-wise kernels
        iz, ix = _csr_integrand_math(
            r_x, r_y, xp_flat, rho_sp,
            tau_s_x, tau_s_y, n_s_x, n_s_y, vx_obs_e,
            tau_sp_x, tau_sp_y, n_sp_x, n_sp_y,
            density_ret, density_x_ret, density_z_ret, vx_ret, vx_x_ret)

        return iz.reshape(N, N_xp, N_sp), ix.reshape(N, N_xp, N_sp)

    def get_CSR_wake(self, s, x):

        t = self.beam.position

        sigma_z = self.beam._sigma_z
        sigma_x = self.beam._sigma_x
        tan_theta = self.beam._slope[0]

        x0 = (s-t)*self.beam._slope[0]
        xmean = self.beam._mean_x

        chirp_band = False

        if np.abs(tan_theta) <= 1:
            # Small chirp: use wide/narrow x-ranges for far/near s-regions
            s2 = s - 500 * sigma_z
            s3 = s - 20*sigma_z
            s4 = s + 5 * sigma_z
            x1_w = x0 - 20 * sigma_x
            x2_w = x0 + 20 * sigma_x
            x1_n = x0 - 10 * sigma_x
            x2_n = x0 + 10 * sigma_x

        else:
            # Large chirp: split x-range into separate regions around observer and chirp center
            chirp_band = True
            if tan_theta > 0:
                tan_alpha = -2 * tan_theta / (1 - tan_theta ** 2)
                d = (10 * sigma_x + xmean - x) / tan_alpha
            else:
                tan_alpha = 2 * tan_theta / (1 - tan_theta ** 2)
                d = -(xmean - x - 10 * sigma_x) / tan_alpha

            s4 = s + 3 * sigma_z
            s3 = np.max((0, s - d))
            s2 = s3 - 200 * sigma_z

            if tan_theta > 0:
                x1_l = x + 0.1 * sigma_x
                x1_r = x + 10 * sigma_x
                x2_l = x - 3 * sigma_x
                x2_r = x1_l
            else:
                x1_l = x - 10 * sigma_x
                x1_r = x - 1 * sigma_x
                x2_l = x1_r
                x2_r = x + 3 * sigma_x

            x3_l = x0 - 5 * sigma_x
            x3_r = x0 + 5 * sigma_x
            x4_l = x0 - 20 * sigma_x
            x4_r = x0 + 20 * sigma_x

        s1 = np.max((0, s2 - self.integration_params.n_formation_length * self.formation_length))

        if chirp_band:
            sp1 = np.linspace(s1, s2, self.integration_params.zbins)
            sp2 = np.linspace(s2, s3, self.integration_params.zbins)
            sp3 = np.linspace(s3, s4, self.integration_params.zbins)
            xp1 = np.linspace(x1_l, x1_r, self.integration_params.xbins)
            xp2 = np.linspace(x2_l, x2_r, self.integration_params.xbins)
            xp3 = np.linspace(x3_l, x3_r, self.integration_params.xbins)
            xp4 = np.linspace(x4_l, x4_r, 2*self.integration_params.xbins)

            # 4 sub-regions: (xp4,sp1), (xp3,sp2), (xp1,sp3), (xp2,sp3)
            regions = [(xp4, sp1), (xp3, sp2), (xp1, sp3), (xp2, sp3)]
            dE_dct_total = 0.0
            x_kick_total = 0.0
            for xp_1d, sp_1d in regions:
                xp_mesh, sp_mesh = np.meshgrid(xp_1d, sp_1d, indexing='ij')
                iz, ix = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh)
                dE_dct_total += -self.CSR_scaling * np.trapz(y=np.trapz(y=iz, x=xp_1d, axis=0), x=sp_1d)
                x_kick_total += self.CSR_scaling * np.trapz(y=np.trapz(y=ix, x=xp_1d, axis=0), x=sp_1d)
            return dE_dct_total, x_kick_total

        else:
            sp1 = np.linspace(s1, s2, self.integration_params.zbins)
            sp2 = np.linspace(s2, s3, self.integration_params.zbins)
            sp3 = np.linspace(s3, s4, self.integration_params.zbins)
            xp_w = np.linspace(x1_w, x2_w, 2*self.integration_params.xbins)
            xp_n = np.linspace(x1_n, x2_n, self.integration_params.xbins)

            # 3 sub-regions: wide x for far s, narrow x for near s
            regions = [(xp_w, sp1), (xp_n, sp2), (xp_n, sp3)]
            dE_dct_total = 0.0
            x_kick_total = 0.0
            for xp_1d, sp_1d in regions:
                xp_mesh, sp_mesh = np.meshgrid(xp_1d, sp_1d, indexing='ij')
                iz, ix = self.get_CSR_integrand(s=s, t=t, x=x, xp=xp_mesh, sp=sp_mesh)
                dE_dct_total += -self.CSR_scaling * np.trapz(y=np.trapz(y=iz, x=xp_1d, axis=0), x=sp_1d)
                x_kick_total += self.CSR_scaling * np.trapz(y=np.trapz(y=ix, x=xp_1d, axis=0), x=sp_1d)
            return dE_dct_total, x_kick_total
          
          
    def get_CSR_integrand(self, s, x, t, sp, xp):

        sp_flat = sp.ravel()
        xp_flat = xp.ravel()

        # Observer velocity (transverse component at observation point)
        vx = interpolate3D(xval=np.array([t]), yval=np.array([x]), zval=np.array([s-t]),
                             data=self.DF_tracker.data_vx_interp,
                             min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y,
                             min_z=self.DF_tracker.min_z,
                             delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y,
                             delta_z=self.DF_tracker.delta_z)[0]

        # Lattice geometry at observer point s (scalars)
        X0_s = interpolate1D(xval=np.array([s]), data=self.lattice.coords[:, 0], min_x=self.lattice.min_x,
                             delta_x=self.lattice.delta_x)[0]
        Y0_s = interpolate1D(xval=np.array([s]), data=self.lattice.coords[:, 1], min_x=self.lattice.min_x,
                             delta_x=self.lattice.delta_x)[0]
        n_vec_s_x = interpolate1D(xval=np.array([s]), data=self.lattice.n_vec[:, 0], min_x=self.lattice.min_x,
                                  delta_x=self.lattice.delta_x)[0]
        n_vec_s_y = interpolate1D(xval=np.array([s]), data=self.lattice.n_vec[:, 1], min_x=self.lattice.min_x,
                                  delta_x=self.lattice.delta_x)[0]
        tau_vec_s_x = interpolate1D(xval=np.array([s]), data=self.lattice.tau_vec[:, 0], min_x=self.lattice.min_x,
                                    delta_x=self.lattice.delta_x)[0]
        tau_vec_s_y = interpolate1D(xval=np.array([s]), data=self.lattice.tau_vec[:, 1], min_x=self.lattice.min_x,
                                    delta_x=self.lattice.delta_x)[0]

        # Lattice geometry at source points sp (arrays)
        X0_sp = interpolate1D(xval=sp_flat, data=self.lattice.coords[:, 0], min_x=self.lattice.min_x,
                              delta_x=self.lattice.delta_x)
        Y0_sp = interpolate1D(xval=sp_flat, data=self.lattice.coords[:, 1], min_x=self.lattice.min_x,
                              delta_x=self.lattice.delta_x)
        n_vec_sp_x = interpolate1D(xval=sp_flat, data=self.lattice.n_vec[:, 0], min_x=self.lattice.min_x,
                                   delta_x=self.lattice.delta_x)
        n_vec_sp_y = interpolate1D(xval=sp_flat, data=self.lattice.n_vec[:, 1], min_x=self.lattice.min_x,
                                   delta_x=self.lattice.delta_x)
        tau_vec_sp_x = interpolate1D(xval=sp_flat, data=self.lattice.tau_vec[:, 0], min_x=self.lattice.min_x,
                                     delta_x=self.lattice.delta_x)
        tau_vec_sp_y = interpolate1D(xval=sp_flat, data=self.lattice.tau_vec[:, 1], min_x=self.lattice.min_x,
                                     delta_x=self.lattice.delta_x)

        # Separation vector r - r'
        r_minus_rp_x = X0_s - X0_sp + x * n_vec_s_x - xp_flat * n_vec_sp_x
        r_minus_rp_y = Y0_s - Y0_sp + x * n_vec_s_y - xp_flat * n_vec_sp_y
        r_minus_rp = np.sqrt(r_minus_rp_x**2 + r_minus_rp_y**2)

        # Curvature at source points (piecewise constant per element)
        rho_sp = np.zeros(sp_flat.shape)
        for count in range(self.lattice.Nelement):
            if count == 0:
                rho_sp[sp_flat < self.lattice.distance[count]] = self.lattice.rho[count]
            else:
                rho_sp[(sp_flat < self.lattice.distance[count]) & (sp_flat >= self.lattice.distance[count - 1])] = self.lattice.rho[count]

        # Retarded-time distribution function lookups
        t_ret = t - r_minus_rp

        density_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                                    data=self.DF_tracker.data_density_interp,
                                    min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y, min_z=self.DF_tracker.min_z,
                                    delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y, delta_z=self.DF_tracker.delta_z)

        density_x_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                                      data=self.DF_tracker.data_density_x_interp,
                                      min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y, min_z=self.DF_tracker.min_z,
                                      delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y, delta_z=self.DF_tracker.delta_z)

        density_z_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                                      data=self.DF_tracker.data_density_z_interp,
                                      min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y, min_z=self.DF_tracker.min_z,
                                      delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y, delta_z=self.DF_tracker.delta_z)

        vx_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                               data=self.DF_tracker.data_vx_interp,
                               min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y, min_z=self.DF_tracker.min_z,
                               delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y, delta_z=self.DF_tracker.delta_z)

        vx_x_ret = interpolate3D(xval=t_ret, yval=xp_flat, zval=sp_flat - t_ret,
                                 data=self.DF_tracker.data_vx_x_interp,
                                 min_x=self.DF_tracker.min_x, min_y=self.DF_tracker.min_y, min_z=self.DF_tracker.min_z,
                                 delta_x=self.DF_tracker.delta_x, delta_y=self.DF_tracker.delta_y, delta_z=self.DF_tracker.delta_z)

        # Simplified velocity model: vs=1, vs_s_ret=0 (no time derivatives)
        scale_term = 1 + xp_flat * rho_sp

        velocity_x = tau_vec_s_x + vx * n_vec_s_x
        velocity_y = tau_vec_s_y + vx * n_vec_s_y

        velocity_ret_x = tau_vec_sp_x + vx_ret * n_vec_sp_x
        velocity_ret_y = tau_vec_sp_y + vx_ret * n_vec_sp_y

        nabla_density_ret_x = density_x_ret * n_vec_sp_x + density_z_ret / scale_term * tau_vec_sp_x
        nabla_density_ret_y = density_x_ret * n_vec_sp_y + density_z_ret / scale_term * tau_vec_sp_y

        # Longitudinal CSR integrand
        dot_v_vret = velocity_x * velocity_ret_x + velocity_y * velocity_ret_y
        CSR_num1 = scale_term * ((velocity_x - dot_v_vret * velocity_ret_x) * nabla_density_ret_x +
                                 (velocity_y - dot_v_vret * velocity_ret_y) * nabla_density_ret_y)
        CSR_num2 = -scale_term * dot_v_vret * density_ret * vx_x_ret
        CSR_integrand_z = (CSR_num1 + CSR_num2) / r_minus_rp

        # Transverse CSR integrand
        n_minus_np_x = n_vec_s_x - n_vec_sp_x
        n_minus_np_y = n_vec_s_y - n_vec_sp_y

        dot_dr_dn = r_minus_rp_x * n_minus_np_x + r_minus_rp_y * n_minus_np_y
        dot_n_tau = n_vec_s_x * tau_vec_sp_x + n_vec_s_y * tau_vec_sp_y

        partial_density = -(velocity_ret_x * nabla_density_ret_x + velocity_ret_y * nabla_density_ret_y) - \
                          density_ret * vx_x_ret

        W1 = scale_term * dot_dr_dn / (r_minus_rp ** 3) * density_ret
        W2 = scale_term * dot_dr_dn / (r_minus_rp ** 2) * partial_density
        W3 = -scale_term * dot_n_tau / r_minus_rp * partial_density

        CSR_integrand_x = (W1 + W2 + W3).reshape(xp.shape)
        CSR_integrand_z = CSR_integrand_z.reshape(xp.shape)

        return CSR_integrand_z, CSR_integrand_x

    def dump_beam(self, label):
        if self.parallel and self.rank != 0:
            return

        path = full_path(self.CSR_params.workdir)
        filename = os.path.join(path, f'{self.prefix}-particles-{label}.h5')

        if os.path.isfile(filename):
            os.remove(filename)
            print("Existing file " + filename + " deleted.")

        print("Beam at position {} is written to {}".format(self.beam.position, filename))

        self.beam.particle_group.write(filename)

    def write_wakes(self):

        if self.parallel and self.rank != 0:
            return

        path = full_path(self.CSR_params.workdir)

        filename = os.path.join(path, f'{self.prefix}-wakes.h5')


        if self.beam.step == 1:
            if os.path.isfile(filename):
                os.remove(filename)
                print("Existing file " + filename + " deleted.")
            print("Wakes written to ", filename)


        with h5py.File(filename, 'a') as hf:
            step = self.beam.step
            groupname = 'step_' + str(step)
            g = hf.create_group(groupname)
            g.attrs['step'] = step
            g.attrs['position']  = self.beam.position
            g.attrs['mean_gamma'] = self.beam.init_gamma
            g.attrs['beam_energy'] = self.beam.init_energy
            g.attrs['element'] = self.lattice.current_element
            g.attrs['charge'] = self.beam.charge
            g1 = g.create_group('longitudinal')
            g1.attrs['unit'] = 'MeV/m'
            xmesh_np = to_numpy(self.CSR_xmesh.reshape(self.dE_dct.shape))
            zmesh_np = to_numpy(self.CSR_zmesh.reshape(self.dE_dct.shape))
            g1.create_dataset('x_grids', data = xmesh_np)
            g1.create_dataset('z_grids', data = zmesh_np)
            g1.create_dataset('dE_dct', data = to_numpy(self.dE_dct))
            g2  = g.create_group('transverse')
            g2.attrs['unit'] = 'MeV/m'
            g2.create_dataset('x_grids', data = xmesh_np)
            g2.create_dataset('z_grids', data = zmesh_np)
            g2.create_dataset('xkicks', data = to_numpy(self.x_kick))
#    @profile
    def _to_float(self, val):
        """Extract a Python float from a torch tensor or numpy scalar."""
        if isinstance(val, torch.Tensor):
            return val.item()
        return float(val)

    def update_statistics(self, step):
        twiss = self.beam.twiss
        self.statistics['twiss']['alpha_x'][step] = twiss['alpha_x']
        self.statistics['twiss']['beta_x'][step] = twiss['beta_x']
        self.statistics['twiss']['gamma_x'][step] = twiss['gamma_x']
        self.statistics['twiss']['emit_x'][step] = twiss['emit_x']
        self.statistics['twiss']['eta_x'][step] = twiss['eta_x']
        self.statistics['twiss']['etap_x'][step] = twiss['etap_x']
        self.statistics['twiss']['norm_emit_x'][step] = twiss['norm_emit_x']
        self.statistics['twiss']['alpha_y'][step] = twiss['alpha_y']
        self.statistics['twiss']['beta_y'][step] = twiss['beta_y']
        self.statistics['twiss']['gamma_y'][step] = twiss['gamma_y']
        self.statistics['twiss']['emit_y'][step] = twiss['emit_y']
        self.statistics['twiss']['eta_y'][step] = twiss['eta_y']
        self.statistics['twiss']['etap_y'][step] = twiss['etap_y']
        self.statistics['twiss']['norm_emit_y'][step] = twiss['norm_emit_y']
        self.statistics['slope'][step, :] = self.beam._slope
        self.statistics['sigma_x'][step] = self.beam._sigma_x
        self.statistics['sigma_z'][step] = self.beam._sigma_z
        self.statistics['sigma_energy'][step] = self._to_float(self.beam.sigma_energy)
        self.statistics['mean_x'][step] = self.beam._mean_x
        self.statistics['mean_z'][step] = self.beam._mean_z
        self.statistics['mean_energy'][step] = self._to_float(self.beam.mean_energy)
    def write_statistics(self):

        if self.parallel and self.rank != 0:
            return

        path = full_path(self.CSR_params.workdir)

        filename = os.path.join(path, f'{self.prefix}-statistics.h5')

        if os.path.isfile(filename):
            os.remove(filename)
            print("Existing file " + filename + " deleted.")
        print("Statistics written to ", filename)

        with h5py.File(filename, 'w') as hf:
            hf.create_dataset(name = 'step_positions', data = self.lattice.steps_record, shape = self.lattice.steps_record.shape)
            hf.create_dataset(name='coords', data=self.lattice.coords)
            hf.create_dataset(name='n_vec', data=self.lattice.n_vec)
            hf.create_dataset(name='tau_vec', data=self.lattice.tau_vec)
            dict2hdf5(hf, self.statistics)



















