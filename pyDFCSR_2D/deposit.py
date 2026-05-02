from numba import jit
import math
import numpy as np
import torch
from collections import deque

from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter, savgol_coeffs
from .interfaces import to_numpy, to_torch

@jit(nopython = True)
def histogram_cic_1d(q1, w, nbins, bins_start, bins_end):
    """
    Return an 1D histogram of the values in `q1` weighted by `w`,
    consisting of `nbins` evenly-spaced bins between `bins_start`
    and `bins_end`. Contribution to each bins is determined by the
    CIC weighting scheme (i.e. linear weights).
    Source: https://github.com/openPMD/openPMD-viewer/blob/dev/openpmd_viewer/openpmd_timeseries/utilities.py
    """
    # Define various scalars
    bin_spacing = (bins_end - bins_start) / nbins
    inv_spacing = 1. / bin_spacing
    n_ptcl = len(w)

    # Allocate array for histogrammed data
    hist_data = np.zeros(nbins, dtype=np.float64)

    # Go through particle array and bin the data
    for i in range(n_ptcl):
        # Calculate the index of lower bin to which this particle contributes
        q1_cell = (q1[i] - bins_start) * inv_spacing
        i_low_bin = int(math.floor(q1_cell))
        # Calculate corresponding CIC shape and deposit the weight
        S_low = 1. - (q1_cell - i_low_bin)
        if (i_low_bin >= 0) and (i_low_bin < nbins):
            hist_data[i_low_bin] += w[i] * S_low
        if (i_low_bin + 1 >= 0) and (i_low_bin + 1 < nbins):
            hist_data[i_low_bin + 1] += w[i] * (1. - S_low)

    return (hist_data)


@jit(nopython = True)
def histogram_cic_2d(q1, q2, w,
                     nbins_1, bins_start_1, bins_end_1,
                     nbins_2, bins_start_2, bins_end_2):
    """
    Return an 2D histogram of the values in `q1` and `q2` weighted by `w`,
    consisting of `nbins_1` bins in the first dimension and `nbins_2` bins
    in the second dimension.
    Contribution to each bins is determined by the
    CIC weighting scheme (i.e. linear weights).
    Source:https://github.com/openPMD/openPMD-viewer/blob/dev/openpmd_viewer/openpmd_timeseries/utilities.py
    """
    # Define various scalars
    bin_spacing_1 = (bins_end_1 - bins_start_1) / nbins_1
    inv_spacing_1 = 1. / bin_spacing_1
    bin_spacing_2 = (bins_end_2 - bins_start_2) / nbins_2
    inv_spacing_2 = 1. / bin_spacing_2
    n_ptcl = len(w)

    # Allocate array for histogrammed data
    hist_data = np.zeros((nbins_1, nbins_2), dtype=np.float64)

    # Go through particle array and bin the data
    for i in range(n_ptcl):

        # Calculate the index of lower bin to which this particle contributes
        q1_cell = (q1[i] - bins_start_1) * inv_spacing_1
        q2_cell = (q2[i] - bins_start_2) * inv_spacing_2
        i1_low_bin = int(math.floor(q1_cell))
        i2_low_bin = int(math.floor(q2_cell))

        # Calculate corresponding CIC shape and deposit the weight
        S1_low = 1. - (q1_cell - i1_low_bin)
        S2_low = 1. - (q2_cell - i2_low_bin)
        if (i1_low_bin >= 0) and (i1_low_bin < nbins_1):
            if (i2_low_bin >= 0) and (i2_low_bin < nbins_2):
                hist_data[i1_low_bin, i2_low_bin] += w[i] * S1_low * S2_low
            if (i2_low_bin + 1 >= 0) and (i2_low_bin + 1 < nbins_2):
                hist_data[i1_low_bin, i2_low_bin + 1] += w[i] * S1_low * (1. - S2_low)
        if (i1_low_bin + 1 >= 0) and (i1_low_bin + 1 < nbins_1):
            if (i2_low_bin >= 0) and (i2_low_bin < nbins_2):
                hist_data[i1_low_bin + 1, i2_low_bin] += w[i] * (1. - S1_low) * S2_low
            if (i2_low_bin + 1 >= 0) and (i2_low_bin + 1 < nbins_2):
                hist_data[i1_low_bin + 1, i2_low_bin + 1] += w[i] * (1. - S1_low) * (1. - S2_low)

    return (hist_data)


def histogram_cic_2d_torch(q1, q2, w, nbins_1, bins_start_1, bins_end_1,
                           nbins_2, bins_start_2, bins_end_2):
    """GPU CIC 2D histogram via scatter_add_."""
    bin_spacing_1 = (bins_end_1 - bins_start_1) / nbins_1
    bin_spacing_2 = (bins_end_2 - bins_start_2) / nbins_2
    inv_spacing_1 = 1.0 / bin_spacing_1
    inv_spacing_2 = 1.0 / bin_spacing_2

    q1_cell = (q1 - bins_start_1) * inv_spacing_1
    q2_cell = (q2 - bins_start_2) * inv_spacing_2
    i1 = q1_cell.to(torch.long)
    i2 = q2_cell.to(torch.long)
    s1 = 1.0 - (q1_cell - i1.to(q1.dtype))
    s2 = 1.0 - (q2_cell - i2.to(q2.dtype))

    hist = torch.zeros(nbins_1 * nbins_2, device=q1.device, dtype=q1.dtype)

    corners = [
        (i1, i2, w * s1 * s2),
        (i1, i2 + 1, w * s1 * (1.0 - s2)),
        (i1 + 1, i2, w * (1.0 - s1) * s2),
        (i1 + 1, i2 + 1, w * (1.0 - s1) * (1.0 - s2)),
    ]
    for ci1, ci2, cw in corners:
        mask = (ci1 >= 0) & (ci1 < nbins_1) & (ci2 >= 0) & (ci2 < nbins_2)
        idx = ci1[mask] * nbins_2 + ci2[mask]
        hist.scatter_add_(0, idx, cw[mask])

    return hist.reshape(nbins_1, nbins_2)


@jit(nopython=True)
def histogram_tsc_2d(q1, q2, w,
                     nbins_1, bins_start_1, bins_end_1,
                     nbins_2, bins_start_2, bins_end_2):
    """2D histogram using Triangular Shaped Cloud (quadratic spline) weighting.
    Each particle contributes to 3x3=9 cells, giving continuous first derivatives."""
    bin_spacing_1 = (bins_end_1 - bins_start_1) / nbins_1
    inv_spacing_1 = 1.0 / bin_spacing_1
    bin_spacing_2 = (bins_end_2 - bins_start_2) / nbins_2
    inv_spacing_2 = 1.0 / bin_spacing_2
    n_ptcl = len(w)

    hist_data = np.zeros((nbins_1, nbins_2), dtype=np.float64)

    for i in range(n_ptcl):
        u1 = (q1[i] - bins_start_1) * inv_spacing_1
        u2 = (q2[i] - bins_start_2) * inv_spacing_2
        i1_center = int(math.floor(u1 + 0.5))
        i2_center = int(math.floor(u2 + 0.5))

        for di in range(-1, 2):
            ii = i1_center + di
            if ii < 0 or ii >= nbins_1:
                continue
            d1 = abs(u1 - ii)
            if d1 < 0.5:
                s1 = 0.75 - d1 * d1
            elif d1 < 1.5:
                s1 = 0.5 * (1.5 - d1) * (1.5 - d1)
            else:
                s1 = 0.0

            for dj in range(-1, 2):
                jj = i2_center + dj
                if jj < 0 or jj >= nbins_2:
                    continue
                d2 = abs(u2 - jj)
                if d2 < 0.5:
                    s2 = 0.75 - d2 * d2
                elif d2 < 1.5:
                    s2 = 0.5 * (1.5 - d2) * (1.5 - d2)
                else:
                    s2 = 0.0

                hist_data[ii, jj] += w[i] * s1 * s2

    return hist_data


def histogram_tsc_2d_torch(q1, q2, w, nbins_1, bins_start_1, bins_end_1,
                           nbins_2, bins_start_2, bins_end_2):
    """GPU TSC 2D histogram via scatter_add_. Quadratic spline, 3x3=9 cells per particle."""
    inv_spacing_1 = nbins_1 / (bins_end_1 - bins_start_1)
    inv_spacing_2 = nbins_2 / (bins_end_2 - bins_start_2)

    u1 = (q1 - bins_start_1) * inv_spacing_1
    u2 = (q2 - bins_start_2) * inv_spacing_2
    i1_center = torch.round(u1).to(torch.long)
    i2_center = torch.round(u2).to(torch.long)

    hist = torch.zeros(nbins_1 * nbins_2, device=q1.device, dtype=q1.dtype)

    for di in range(-1, 2):
        ii = i1_center + di
        d1 = torch.abs(u1 - ii.to(q1.dtype))
        s1 = torch.where(d1 < 0.5, 0.75 - d1 * d1,
             torch.where(d1 < 1.5, 0.5 * (1.5 - d1) * (1.5 - d1),
             torch.zeros_like(d1)))

        for dj in range(-1, 2):
            jj = i2_center + dj
            d2 = torch.abs(u2 - jj.to(q2.dtype))
            s2 = torch.where(d2 < 0.5, 0.75 - d2 * d2,
                 torch.where(d2 < 1.5, 0.5 * (1.5 - d2) * (1.5 - d2),
                 torch.zeros_like(d2)))

            mask = (ii >= 0) & (ii < nbins_1) & (jj >= 0) & (jj < nbins_2)
            idx = ii[mask] * nbins_2 + jj[mask]
            hist.scatter_add_(0, idx, (w * s1 * s2)[mask])

    return hist.reshape(nbins_1, nbins_2)


def _savgol_conv1d(data_2d, kernel, axis):
    """Apply 1D convolution along axis 0 or 1 of a 2D tensor, matching savgol_filter behavior."""
    half = kernel.shape[0] // 2
    if axis == 0:
        x = data_2d.unsqueeze(0).unsqueeze(0)
        k = kernel.reshape(1, 1, -1, 1)
        x = torch.nn.functional.pad(x, (0, 0, half, half), mode='reflect')
        return torch.nn.functional.conv2d(x, k).squeeze(0).squeeze(0)
    else:
        x = data_2d.unsqueeze(0).unsqueeze(0)
        k = kernel.reshape(1, 1, 1, -1)
        x = torch.nn.functional.pad(x, (half, half, 0, 0), mode='reflect')
        return torch.nn.functional.conv2d(x, k).squeeze(0).squeeze(0)


def _savgol_filter_2d_torch(data_2d, kernel):
    """Separable Savitzky-Golay: filter axis=0 then axis=1."""
    return _savgol_conv1d(_savgol_conv1d(data_2d, kernel, axis=0), kernel, axis=1)


def _gradient_torch(data_2d, x_grids, z_grids):
    """Central-difference gradient matching np.gradient behavior (including edges)."""
    dx = x_grids[1] - x_grids[0]
    dz = z_grids[1] - z_grids[0]

    grad_x = torch.empty_like(data_2d)
    grad_x[0] = (data_2d[1] - data_2d[0]) / dx
    grad_x[-1] = (data_2d[-1] - data_2d[-2]) / dx
    grad_x[1:-1] = (data_2d[2:] - data_2d[:-2]) / (2 * dx)

    grad_z = torch.empty_like(data_2d)
    grad_z[:, 0] = (data_2d[:, 1] - data_2d[:, 0]) / dz
    grad_z[:, -1] = (data_2d[:, -1] - data_2d[:, -2]) / dz
    grad_z[:, 1:-1] = (data_2d[:, 2:] - data_2d[:, :-2]) / (2 * dz)

    return grad_x, grad_z


class DF_tracker:
    def __init__(self, input_dic={}, device='cpu'):

        self.device = device
        self.use_torch = (device != 'cpu')
        self.configure_params(**input_dic)
        #params for current DF
        self.sigma_x = None
        self.sigma_z = None
        self.x_bounds = None
        self.z_bounds = None
        self.density = None
        self.density_x = None
        self.density_z = None
        self.vx = None
        self.vx_x = None
        self.x_grids = None
        self.z_grids = None
        self.start_time = 0.0
        self.t = 0.0

        #params for DF log
        self.slope_log = deque([])
        self.DF_log = deque([])
        self.sigma_x_log = deque([])
        self.sigma_z_log = deque([])
        self.x_bounds_log = deque([])
        self.z_bounds_log = deque([])
        self.time_log = deque([])


        #params for interpolant
        self.sigma_x_interp = None
        self.sigma_z_interp = None
        # Todo: Add chirp
        self.time_interp = deque([])
        self.density_interp = deque([])
        self.density_x_interp = deque([])
        self.density_z_interp = deque([])
        self.vx_interp = deque([])
        self.vx_x_interp = deque([])
        self.interp_start = 0
        self.interp_end = 0
        self.x_grid_interp = None
        self.z_grid_interp = None


    def configure_params(self, xbins=100, zbins=100, xlim=5, zlim=5,
                         filter_order=0, filter_window=0,
                         velocity_threhold=5, upper_limit=None,
                         grid_mode='sigma', grid_percentile=0.9995,
                         grid_padding=0.05, deposition='cic'):
        self.xbins = xbins
        self.zbins = zbins
        self.xlim = xlim
        self.zlim = zlim
        self.velocity_threhold = velocity_threhold

        self.filter_order = filter_order
        self.filter_window = filter_window
        self.upper_limit = upper_limit

        self.grid_mode = grid_mode
        self.grid_percentile = grid_percentile
        self.grid_padding = grid_padding
        self.deposition = deposition

        if self.use_torch and filter_window > 0:
            coeffs = savgol_coeffs(filter_window, filter_order)
            self._savgol_kernel = torch.tensor(coeffs, device=self.device, dtype=torch.float64)
            self._savgol_kernel_default = None
        else:
            self._savgol_kernel = None
            self._savgol_kernel_default = None

    def get_DF(self, x, z, px, t):
        if self.use_torch:
            return self._get_DF_torch(x, z, px, t)
        return self._get_DF_numpy(x, z, px, t)

    def _get_DF_numpy(self, x, z, px, t):
        x = to_numpy(x)
        z = to_numpy(z)
        px = to_numpy(px)
        if isinstance(t, torch.Tensor):
            t = t.item()
        sigma_x = np.std(x)
        sigma_z = np.std(z)
        self.sigma_x = sigma_x
        self.sigma_z = sigma_z
        self.xmean = np.mean(x)
        self.zmean = np.mean(z)

        slice_ind = np.argwhere(np.abs(z) < 0.1*sigma_z)
        slice_sigX = np.std(x[slice_ind])
        frac = sigma_x/slice_sigX
        if frac > 5:
            xbins_t = self.xbins
            zbins_t = self.zbins
            filter_window = self.filter_window
        else:
            xbins_t = 100
            zbins_t = 100
            filter_window = 5

        if self.grid_mode == 'percentile':
            lo_pct = (1 - self.grid_percentile) / 2 * 100
            hi_pct = (1 + self.grid_percentile) / 2 * 100
            x_lo, x_hi = np.percentile(x, [lo_pct, hi_pct])
            z_lo, z_hi = np.percentile(z, [lo_pct, hi_pct])
            x_range = x_hi - x_lo
            z_range = z_hi - z_lo
            x_start = x_lo - self.grid_padding * x_range
            x_end = x_hi + self.grid_padding * x_range
            z_start = z_lo - self.grid_padding * z_range
            z_end = z_hi + self.grid_padding * z_range
        else:
            x_start = self.xmean - self.xlim * sigma_x
            x_end = self.xmean + self.xlim * sigma_x
            z_start = self.zmean - self.zlim * sigma_z
            z_end = self.zmean + self.zlim * sigma_z

        x_grids = np.linspace(x_start, x_end, xbins_t)
        z_grids = np.linspace(z_start, z_end, zbins_t)
        hist_fn = histogram_tsc_2d if self.deposition == 'tsc' else histogram_cic_2d
        density = hist_fn(q1=x, q2=z, w=np.ones(x.shape),
                          nbins_1=xbins_t, bins_start_1=x_start,
                          bins_end_1=x_end,
                          nbins_2=zbins_t, bins_start_2=z_start,
                          bins_end_2=z_end)

        vx = hist_fn(q1=x, q2=z, w=px,
                     nbins_1=xbins_t, bins_start_1=x_start,
                     bins_end_1=x_end,
                     nbins_2=zbins_t, bins_start_2=z_start,
                     bins_end_2=z_end)
        threshold = np.max(density) / self.velocity_threhold
        vx[density > threshold] /= density[density > threshold]

        density = savgol_filter(x= savgol_filter(x = density, window_length=filter_window, polyorder=self.filter_order, axis = 0),
                                window_length=filter_window, polyorder=self.filter_order, axis = 1)

        vx = savgol_filter(x= savgol_filter(x = vx, window_length=filter_window, polyorder=self.filter_order, axis = 0),
                                window_length=filter_window, polyorder=self.filter_order, axis = 1)

        dsum = np.trapz(np.trapz(density, x_grids, axis=0), z_grids)
        density /= dsum

        vx[density <= threshold] = 0

        density_x, density_z = np.gradient(density, x_grids, z_grids)
        vx_x, vx_z = np.gradient(vx, x_grids, z_grids)

        density_x = savgol_filter(
            x=savgol_filter(x=density_x, window_length=filter_window, polyorder=self.filter_order, axis=0),
            window_length=filter_window, polyorder=self.filter_order, axis=1)
        density_z = savgol_filter(
            x=savgol_filter(x=density_z, window_length=filter_window, polyorder=self.filter_order, axis=0),
            window_length=filter_window, polyorder=self.filter_order, axis=1)

        vx_x = savgol_filter(
            x=savgol_filter(x=vx_x, window_length=filter_window, polyorder=self.filter_order, axis=0),
            window_length=filter_window, polyorder=self.filter_order, axis=1)

        threshold = np.max(density) / self.velocity_threhold * 8
        vx_x[density < threshold] =  np.mean(vx_x[density > threshold])

        self.x_grids = x_grids
        self.z_grids = z_grids
        self.density = density
        self.vx = vx
        self.density_x = density_x
        self.density_z = density_z
        self.vx_x = vx_x
        self.x_bounds = (x_start, x_end)
        self.z_bounds = (z_start, z_end)
        self.t = t

    def _get_savgol_kernel(self, window):
        """Get or create savgol kernel for the given window size."""
        if window == self.filter_window and self._savgol_kernel is not None:
            return self._savgol_kernel
        if self._savgol_kernel_default is not None and self._savgol_kernel_default.shape[0] == window:
            return self._savgol_kernel_default
        self._savgol_kernel_default = torch.tensor(
            savgol_coeffs(window, self.filter_order), device=self.device, dtype=torch.float64)
        return self._savgol_kernel_default

    def _get_DF_torch(self, x, z, px, t):
        if isinstance(t, torch.Tensor):
            t = t.item()
        device = self.device

        sigma_x = torch.std(x, correction=0).item()
        sigma_z = torch.std(z, correction=0).item()
        self.sigma_x = sigma_x
        self.sigma_z = sigma_z
        self.xmean = x.mean().item()
        self.zmean = z.mean().item()

        slice_mask = torch.abs(z) < 0.1 * sigma_z
        slice_sigX = torch.std(x[slice_mask], correction=0).item()
        frac = sigma_x / slice_sigX
        if frac > 5:
            xbins_t = self.xbins
            zbins_t = self.zbins
            filter_window = self.filter_window
        else:
            xbins_t = 100
            zbins_t = 100
            filter_window = 5

        kernel = self._get_savgol_kernel(filter_window)

        if self.grid_mode == 'percentile':
            lo_q = (1 - self.grid_percentile) / 2
            hi_q = (1 + self.grid_percentile) / 2
            x_lo = torch.quantile(x, lo_q).item()
            x_hi = torch.quantile(x, hi_q).item()
            z_lo = torch.quantile(z, lo_q).item()
            z_hi = torch.quantile(z, hi_q).item()
            x_range = x_hi - x_lo
            z_range = z_hi - z_lo
            x_start = x_lo - self.grid_padding * x_range
            x_end = x_hi + self.grid_padding * x_range
            z_start = z_lo - self.grid_padding * z_range
            z_end = z_hi + self.grid_padding * z_range
        else:
            x_start = self.xmean - self.xlim * sigma_x
            x_end = self.xmean + self.xlim * sigma_x
            z_start = self.zmean - self.zlim * sigma_z
            z_end = self.zmean + self.zlim * sigma_z

        x_grids = torch.linspace(x_start, x_end, xbins_t, device=device, dtype=torch.float64)
        z_grids = torch.linspace(z_start, z_end, zbins_t, device=device, dtype=torch.float64)

        ones = torch.ones_like(x)
        hist_fn = histogram_tsc_2d_torch if self.deposition == 'tsc' else histogram_cic_2d_torch
        density = hist_fn(x, z, ones,
                          xbins_t, x_start, x_end,
                          zbins_t, z_start, z_end)
        vx = hist_fn(x, z, px,
                     xbins_t, x_start, x_end,
                     zbins_t, z_start, z_end)

        threshold = density.max() / self.velocity_threhold
        high_mask = density > threshold
        vx[high_mask] /= density[high_mask]

        density = _savgol_filter_2d_torch(density, kernel)
        vx = _savgol_filter_2d_torch(vx, kernel)

        dsum = torch.trapezoid(torch.trapezoid(density, x_grids, dim=0), z_grids)
        density /= dsum

        vx[density <= threshold] = 0

        density_x, density_z = _gradient_torch(density, x_grids, z_grids)
        vx_x, _ = _gradient_torch(vx, x_grids, z_grids)

        density_x = _savgol_filter_2d_torch(density_x, kernel)
        density_z = _savgol_filter_2d_torch(density_z, kernel)
        vx_x = _savgol_filter_2d_torch(vx_x, kernel)

        threshold2 = density.max() / self.velocity_threhold * 8
        low_mask = density < threshold2
        vx_x[low_mask] = vx_x[~low_mask].mean()

        self.x_grids = x_grids
        self.z_grids = z_grids
        self.density = density
        self.vx = vx
        self.density_x = density_x
        self.density_z = density_z
        self.vx_x = vx_x
        self.x_bounds = (x_start, x_end)
        self.z_bounds = (z_start, z_end)
        self.t = t

    def append_DF(self):
        """
        append current DF to the log
        :return:
        """
        self.DF_log.append((self.x_grids, self.z_grids, self.density, self.vx, self.density_x, self.density_z, self.vx_x))
        self.time_log.append(self.t)
        self.sigma_x_log.append(self.sigma_x)
        self.sigma_z_log.append(self.sigma_z)
        self.x_bounds_log.append(self.x_bounds)
        self.z_bounds_log.append(self.z_bounds)
        self.end_time = self.t

    def pop_left_DF(self, new_start_time):
        """
        pop history of DFs until new_start_time
        :param new_start_time:
        :return:
        """
        # remove outdated DF log
        while self.start_time < new_start_time:
            self.DF_log.popleft()
            self.time_log.popleft()
            self.sigma_x_log.popleft()
            self.sigma_z_log.popleft()
            self.x_bounds_log.popleft()
            self.z_bounds_log.popleft()
            self.start_time = self.time_log[0]

        #  remove outdated interpolant
        while self.interp_start < new_start_time:
            self.density_interp.popleft()
            self.density_x_interp.popleft()
            self.density_z_interp.popleft()
            self.vx_interp.popleft()
            self.vx_x_interp.popleft()
            self.time_interp.popleft()
            self.interp_start = self.time_interp[0]


    def pop_right_DF(self):
        """
        pop the newest DF from the log
        :return:
        """
        self.DF_log.pop()
        self.time_log.pop()
        self.sigma_x_log.pop()
        self.sigma_z_log.pop()
        self.x_bounds_log.pop()
        self.z_bounds_log.pop()
        self.end_time = self.time_log[-1]



    def DF_interp(self, DF, x_grid_interp=None, z_grid_interp=None, x_grids=None, z_grids=None, fill_value=0.0):
        if x_grids is None:
            x_grids = self.x_grids
        if z_grids is None:
            z_grids = self.z_grids
        if x_grid_interp is None:
            x_grid_interp = self.x_grid_interp
        if z_grid_interp is None:
            z_grid_interp = self.z_grid_interp

        if self.use_torch:
            return self._DF_interp_torch(DF, x_grid_interp, z_grid_interp, x_grids, z_grids, fill_value)

        X, Z = np.meshgrid(x_grid_interp, z_grid_interp, indexing='ij')
        interp = RegularGridInterpolator((x_grids, z_grids), DF, method='linear', fill_value=fill_value, bounds_error=False)
        return interp((X, Z))

    def _DF_interp_torch(self, DF, x_grid_interp, z_grid_interp, x_grids, z_grids, fill_value):
        """Bilinear grid-to-grid interpolation on GPU."""
        from .torch_interp import interpolate2d_bilinear_torch
        device = self.device
        data_t = DF if isinstance(DF, torch.Tensor) else torch.tensor(DF, device=device, dtype=torch.float64)
        xg_t = x_grids if isinstance(x_grids, torch.Tensor) else torch.tensor(x_grids, device=device, dtype=torch.float64)
        zg_t = z_grids if isinstance(z_grids, torch.Tensor) else torch.tensor(z_grids, device=device, dtype=torch.float64)
        xi_t = x_grid_interp if isinstance(x_grid_interp, torch.Tensor) else torch.tensor(x_grid_interp, device=device, dtype=torch.float64)
        zi_t = z_grid_interp if isinstance(z_grid_interp, torch.Tensor) else torch.tensor(z_grid_interp, device=device, dtype=torch.float64)

        X, Z = torch.meshgrid(xi_t, zi_t, indexing='ij')
        result = interpolate2d_bilinear_torch(X.reshape(-1), Z.reshape(-1), data_t, xg_t, zg_t)
        if fill_value != 0.0:
            min_x, max_x = xg_t[0], xg_t[-1]
            min_z, max_z = zg_t[0], zg_t[-1]
            oob = (X.reshape(-1) < min_x) | (X.reshape(-1) > max_x) | (Z.reshape(-1) < min_z) | (Z.reshape(-1) > max_z)
            result[oob] = fill_value
        return result.reshape(X.shape)


    def _interp_grid_covers_current(self):
        """Check if the existing interpolation grid still adequately covers the current DF bounds."""
        if self.x_grid_interp is None or self.z_grid_interp is None:
            return False
        x_s, x_e = self.x_bounds
        z_s, z_e = self.z_bounds
        if x_s < self.x_grid_interp[0] or x_e > self.x_grid_interp[-1]:
            return False
        if z_s < self.z_grid_interp[0] or z_e > self.z_grid_interp[-1]:
            return False
        x_range_interp = self.x_grid_interp[-1] - self.x_grid_interp[0]
        z_range_interp = self.z_grid_interp[-1] - self.z_grid_interp[0]
        x_range_current = x_e - x_s
        z_range_current = z_e - z_s
        if x_range_interp > 2 * x_range_current or z_range_interp > 2 * z_range_current:
            return False
        return True

    def append_interpolant(self, formation_length, n_formation_length):
        start_point = np.amax(a=(0, self.end_time - n_formation_length * formation_length))
        self.pop_left_DF(new_start_time=start_point)

        if self.grid_mode == 'percentile':
            no_rebuild = self._interp_grid_covers_current()
        else:
            no_rebuild = (self.sigma_x_interp and self.sigma_z_interp and
                          2 > self.sigma_x/self.sigma_x_interp > 1/2 and
                          2 > self.sigma_z/self.sigma_z_interp > 1/2)

        if no_rebuild:
            self.time_interp.append(self.t)
            current_density_interp = self.DF_interp(DF = self.density)
            current_density_x_interp = self.DF_interp(DF = self.density_x)
            current_density_z_interp = self.DF_interp(DF = self.density_z)
            current_vx_interp = self.DF_interp(DF = self.vx)
            vx_x_fill = self.vx_x.mean().item() if isinstance(self.vx_x, torch.Tensor) else np.mean(self.vx_x)
            current_vx_x_interp = self.DF_interp(DF = self.vx_x, fill_value=vx_x_fill)
            self.density_interp.append(current_density_interp)
            self.density_x_interp.append(current_density_x_interp)
            self.density_z_interp.append(current_density_z_interp)
            self.vx_interp.append(current_vx_interp)
            self.vx_x_interp.append(current_vx_x_interp)

        else:
            print('start reinterpolation. number of slice', str(len(self.time_log)))

            if self.grid_mode == 'percentile':
                x_starts = [b[0] for b in self.x_bounds_log]
                x_ends = [b[1] for b in self.x_bounds_log]
                z_starts = [b[0] for b in self.z_bounds_log]
                z_ends = [b[1] for b in self.z_bounds_log]

                x_ranges = [e - s for s, e in self.x_bounds_log]
                z_ranges = [e - s for s, e in self.z_bounds_log]
                t_x = max(x_ranges) / min(x_ranges)
                t_z = max(z_ranges) / min(z_ranges)

                xbins = int(500 * t_x)
                zbins = int(500 * t_z)
                if isinstance(self.upper_limit, int):
                    xbins = min(xbins, self.upper_limit)
                    zbins = min(zbins, self.upper_limit)

                print("xbins =", xbins, " zbins = ", zbins)

                x_total_range = max(x_ends) - min(x_starts)
                z_total_range = max(z_ends) - min(z_starts)
                x_margin = self.grid_padding * x_total_range
                z_margin = self.grid_padding * z_total_range
                if self.use_torch:
                    self.x_grid_interp = torch.linspace(min(x_starts) - x_margin, max(x_ends) + x_margin, xbins, device=self.device, dtype=torch.float64)
                    self.z_grid_interp = torch.linspace(min(z_starts) - z_margin, max(z_ends) + z_margin, zbins, device=self.device, dtype=torch.float64)
                else:
                    self.x_grid_interp = np.linspace(min(x_starts) - x_margin, max(x_ends) + x_margin, xbins)
                    self.z_grid_interp = np.linspace(min(z_starts) - z_margin, max(z_ends) + z_margin, zbins)
            else:
                max_sigma_x = np.max(self.sigma_x_log)
                min_sigma_x = np.min(self.sigma_x_log)
                max_sigma_z = np.max(self.sigma_z_log)
                min_sigma_z = np.min(self.sigma_z_log)

                t_x = max_sigma_x / min_sigma_x
                t_z = max_sigma_z / min_sigma_z

                xbins = int(500 * t_x)
                zbins = int(500 * t_z)

                if isinstance(self.upper_limit, int):
                    xbins = min(xbins, self.upper_limit)
                    zbins = min(zbins, self.upper_limit)

                print("xbins =", xbins, " zbins = ", zbins)

                self.sigma_x_interp = max_sigma_x
                self.sigma_z_interp = max_sigma_z

                if self.use_torch:
                    self.x_grid_interp = torch.linspace(self.xmean-5*self.sigma_x_interp, self.xmean + 5*self.sigma_x_interp, xbins, device=self.device, dtype=torch.float64)
                    self.z_grid_interp = torch.linspace(self.zmean -5* self.sigma_z_interp, self.zmean + 5* self.sigma_z_interp, zbins, device=self.device, dtype=torch.float64)
                else:
                    self.x_grid_interp = np.linspace(self.xmean-5*self.sigma_x_interp, self.xmean + 5*self.sigma_x_interp, xbins)
                    self.z_grid_interp = np.linspace(self.zmean -5* self.sigma_z_interp, self.zmean + 5* self.sigma_z_interp, zbins)

            self.density_interp = deque([])
            self.density_x_interp = deque([])
            self.density_z_interp = deque([])
            self.vx_interp = deque([])
            self.vx_x_interp = deque([])
            self.time_interp = self.time_log.copy()

            for x_grids, z_grids, density, vx, density_x, density_z, vx_x in self.DF_log:
                current_density_interp = self.DF_interp(DF=density, x_grids = x_grids, z_grids = z_grids)
                current_density_x_interp = self.DF_interp(DF=density_x, x_grids = x_grids, z_grids = z_grids)
                current_density_z_interp = self.DF_interp(DF=density_z, x_grids = x_grids, z_grids = z_grids)
                current_vx_interp = self.DF_interp(DF=vx, x_grids = x_grids, z_grids = z_grids)
                vx_x_fill = vx_x.mean().item() if isinstance(vx_x, torch.Tensor) else np.mean(vx_x)
                current_vx_x_interp = self.DF_interp(DF=vx_x, x_grids = x_grids, z_grids = z_grids, fill_value=vx_x_fill)

                self.density_interp.append(current_density_interp)
                self.density_x_interp.append(current_density_x_interp)
                self.density_z_interp.append(current_density_z_interp)
                self.vx_interp.append(current_vx_interp)
                self.vx_x_interp.append(current_vx_x_interp)


    def build_interpolant(self):
        """
        build interpolant for CSR intergration with the 3D matrix self.*_interp
        :return:
        """
        #Todo: check fill value
        #Todo: Important! consider faster 3D interpolation (Cython and parellel with prange, GIL release) https://github.com/jglaser/interp3d, https://ndsplines.readthedocs.io/en/latest/compare.html
        #Todo: Probably accelerate trapz with jit (parallel) https://berkeley-stat159-f17.github.io/stat159-f17/lectures/09-intro-numpy/trapezoid..html
        #Todo: Parallel cic with jit?
        #Todo: Also think about accelerate all numpy with numba.https://towardsdatascience.com/supercharging-numpy-with-numba-77ed5b169240
        #Todo: Also consider GPU acceleration of trapz and scipy regulargridinterpolant with Cupy
        #self.F_density = RegularGridInterpolator((self.time_interp, self.x_grid_interp, self.z_grid_interp),
        #                                         self.density_interp, fill_value= 0.0,bounds_error=False)
        #self.F_density_x = RegularGridInterpolator((self.time_interp, self.x_grid_interp, self.z_grid_interp),
        #                                         self.density_x_interp, fill_value= 0.0,bounds_error=False)
        #self.F_density_z = RegularGridInterpolator((self.time_interp, self.x_grid_interp, self.z_grid_interp),
        #                                           self.density_z_interp, fill_value= 0.0,bounds_error=False)
        #self.F_vx = RegularGridInterpolator((self.time_interp, self.x_grid_interp, self.z_grid_interp),
        #                                           self.vx_interp, fill_value= 0.0,bounds_error=False)
        #self.F_vx_x= RegularGridInterpolator((self.time_interp, self.x_grid_interp, self.z_grid_interp),
        #                                           self.vx_x_interp, fill_value= 0.0,bounds_error=False)
        self.min_x, self.max_x = self.time_interp[0], self.time_interp[-1]
        self.min_y, self.max_y = self.x_grid_interp[0], self.x_grid_interp[-1]
        self.min_z, self.max_z = self.z_grid_interp[0], self.z_grid_interp[-1]
        self.delta_x = (self.max_x - self.min_x) / (len(self.time_interp) - 1)
        self.delta_y = (self.max_y - self.min_y) / (self.x_grid_interp.shape[0] - 1)
        self.delta_z =  (self.max_z - self.min_z) / (self.z_grid_interp.shape[0] - 1)
        if self.use_torch:
            self.data_density_interp = torch.stack(list(self.density_interp))
            self.data_density_z_interp = torch.stack(list(self.density_z_interp))
            self.data_density_x_interp = torch.stack(list(self.density_x_interp))
            self.data_vx_interp = torch.stack(list(self.vx_interp))
            self.data_vx_x_interp = torch.stack(list(self.vx_x_interp))
        else:
            self.data_density_interp = np.array(self.density_interp)
            self.data_density_z_interp = np.array(self.density_z_interp)
            self.data_density_x_interp = np.array(self.density_x_interp)
            self.data_vx_interp = np.array(self.vx_interp)
            self.data_vx_x_interp = np.array(self.vx_x_interp)






