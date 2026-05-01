import torch


def interpolate1d_torch(xval, data, min_x, delta_x):
    """Torch equivalent of the Numba interpolate1D. Matches boundary behavior exactly:
    uses int() truncation (toward zero) and checks x0/x1 bounds, not x bounds."""
    x_size = data.shape[0]
    x = (xval - min_x) / delta_x
    x0 = x.to(torch.long)  # truncation toward zero, same as Python int()
    x1 = torch.where(x0 == x_size - 1, x0, x0 + 1)
    xd = x - x0.to(x.dtype)

    in_bounds = (x0 >= 0) & (x1 < x_size)
    x0c = x0.clamp(0, x_size - 1)
    x1c = x1.clamp(0, x_size - 1)
    result = data[x0c] * (1 - xd) + data[x1c] * xd
    return torch.where(in_bounds, result, torch.zeros_like(result))


def interpolate3d_torch(xval, yval, zval, data, min_x, min_y, min_z,
                         delta_x, delta_y, delta_z):
    """Torch equivalent of the Numba interpolate3D. Matches boundary behavior exactly."""
    x_size, y_size, z_size = data.shape

    x = (xval - min_x) / delta_x
    y = (yval - min_y) / delta_y
    z = (zval - min_z) / delta_z

    x0 = x.to(torch.long)
    y0 = y.to(torch.long)
    z0 = z.to(torch.long)

    x1 = torch.where(x0 == x_size - 1, x0, x0 + 1)
    y1 = torch.where(y0 == y_size - 1, y0, y0 + 1)
    z1 = torch.where(z0 == z_size - 1, z0, z0 + 1)

    in_bounds = (
        (x0 >= 0) & (y0 >= 0) & (z0 >= 0) &
        (x1 < x_size) & (y1 < y_size) & (z1 < z_size)
    )

    x0c = x0.clamp(0, x_size - 1)
    x1c = x1.clamp(0, x_size - 1)
    y0c = y0.clamp(0, y_size - 1)
    y1c = y1.clamp(0, y_size - 1)
    z0c = z0.clamp(0, z_size - 1)
    z1c = z1.clamp(0, z_size - 1)

    xd = x - x0.to(x.dtype)
    yd = y - y0.to(y.dtype)
    zd = z - z0.to(z.dtype)

    c00 = data[x0c, y0c, z0c] * (1 - xd) + data[x1c, y0c, z0c] * xd
    c01 = data[x0c, y0c, z1c] * (1 - xd) + data[x1c, y0c, z1c] * xd
    c10 = data[x0c, y1c, z0c] * (1 - xd) + data[x1c, y1c, z0c] * xd
    c11 = data[x0c, y1c, z1c] * (1 - xd) + data[x1c, y1c, z1c] * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    result = c0 * (1 - zd) + c1 * zd
    return torch.where(in_bounds, result, torch.zeros_like(result))


def interpolate3d_multi_torch(xval, yval, zval, data_list, min_x, min_y, min_z,
                               delta_x, delta_y, delta_z):
    """Trilinear interpolation with shared coordinates, multiple data arrays.
    Computes indices once, gathers from each data array. All arrays must have the same shape."""
    x_size, y_size, z_size = data_list[0].shape

    x = (xval - min_x) / delta_x
    y = (yval - min_y) / delta_y
    z = (zval - min_z) / delta_z

    x0 = x.to(torch.long)
    y0 = y.to(torch.long)
    z0 = z.to(torch.long)

    x1 = torch.where(x0 == x_size - 1, x0, x0 + 1)
    y1 = torch.where(y0 == y_size - 1, y0, y0 + 1)
    z1 = torch.where(z0 == z_size - 1, z0, z0 + 1)

    in_bounds = (
        (x0 >= 0) & (y0 >= 0) & (z0 >= 0) &
        (x1 < x_size) & (y1 < y_size) & (z1 < z_size)
    )

    x0c = x0.clamp(0, x_size - 1)
    x1c = x1.clamp(0, x_size - 1)
    y0c = y0.clamp(0, y_size - 1)
    y1c = y1.clamp(0, y_size - 1)
    z0c = z0.clamp(0, z_size - 1)
    z1c = z1.clamp(0, z_size - 1)

    xd = x - x0.to(x.dtype)
    yd = y - y0.to(y.dtype)
    zd = z - z0.to(z.dtype)

    wx0 = 1 - xd
    wy0 = 1 - yd
    wz0 = 1 - zd

    results = []
    zero = torch.zeros_like(xd)
    for data in data_list:
        c00 = data[x0c, y0c, z0c] * wx0 + data[x1c, y0c, z0c] * xd
        c01 = data[x0c, y0c, z1c] * wx0 + data[x1c, y0c, z1c] * xd
        c10 = data[x0c, y1c, z0c] * wx0 + data[x1c, y1c, z0c] * xd
        c11 = data[x0c, y1c, z1c] * wx0 + data[x1c, y1c, z1c] * xd
        c0 = c00 * wy0 + c10 * yd
        c1 = c01 * wy0 + c11 * yd
        result = c0 * wz0 + c1 * zd
        results.append(torch.where(in_bounds, result, zero))
    return results


def interpolate1d_multi_torch(xval, data_list, min_x, delta_x):
    """1D linear interpolation with shared coordinates, multiple data arrays."""
    x_size = data_list[0].shape[0]
    x = (xval - min_x) / delta_x
    x0 = x.to(torch.long)
    x1 = torch.where(x0 == x_size - 1, x0, x0 + 1)
    xd = x - x0.to(x.dtype)

    in_bounds = (x0 >= 0) & (x1 < x_size)
    x0c = x0.clamp(0, x_size - 1)
    x1c = x1.clamp(0, x_size - 1)

    results = []
    zero = torch.zeros_like(xd)
    for data in data_list:
        result = data[x0c] * (1 - xd) + data[x1c] * xd
        results.append(torch.where(in_bounds, result, zero))
    return results


def interpolate2d_bilinear_torch(xval, yval, data, xgrid, ygrid):
    """Bilinear interpolation on a regular 2D grid. Replaces RegularGridInterpolator
    for the apply_wakes use case. Out-of-bounds values return 0."""
    nx, ny = data.shape
    min_x, min_y = xgrid[0], ygrid[0]
    delta_x = (xgrid[-1] - xgrid[0]) / (nx - 1)
    delta_y = (ygrid[-1] - ygrid[0]) / (ny - 1)

    x = (xval - min_x) / delta_x
    y = (yval - min_y) / delta_y

    x0 = x.to(torch.long)
    y0 = y.to(torch.long)

    x1 = torch.where(x0 == nx - 1, x0, x0 + 1)
    y1 = torch.where(y0 == ny - 1, y0, y0 + 1)

    in_bounds = (x0 >= 0) & (y0 >= 0) & (x1 < nx) & (y1 < ny)

    x0c = x0.clamp(0, nx - 1)
    x1c = x1.clamp(0, nx - 1)
    y0c = y0.clamp(0, ny - 1)
    y1c = y1.clamp(0, ny - 1)

    xd = x - x0.to(x.dtype)
    yd = y - y0.to(y.dtype)

    c0 = data[x0c, y0c] * (1 - xd) + data[x1c, y0c] * xd
    c1 = data[x0c, y1c] * (1 - xd) + data[x1c, y1c] * xd
    result = c0 * (1 - yd) + c1 * yd
    return torch.where(in_bounds, result, torch.zeros_like(result))
