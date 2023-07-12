import xarray as xr

def detrend_dim(da, dim, deg=1):
    """
    Detrend along a single dimension.
    
    da: array to detrend
    dim: dimension along which to detrend
    deg: degree of polynomial to fit (1 for linear fit)
    
    Adapted from the original code here:
    Author: Ryan Abernathy
    From: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    """
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit