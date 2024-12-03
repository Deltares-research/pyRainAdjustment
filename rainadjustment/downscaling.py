# -*- coding: utf-8 -*-
"""
Functions to downscale gridded precipitation data using monthly climatology of rainfall.
This functionality makes use of the HydroMT python package (Eilander et al., 2023, JOSS).
"""

import numpy as np
import xarray as xr

import hydromt


def downscale_gridded_precip(precip_orig, clim_file):
    """
    Parameters
    ----------
    precip_orig: str
        The netCDF file containing the gridded rainfall information for
        the last hour.
    clim_file: str
        The (monthly) climatology file to be used.

    Returns
    ------
    precip_downscaled: Xarray Dataset
        Dataset containing the downscaled gridded precipitation values.
    """
    # First read the data to RadarDataSet classes in HydroMT
    ds_orig = xr.open_dataset(precip_orig)
    # Read as RasterDataset using HydroMT
    P_coarse = hydromt.raster.RasterDataset(ds_orig)

    # Read the climatology file
    ds_clim = hydromt.open_mfraster(clim_file, concat=True, concat_dim="time")

    # Make sure the climatology is monthly

    # # Reproject to the fine resolution of the climatology used for downscaling
    # P_fine

    # if clim is not None:
    #     # make sure first dim is month
    #     if clim.raster.dim0 != "month":
    #         clim = clim.rename({clim.raster.dim0: "month"})
    #     if not clim["month"].size == 12:
    #         raise ValueError("Precip climatology does not contain 12 months.")
    #     # set missings to NaN
    #     clim = clim.raster.mask_nodata()
    #     assert clim is not None
    #     # calculate downscaling multiplication factor
    #     clim_coarse = clim.raster.reproject_like(
    #         precip, method="average"
    #     ).raster.reproject_like(da_like, method="average")
    #     clim_fine = clim.raster.reproject_like(da_like, method="average")
    #     p_mult = xr.where(
    #         clim_coarse > 0, clim_fine / clim_coarse, 1.0
    #     ).fillna(1.0)
    #     # multiply with monthly multiplication factor
    #     p_out = p_out.groupby("time.month") * p_mult

    return precip_downscaled


# Test it


downscale_gridded_precip(
    precip_orig="c:/Users/imhof_rn/temp/FEWS_Accra/Modules/Rain_Gauge_Adjustment/input/gridded_rainfall.nc",
    clim_file="c:/Users/imhof_rn/OneDrive - Stichting Deltares/Documents/Projects/Rainfall_downscaling_correction/downscaling/worldclim/wc2.1/*.tif",
)
