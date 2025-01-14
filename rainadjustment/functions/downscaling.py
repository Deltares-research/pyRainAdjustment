# -*- coding: utf-8 -*-
"""
Functions to downscale gridded precipitation data using monthly climatology of rainfall.
This functionality makes use of the HydroMT python package (Eilander et al., 2023, JOSS).
"""

import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling

from utils.io import check_dimensions


def downscale_gridded_precip(precip_orig, clim_file, downscale_factor):
    """
    Parameters
    ----------
    precip_orig: str
        The netCDF file containing the gridded rainfall information for
        the last hour.
    clim_file: str
        The (monthly) climatology file to be used.
    downscale_factor: int
        The factor by which the gridded precipitation should be downscaled.
        For instance, a factor of 2 indicates that the downscaled precipitation
        will have a grid resolution that is two times finer than the original
        grid resolution.

    Returns
    ------
    precip_downscaled: Xarray Dataset
        Dataset containing the downscaled gridded precipitation values.
    """
    # First read the data
    precip_coarse = xr.open_dataset(precip_orig)
    precip_coarse = check_dimensions(precip_coarse)

    # Write the CRS of the data
    precip_coarse = precip_coarse.rio.write_crs(precip_coarse.crs.attrs["epsg_code"])

    # Get the output grid, which will be used for all reprojections later on
    new_x = precip_coarse.rio.width * downscale_factor
    new_y = precip_coarse.rio.height * downscale_factor
    precip_out = precip_coarse.rio.reproject(
        precip_coarse.rio.crs, shape=(new_y, new_x), resampling=Resampling.nearest
    )

    # Read the climatology file
    ds_clim = xr.open_dataset(clim_file)  # hydromt.open_raster(clim_file)
    try:
        ds_clim = ds_clim.rio.write_crs(ds_clim.crs.attrs["crs_wkt"])
    except AttributeError:
        try:
            ds_clim = ds_clim.rename({"spatial_ref": "crs"})
            ds_clim = ds_clim.rio.write_crs(ds_clim.crs.attrs["crs_wkt"])
        except ValueError:
            raise ValueError(
                "Coordinate reference system of climatology file not found. Make sure that the CRS is called crs or spatial_ref in the netCDF file."
            )

    # First, clip the data
    min_x = precip_out.x.min().item()
    max_x = precip_out.x.max().item()
    min_y = precip_out.y.min().item()
    max_y = precip_out.y.max().item()
    ds_clim_clipped = ds_clim.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)

    # Make sure the climatology is monthly
    if not "month" in ds_clim_clipped.dims:
        if "time" in ds_clim_clipped.dims:
            time_freq = pd.infer_freq(ds_clim_clipped["time"].to_index())
            if time_freq != "ME":
                ds_clim_clipped = ds_clim_clipped.resample(time="ME").sum()
                ds_clim_clipped = ds_clim_clipped.rename({"time": "month"})
    # Ensure that the climatology contains 12 months
    if not ds_clim_clipped["month"].size == 12:
        raise ValueError("Precip climatology does not contain 12 months.")
    assert ds_clim_clipped is not None

    # Now, calculate downscaling multiplication factor
    clim_coarse = ds_clim_clipped.rio.reproject_match(
        match_data_array=precip_coarse, resampling=Resampling.average
    ).rio.reproject_match(match_data_array=precip_out, resampling=Resampling.average)
    clim_fine = ds_clim_clipped.rio.reproject_match(
        match_data_array=precip_out, resampling=Resampling.average
    )
    downscaling_factor_grid = xr.where(clim_coarse > 0, clim_fine / clim_coarse, 1.0).fillna(1.0)

    # Multiply with monthly multiplication factor to estimate precip_downscaled
    precip_downscaled = precip_out.groupby("time.month") * downscaling_factor_grid

    # Finally, drop spatial_ref if it exists
    if "spatial_ref" in precip_downscaled:
        precip_downscaled = precip_downscaled.drop_vars("spatial_ref")

    return precip_downscaled.drop_vars("month")
