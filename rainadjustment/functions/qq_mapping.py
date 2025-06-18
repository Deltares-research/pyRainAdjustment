# -*- coding: utf-8 -*-
"""
Functions to apply quantile mapping (QQ maping).

Available functions:
- apply_qmap_correction
- create_qmapping_factors
- derive_and_store_qmapping_factors
- preprocess_netcdf_for_qmapping
- qmapping_correction_factors
"""
import glob
import logging
import os

import numpy as np
import numpy.typing as npt
from numpy import inf
import xarray as xr

from utils.io import check_dimensions


def apply_qmap_correction(
    forecast: xr.DataArray,
    historical_data: npt.NDArray[np.float64],
    cor_factor: npt.NDArray[np.float64],
    logger: logging.Logger,
) -> tuple[npt.NDArray[np.float64], xr.DataArray]:
    """
    Apply quantile mapping correction to a forecast using historical data and correction factors.

    This function interpolates correction factors based on the position of the forecast values
    within the historical forecast distribution. It then applies the correction, capping the
    correction factor at 1000 to avoid extreme values.

    Parameters
    ----------
    forecast : xr.DataArray
        Forecast values for a specific station or grid point.
    historical_data : np.ndarray
        1D array containing the historical forecast distribution corresponding to percentiles.
    cor_factor : np.ndarray
        1D array of correction factors associated with the percentiles of the historical data.
    logger : logging.Logger
        Logger instance for logging the correction process.

    Returns
    -------
    np.ndarray
        A 2D array where the first row contains the interpolated correction factors and the
        second row contains the corrected forecast values.
    """
    logger.info("Apply quantile mapping corrections")
    correction = np.interp(forecast, historical_data, cor_factor)
    # Cap the correction factors
    apply_corr = np.where(correction > 1000, 1000, correction)
    # Also fill nans in the correction factors
    apply_corr = np.nan_to_num(apply_corr, nan=1.0)

    return np.array([correction, forecast * apply_corr])


def create_qmapping_factors(
    grid_values: xr.DataArray, obs_values: xr.DataArray
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute quantile mapping correction factors based on historical model data and observations.
    This function calculates the quantiles of both the model (grid) data and the observed data,
    and derives correction factors as the ratio of observed to modeled quantiles. These factors
    can be used to adjust future forecasts to better match observed distributions.

    Parameters
    ----------
    grid_values : xr.DataArray
        Historical model (hindcast) data for a given period and timestep.
    obs_values : xr.DataArray
        Corresponding observed values for the same period and timestep.

    Returns
    -------
    A tuple containing:
        - correction_factors : np.ndarray
            The ratio of observed to modeled quantiles (obs_q / grid_q).
        - grid_q : np.ndarray
            Quantiles computed from the forecast data.
        - obs_q : np.ndarray
            Quantiles computed from the observed data.
    """
    # Obtain the quantiles from the historical forecasts and the observations
    grid_q = np.nanquantile(grid_values.flatten(), q=np.linspace(0, 1, 201))
    obs_q = np.nanquantile(obs_values, q=np.linspace(0, 1, 201))
    # Determine the correction factors
    correction_factors = obs_q / grid_q
    # Make sure there are no infinitives in the correction factors
    correction_factors[correction_factors == -inf] = np.nan
    correction_factors[correction_factors == inf] = np.nan

    return np.array([correction_factors, grid_q, obs_q])


def derive_and_store_qmapping_factors(
    init_month: int,
    work_dir: str,
    qq_factors_folder: str,
    logger: logging.Logger,
    preprocess_files: bool = False,
    leadtime_specific: bool = False,
) -> None:
    """
    Generate quantile mapping correction factors for a gridded forecast dataset.

    This function processes historical forecast and observed rainfall data to compute
    quantile mapping correction factors on a grid. It supports optional preprocessing
    of NetCDF files and can compute corrections either aggregated across all lead times
    or separately for each lead time. Correction factors are computed and stored per
    month, so this function needs to be called for each month in a dataset.

    Parameters
    ----------
    init_month : int
        The initialization month (1 - 12) used to filter the historical forecasts.
    work_dir : str
        Working directory containing input and output subdirectories.
    qq_factors_folder : str
        The filepath to the folder where the quantile mapping correction factors should be
        stored.
    logger : logging.Logger
        Logger instance for logging progress and diagnostics.
    preprocess_files : bool, optional
        If True, preprocess and combine individual NetCDF files from the historic forecast
        directory. If False, load a pre-combined dataset. Default is False.
    leadtime_specific : bool, optional
        If True, compute correction factors separately for each lead time. If False, compute
        a single set of factors across all lead times. Default is False.

    Returns
    -------
    None, but stores a dataset containing:
    - correction_factors : Quantile-based correction factors.
    - forecast_P_at_q : Forecast quantiles.
    - obs_P_at_q : Observed quantiles.

    Notes
    -----
    - Output is saved to: [work_dir]/qq_correction_factors/grid_quantile_correction_<init_month>.nc
    """
    # First check if all paths exists
    if not os.path.isdir(os.path.join(work_dir, "input")):
        os.mkdir(os.path.join(work_dir, "input"))
    if not os.path.isdir(os.path.join(work_dir, "input", "historic_forecasts")):
        os.mkdir(os.path.join(work_dir, "input", "historic_forecasts"))
    if not os.path.isdir(qq_factors_folder):
        os.mkdir(qq_factors_folder)

    # Load the forecast datasets and pre-process if needed
    logger.info("Loading and pre-processing the datasets")
    if preprocess_files:
        files = glob.glob(os.path.join(work_dir, "input", "historic_forecasts", "*.nc"))
        grid_hist = xr.open_mfdataset(files, preprocess=preprocess_netcdf_for_qmapping)
        grid_hist.to_netcdf(os.path.join(work_dir, "input", "combined_historic_forecasts.nc"))
    else:
        grid_hist = xr.load_dataset(
            os.path.join(work_dir, "input", "combined_historic_forecasts.nc")
        )
    grid_hist = check_dimensions(grid_hist, logger=logger)

    # Make sure all dimensions are right
    grid_hist["lon"] = np.round(grid_hist.lon.values, decimals=2)
    grid_hist["lat"] = np.round(grid_hist.lat.values, decimals=2)
    select_timestamps = grid_hist.analysis_time.dt.month.values == init_month
    grid_hist = grid_hist.isel({"analysis_time": select_timestamps})
    grid_hist = grid_hist.isel({"analysis_time": grid_hist.analysis_time.dt.day.values < 27})
    logger.info("Valid times in this forecast are: %s", grid_hist.valid_time.values)

    # Load and prep-process the gridded reference data
    logger.info("Loading and pre-processing the reference dataset")
    reference_hist = xr.load_dataset(os.path.join(work_dir, "input", "reference_rainfall.nc"))
    reference_hist = check_dimensions(reference_hist, logger=logger)
    reference_hist["lon"] = np.round(reference_hist.lon.values, decimals=2)
    reference_hist["lat"] = np.round(reference_hist.lat.values, decimals=2)

    # Compute the quantile mapping correction factors
    grid_hist = grid_hist.compute()
    if not leadtime_specific:
        logger.info(
            "Computing the quantile mapping correction factors. No lead-time dependency requested."
        )
        factors = xr.apply_ufunc(
            create_qmapping_factors,
            grid_hist.P,
            reference_hist.P,
            input_core_dims=[["analysis_time", "step"], ["time"]],
            output_core_dims=[["variable", "percentile"]],
            vectorize=True,
        )
        q_array = factors
    else:
        logger.info(
            "Computing the quantile mapping correction factors. Lead-time dependency requested."
        )
        q_list = []
        for step_it, leadtime in enumerate(grid_hist.step.values):
            print(step_it)
            grid_at_step = grid_hist.isel({"step": step_it})
            factors = xr.apply_ufunc(
                create_qmapping_factors,
                grid_at_step.P,
                reference_hist.P,
                input_core_dims=[["analysis_time"], ["time"]],
                output_core_dims=[["variable", "percentile"]],
                vectorize=True,
            )
            q_list.append(factors)
        q_array = xr.concat(q_list, dim="step")

    # Make the output ready and store it
    logger.info("Store the output qq correction factors")
    q_array = q_array.assign_coords(percentile=np.linspace(0, 1, 201))
    q_array = q_array.to_dataset(dim="variable")
    q_array = q_array.rename({0: "correction_factors", 1: "forecast_P_at_q", 2: "obs_P_at_q"})
    q_array.to_netcdf(os.path.join(qq_factors_folder, f"grid_quantile_correction_{init_month}.nc"))


def preprocess_netcdf_for_qmapping(input_ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess a NetCDF dataset for quantile mapping by adjusting coordinates and dimensions.
    This function is intended to be used as a preprocessing step in `xr.open_mfdataset`.
    It modifies the dataset by:
        - Creating a new coordinate called "step", representing the lead time relative to the
            first time value.
        - Swapping the "time" dimension with "step".
        - Renaming the original "time" coordinate to "valid_time".

    Parameters
    ----------
    input_ds : xr.Dataset
        The input xarray Dataset, typically representing a single NetCDF file.

    Returns
    -------
    xr.Dataset
        The preprocessed Dataset with updated coordinates and dimensions suitable for quantile
        mapping.
    """
    # Add 'step' to the coordinates
    ds = input_ds.assign_coords({"step": ("time", input_ds.time.values - input_ds.time.values[0])})
    # Swap and rename dimensions
    ds = ds.swap_dims({"time": "step"})
    ds = ds.rename({"time": "valid_time"})

    return ds


def qmapping_correction_factors(
    grid_forecast: xr.Dataset, grid_clim_path: str
) -> tuple[npt.NDArray[np.float64], xr.DataArray]:
    """
    Apply quantile mapping correction to a forecast dataset using precomputed climatological
    factors. This function loads a climatology dataset containing quantile-based correction
    factors and forecast distributions. It then applies quantile mapping to the input forecast
    using `apply_qmap_correction` via `xr.apply_ufunc`.

    Parameters
    ----------
    grid_forecast : xr.Dataset
        Forecast dataset containing a variable `P` to be corrected.
    grid_clim_path : str
        Path to the NetCDF file containing climatological correction factors and forecast
        percentiles.

    Returns
    -------
    tuple
        A tuple containing:
        - corrections : np.ndarray
            A 2D array with correction factors and corrected forecast values.
        - xr.DataArray
            The result of applying quantile mapping to the forecast data.
    """
    # Load the dataset and get the correction factors
    climatology = xr.load_dataset(grid_clim_path)
    factors = climatology.correction_factors
    distribution = climatology.forecast_P_at_q

    # Apply the quantile mapping function
    corrections = xr.apply_ufunc(
        apply_qmap_correction,
        grid_forecast.P,
        distribution,
        factors,
        input_core_dims=[[], ["percentile"], ["percentile"]],
        output_core_dims=[["variable"]],
        vectorize=True,
    )

    return corrections
