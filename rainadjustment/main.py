# -*- coding: utf-8 -*-
"""
Main script that handles the gauge - gridded rainfall product
adjustment procedure. The script calls either the downscaling module, the
hindcasting adjustment procedure or the quantile mapping procedure.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any

import numpy as np
import xarray as xr

sys.path.append(os.getcwd())
from functions.adjusters import apply_adjustment, check_adjustment_factor
from functions.climatology_preprocessor import get_climatology_dataset
from functions.downscaling import downscale_gridded_precip
from functions.qq_mapping import (
    derive_and_store_qmapping_factors,
    preprocess_netcdf_for_qmapping,
    qmapping_correction_factors,
)
from utils.io import (
    check_dimensions,
    obtain_gauge_information,
    obtain_gridded_rainfall_information,
    store_as_netcdf,
)
from utils.xml_config_parser import parse_run_xml


# ----------------------------------------------------------------------- #
# The functions
# ----------------------------------------------------------------------- #
def apply_downscaling(config_xml: dict[str, Any], work_dir: str, logger: logging.Logger) -> None:
    """
    Apply statistical downscaling to gridded precipitation data using a climatology
    reference.

    Parameters
    ----------
    config_xml : dict[str, Any]
        Configuration dictionary containing all settings.
    work_dir: str
        The working directory.
    logger : logging.Logger
        Logger instance for logging progress and status messages.

    Returns
    -------
    None
        The function performs file I/O and logging but does not return any value.
    """
    # 1. Check if there is a climatology dataset, if not download it
    clim_file = get_climatology_dataset(config_xml=config_xml, work_dir=work_dir, logger=logger)

    # 2. Downscale the precpitation
    precip_downscaled = downscale_gridded_precip(
        precip_orig=os.path.join(work_dir, "ToModel", "gridded_rainfall.nc"),
        clim_file=clim_file,
        downscale_factor=config_xml["downscaling_factor"],
        logger=logger,
    )
    logger.info(
        "Gridded preciptation successfully downscaled with a factor %s",
        config_xml["downscaling_factor"],
    )

    # 3. Store the downscaled precipitation in a netCDF
    compression_settings = {
        "zlib": True,
        "complevel": 4,  # Compression level (1-9), higher means more compression
    }
    precip_downscaled.to_netcdf(
        os.path.join(work_dir, "FromModel", "downscaled_gridded_rainfall.nc"),
        encoding={var: compression_settings for var in precip_downscaled.data_vars},
    )
    logger.info("Downscaled gridded rainfall stored to a netCDF.")
    logger.info("Finished downscaling procedure.")


def apply_hindcasting_adjustment(
    config_xml: dict[str, Any], work_dir: str, logger: logging.Logger
) -> None:
    """
    Apply adjustment of historical or real-time gridded precipitation data with measurements
    from rain gauges.

    Parameters
    ----------
    config_xml : dict[str, Any]
        Configuration dictionary containing all settings.
    work_dir: str
        The working directory.
    logger : logging.Logger
        Logger instance for logging progress and status messages.

    Returns
    -------
    None
        The function performs file I/O and logging but does not return any value.
    """
    # 1. Perform some checks
    adjustment_methods = [
        "MFB",
        "Additive",
        "Multiplicative",
        "Mixed",
        "KED",
    ]
    if config_xml["adjustment_method"] not in adjustment_methods:
        logger.error(
            "Requested adjustment method not present. Select an adjustment method from %s",
            adjustment_methods,
        )
        raise KeyError("Requested adjustment method not present")
    # Make sure the threshold is always 0.0 when the adjustment_method
    # is Additive
    if config_xml["adjustment_method"] == "Additive" and config_xml["threshold"] > 0.0:
        config_xml["threshold"] = 0.0

    # 2. Get the rain gauge information
    obs_coords, obs_names, obs_values = obtain_gauge_information(
        gauge_folder=os.path.join(work_dir, "ToModel"), logger=logger
    )
    logger.info("Rain gauge information read successfully.")

    # 3. obtain gridded rainfall field
    grid_coords, grid_values, grid_shape = obtain_gridded_rainfall_information(
        grid_file=os.path.join(work_dir, "ToModel", "gridded_rainfall.nc"), logger=logger
    )
    logger.info("Gridded rainfall information read successfully.")

    # Check if obs_values and grid_values have the same number of timesteps
    if obs_values.shape[0] != grid_values.shape[0]:
        logger.error(
            "No. of supplied timesteps in observations (%s) is different from number of timesteps in gridded precip dataset (%s)",
            str(obs_values.shape[0]),
            str(grid_values.shape[0]),
        )
        raise AssertionError(
            "No. of supplied timesteps in observations is different from those in the gridded precip dataset"
        )

    # 4. perform correction
    adjustment_factor_out = []
    adjusted_grid_out = []

    for t in range(grid_values.shape[0]):
        adjusted_values = apply_adjustment(
            config_xml=config_xml,
            obs_coords=obs_coords,
            obs_values=obs_values[t],
            grid_coords=grid_coords,
            grid_values=grid_values[t],
            logger=logger,
        )

        # A final check to ensure that the adjustment has taken place.
        if np.array_equal(adjusted_values, grid_values[t]):
            if np.isfinite(grid_values).any():
                logger.info(
                    "Adjustment for time step %s out of %s has not taken place. There were too few valid gauge-grid pairs. The original grid values will be returned for this time step.",
                    str(t + 1),
                    str(grid_values.shape[0]),
                )
                adjusted_values_checked = adjusted_values.copy()
            else:
                logger.warning(
                    "Adjustment for time step %s out of %s has not taken place. The gridded rainfall only contains nans. The original grid values will be returned for this time step.",
                    str(t + 1),
                    str(grid_values.shape[0]),
                )
                adjusted_values_checked = adjusted_values * np.nan
        else:
            logger.info(
                "Adjustment for time step %s out of %s has taken place successfully.",
                str(t + 1),
                str(grid_values.shape[0]),
            )
            # Also ensure that the correction values have not been too high.
            adjusted_values_checked = check_adjustment_factor(
                adjusted_values=adjusted_values,
                original_values=grid_values[t],
                max_change_factor=config_xml["max_change_factor"],
            )
            if np.array_equal(adjusted_values_checked, adjusted_values) is False:
                logger.warning(
                    "Some of the adjusted values were above the set maximum adjustment factor."
                )

        # 5. Get the adjustment factor
        adjustment_factor = adjusted_values_checked / grid_values[t]
        # Make sure there are no negative numbers and no nans in the adjustment
        adjusted_values_checked = np.where(
            adjusted_values_checked < 0.0, 0.0, adjusted_values_checked
        )
        adjustment_factor = np.where(adjustment_factor < 0.0, 1.0, adjustment_factor)
        adjustment_factor = np.nan_to_num(adjustment_factor, nan=1.0, posinf=1.0, neginf=1.0)

        # Reshape and store as a variable
        adjustment_factor_out.append(np.reshape(adjustment_factor, grid_shape))
        adjusted_grid_out.append(np.reshape(adjusted_values_checked, grid_shape))

    # 6. Store both datasets in a netCDF
    store_as_netcdf(
        gridded_array=np.array(adjustment_factor_out),
        dataset_example=xr.open_dataset(os.path.join(work_dir, "ToModel", "gridded_rainfall.nc")),
        variable_name="adjustment_factor",
        outfile=os.path.join(work_dir, "FromModel", "adjustment_factors_gridded_rainfall.nc"),
    )
    store_as_netcdf(
        gridded_array=np.array(adjusted_grid_out),
        dataset_example=xr.open_dataset(os.path.join(work_dir, "ToModel", "gridded_rainfall.nc")),
        variable_name="P",
        outfile=os.path.join(work_dir, "FromModel", "adjusted_gridded_rainfall.nc"),
    )
    logger.info("Adjusted gridded rainfall stored to a netCDF.")
    logger.info("Finished rain gauge adjustment. %s gauges were provided.", str(len(obs_names)))


def apply_quantile_mapping(
    config_xml: dict[str, Any], work_dir: str, logger: logging.Logger
) -> None:
    """
    Apply the quantile mapping procedure to a gridded precipitation forecast. This function either
    derives the quantile mapping correction factors or applies them to the current forecast.

    Parameters
    ----------
    config_xml : dict[str, Any]
        Configuration dictionary containing all settings.
    work_dir: str
        The working directory.
    logger : logging.Logger
        Logger instance for logging progress and status messages.

    Returns
    -------
    None
        The function performs file I/O and logging but does not return any value.
    """
    # Get the filepath for the correction factors
    if config_xml["qq_filepath"] is not None:
        if os.path.exists(config_xml["qq_filepath"]):
            logger.info(
                "Quantile mapping file path provided, using the following qq mapping file: %s",
                config_xml["qq_filepath"],
            )
            qq_factors_folder = config_xml["qq_filepath"]
        else:
            logger.warning("Provided quantile mapping file path not found")
            qq_factors_folder = os.path.join(work_dir, "qq_correction_factors")
    else:
        logger.info("Using default quantile mapping file path")
        qq_factors_folder = os.path.join(work_dir, "qq_correction_factors")

    # Check if the quantile mapping factors need to be derived
    if config_xml["derive_qmapping_factors"]:
        logger.info("Requested to derive the quantile mapping factors")
        months = (
            [config_xml["qmapping_month"]]
            if config_xml["qmapping_month"] is not None
            else range(1, 13)
        )
        logger.info(
            "Deriving the quantile mapping factors for %s",
            f"month number: {months[0]}" if len(months) == 1 else "all months",
        )

        for month_num in months:
            derive_and_store_qmapping_factors(
                init_month=month_num,
                work_dir=work_dir,
                qq_factors_folder=qq_factors_folder,
                config_xml=config_xml,
                logger=logger,
                preprocess_files=True,
            )
    else:
        # Apply the qq mapping factors
        logger.info("Requested to apply the quantile mapping factors to the current forecast")
        # Load the forecast and pre-process it
        forecast = xr.load_dataset(
            os.path.join(work_dir, "ToModel", "gridded_rainfall_forecast.nc")
        )
        forecast = preprocess_netcdf_for_qmapping(forecast)
        month_int = forecast.analysis_time.dt.month.values[0]
        forecast = check_dimensions(forecast, logger=logger)
        forecast = forecast.isel(
            {"analysis_time": forecast.analysis_time.dt.month == int(month_int)}
        )
        forecast["lon"] = np.round(forecast.lon.values, decimals=2)
        forecast["lat"] = np.round(forecast.lat.values, decimals=2)

        # Apply the correction factors
        logger.info("Applying correction factors for month: %s", str(month_int))
        corrections = qmapping_correction_factors(
            grid_forecast=forecast,
            grid_clim_path=os.path.join(
                qq_factors_folder, f"grid_quantile_correction_{month_int}.nc"
            ),
            logger=logger,
        )

        # Post-process the correction factors
        corrections = corrections.to_dataset(dim="variable")
        corrections = corrections.rename({0: "adjustment_factor", 1: "P"})
        corrections = corrections.swap_dims({"step": "valid_time"})
        corrections = corrections.rename({"valid_time": "time"})

        # Store it as a netCDF
        logger.info("Store the correction factors in a netCDF")
        corrections.to_netcdf(os.path.join(work_dir, "FromModel", "corrected_forecast.nc"))


# ----------------------------------------------------------------------- #
# The main work
# ----------------------------------------------------------------------- #
def main():
    start = time.perf_counter()
    work_dir = os.getcwd()

    # Initial parser for global arguments
    global_parser = argparse.ArgumentParser(
        description="Utility to adjust a gridded rainfall product using rain gauges",
        add_help=False,
    )
    global_parser.add_argument(
        "--xml_config",
        type=str,
        help="Path to run.xml config file created by Delft-FEWS.",
    )
    global_parser.add_argument(
        "--requested_functionality",
        type=str,
        help="Requested functionality: adjustment, downscaling or qq_mapping",
    )

    # Parse known global args first
    global_args = global_parser.parse_known_args()[0]
    requested_functionality = global_args.requested_functionality

    # Set up the logger
    if not os.path.isdir(os.path.join(work_dir, "logs")):
        os.mkdir(os.path.join(work_dir, "logs"))
    logfn = os.path.join(work_dir, "logs", "log_pyRainAdjustment.txt")
    logging.basicConfig(
        filename=logfn,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    logger.info("Arguments parsed:\n %s", global_args)

    # If --xml_config is provided, parse it and prepare to use its values
    config_xml = {}
    if global_args.xml_config:
        try:
            config_xml = parse_run_xml(global_args.xml_config)
        except Exception as exception:
            logger.exception(exception, exc_info=True)
    else:
        try:
            config_xml = parse_run_xml("ToModel/adjustment_settings.xml")
        except Exception as exception:
            logger.exception(exception, exc_info=True)

    # The actual work
    try:
        # Check what functionality is requested and execute that functionality
        if requested_functionality == "adjustment":
            apply_hindcasting_adjustment(config_xml=config_xml, work_dir=work_dir, logger=logger)

        elif requested_functionality == "downscaling":
            if config_xml["downscaling_factor"] is not None:
                apply_downscaling(config_xml=config_xml, work_dir=work_dir, logger=logger)
            else:
                logger.error(
                    "clim_filepath and/or downscaling_factor were not provided, but are needed for downscaling."
                )
                raise ValueError(
                    "clim_filepath and/or downscaling_factor were not provided, but are needed for downscaling."
                )

        elif requested_functionality == "qq_mapping":
            apply_quantile_mapping(config_xml=config_xml, work_dir=work_dir, logger=logger)

        else:
            logger.error(
                "The requested functionality %s is not one of the supported options. Make sure to pick one from adjustment or downscaling.",
                requested_functionality,
            )
            raise KeyError(
                "The requested functionality %s is not one of the supported options. Make sure to pick one from adjustment or downscaling.",
                requested_functionality,
            )

    # pylint: disable=broad-exception-caught
    except Exception as exception:
        logger.exception(exception, exc_info=True)

    end = time.perf_counter()
    logger.info("Total adjustment workflow took %s minutes", ((end - start) / 60.0))


if __name__ == "__main__":
    main()
