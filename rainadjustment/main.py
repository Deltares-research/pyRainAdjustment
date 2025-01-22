# -*- coding: utf-8 -*-
"""
Main script that handles the gauge - gridded rainfall product
adjustment procedure.
"""

import argparse
import logging
import os
import time

import numpy as np
import xarray as xr

from functions.adjusters import apply_adjustment, check_adjustment_factor
from functions.downscaling import downscale_gridded_precip
from utils.io import obtain_gauge_information, obtain_gridded_rainfall_information, store_as_netcdf
from utils.xml_config_parser import parse_run_xml


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
        help="Requested functionality: adjustment or downscaling",
    )

    # Parse known global args first
    global_args = global_parser.parse_known_args()[0]
    requested_functionality = global_args.requested_functionality

    # Set up the logger
    if not os.path.isdir(os.path.join(work_dir, "logs")):
        os.mkdir(os.path.join(work_dir, "logs"))
    logfn = os.path.join(work_dir, "logs", f"log_pyRainAdjustment.txt")
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

    # The actual work
    try:
        # Check what functionality is requested and execute that functionality
        if requested_functionality == "adjustment":
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
                    f"Requested adjustment method not present. Select an adjustment method from {adjustment_methods}"
                )
                raise KeyError("Requested adjustment method not present")
            # Make sure the threshold is always 0.0 when the adjustment_method
            # is Additive
            if config_xml["adjustment_method"] == "Additive" and config_xml["threshold"] > 0.0:
                config_xml["threshold"] = 0.0

            # 2. Get the rain gauge information
            obs_coords, obs_names, obs_values = obtain_gauge_information(
                gauge_folder=os.path.join(work_dir, "input")
            )
            logger.info("Rain gauge information read successfully.")

            # 3. obtain gridded rainfall field
            grid_coords, grid_values, grid_shape = obtain_gridded_rainfall_information(
                grid_file=os.path.join(work_dir, "input", "gridded_rainfall.nc")
            )
            logger.info("Gridded rainfall information read successfully.")

            # Check if obs_values and grid_values have the same number of timesteps
            if obs_values.shape[0] != grid_values.shape[0]:
                logger.error(
                    f"No. of supplied timesteps in observations ({str(obs_values.shape[0])}) is different from number of timesteps in gridded precip dataset ({grid_values.shape[0]})"
                )
                raise AssertionError(
                    "No. of supplied timesteps in observations is different from those in the gridded precip dataset"
                )

            # 4. perform correction
            adjustment_factor_out = []
            for t in range(grid_values.shape[0]):
                adjusted_values = apply_adjustment(
                    config_xml=config_xml,
                    obs_coords=obs_coords,
                    obs_values=obs_values[t],
                    grid_coords=grid_coords,
                    grid_values=grid_values[t],
                )

                # A final check to ensure that the adjustment has taken place.
                if np.array_equal(adjusted_values, grid_values[t]):
                    if np.isfinite(grid_values).any():
                        logger.warning(
                            f"Adjustment for time step {str(t + 1)} out of {str(grid_values.shape[0])} has not taken place. There were too few valid gauge-grid pairs. The original grid values will be returned for this time step."
                        )
                        adjusted_values_checked = adjusted_values * np.nan
                    else:
                        logger.warning(
                            f"Adjustment for time step {str(t + 1)} out of {str(grid_values.shape[0])} has not taken place. The gridded rainfall only contains nans. The original grid values will be returned for this time step."
                        )
                        adjusted_values_checked = adjusted_values * np.nan
                else:
                    logger.info(
                        f"Adjustment for time step {str(t + 1)} out of {str(grid_values.shape[0])} has taken place successfully."
                    )
                    # Also ensure that the correction values have not been too high.
                    adjusted_values_checked = check_adjustment_factor(
                        adjusted_values=adjusted_values,
                        original_values=grid_values[t],
                        max_change_factor=config_xml["max_change_factor"],
                    )
                    if np.array_equal(adjusted_values_checked, adjusted_values) == False:
                        logger.warning(
                            "Some of the adjusted values were above the set maximum adjustment factor."
                        )

                # 5. Get the adjustment factor
                adjustment_factor = adjusted_values_checked / grid_values[t]
                # Make sure there are no negative numbers and no nans in the adjustment
                adjustment_factor = np.where(adjustment_factor < 0.0, 1.0, adjustment_factor)
                adjustment_factor = np.nan_to_num(
                    adjustment_factor, nan=1.0, posinf=1.0, neginf=1.0
                )

                # Reshape and store as a variable
                adjustment_factor_out.append(np.reshape(adjustment_factor, grid_shape))

            # 6. Store it in a netCDF
            store_as_netcdf(
                adjustment_factor=np.array(adjustment_factor_out),
                dataset_example=xr.open_dataset(
                    os.path.join(work_dir, "input", "gridded_rainfall.nc")
                ),
                outfile=os.path.join(work_dir, "output", "adjustment_factors_gridded_rainfall.nc"),
            )
            logger.info("Adjusted gridded rainfall stored to a netCDF.")
            logger.info(f"Finished rain gauge adjustment. {len(obs_names)} gauges were provided.")

        elif requested_functionality == "downscaling":
            # 1. Downscale the precipitation
            if (
                config_xml["clim_filepath"] is not None
                and config_xml["downscaling_factor"] is not None
            ):
                precip_downscaled = downscale_gridded_precip(
                    precip_orig=os.path.join(work_dir, "input", "gridded_rainfall.nc"),
                    clim_file=config_xml["clim_filepath"],
                    downscale_factor=config_xml["downscaling_factor"],
                )
                logger.info(
                    f"Gridded preciptation successfully downscaled with a factor {config_xml["downscaling_factor"]}"
                )

                # 2. Store the downscaled precipitation in a netCDF
                compression_settings = {
                    "zlib": True,
                    "complevel": 4,  # Compression level (1-9), higher means more compression
                }

                precip_downscaled.to_netcdf(
                    os.path.join(work_dir, "output", "downscaled_gridded_rainfall.nc"),
                    encoding={var: compression_settings for var in precip_downscaled.data_vars},
                )
                logger.info("Downscaled gridded rainfall stored to a netCDF.")
                logger.info("Finished downscaling procedure.")

            else:
                logger.error(
                    "clim_filepath and/or downscaling_factor were not provided, but are needed for downscaling."
                )
                raise ValueError(
                    "clim_filepath and/or downscaling_factor were not provided, but are needed for downscaling."
                )

        else:
            logger.error(
                f"The requested functionality '{requested_functionality}' is not one of the supported options. Make sure to pick one from adjustment or downscaling."
            )
            raise KeyError(
                f"The requested functionality '{requested_functionality}' is not one of the supported options. Make sure to pick one from adjustment or downscaling."
            )

    # pylint: disable=broad-exception-caught
    except Exception as exception:
        logger.exception(exception, exc_info=True)

    end = time.perf_counter()
    logger.info("Total adjustment workflow took %s minutes", ((end - start) / 60.0))


if __name__ == "__main__":
    main()
