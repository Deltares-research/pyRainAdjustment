# -*- coding: utf-8 -*-
"""
Functions to downscale gridded precipitation data using monthly climatology of rainfall.
This functionality makes use of the HydroMT python package (Eilander et al., 2023, JOSS).

Available functions:
- get_climatology_dataset
- __post_process_clim_files
"""

import os
import zipfile

import numpy as np
import requests
import rioxarray
import xarray as xr


def get_climatology_dataset(config_xml, work_dir, logger):
    """
    Function to either find the filepath or download a climatology dataset.
    Parameters
    ----------
    config_xml: dict
        Dictionary containing the adjustment settings.
    work_dir: str
        The filepath to the working directory where the module is run.
    logger: logging instance
        Logger for log messages, passed on from the main.py script.

    Returns
    ------
    climatology_file: str (filepath)
        The filepath to the climatogy file, either downloaded or already stored.
    """
    # Try to import datasets, if None are given, continue and download a monthly
    # climatological dataset
    if config_xml["clim_filepath"] is not None:
        # Check if the provided clim_filepath exists, if not, a climatology file
        # will still be downloaded
        if os.path.exists(config_xml["clim_filepath"]):
            logger.info(
                "Climatology file path provided, using the following clim file: %s",
                config_xml["clim_filepath"],
            )
            return config_xml["clim_filepath"]
        logger.warning(
            "Provided climatology file path not found, continue with downloading monthly climatology"
        )

    # Check if climatology folder exists - if not, create it
    if not os.path.isdir(os.path.join(work_dir, "clim")):
        os.mkdir(os.path.join(work_dir, "clim"))

    # Download the worldclim climatology data
    url = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_prec.zip"
    logger.info("Downloading the wordclim dataset from: %s", url)
    filepath = os.path.join(
        work_dir, "clim", "wc2.1_30s_prec.zip"
    )  # the name you want to save file as
    requests_response = requests.get(url, timeout=10)  # making requests to server
    with open(filepath, "wb") as f:  # opening a file handler to create new file
        logger.info("Downloading the worldclim climatology files")
        f.write(requests_response.content)  # writing content to file

    # Unzip it
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        logger.info("Unzipping the worldclim climatology files")
        zip_ref.extractall(os.path.join(work_dir, "clim"))

    # Directly remove the zipped file
    os.remove(filepath)
    logger.info("Unzipping successful, original climatology zipped file removed")

    # We now have a a list of monthly tifs. Process them into one
    # netCDF file.
    logger.info("Converting the climatology geoTIFF files into one netCDF file")
    return __post_process_clim_files(os.path.join(work_dir, "clim"), logger=logger)


def __post_process_clim_files(clim_dir, logger):
    """
    Function to convert individual monthly precipitation climatology files
    from geoTIFF into one netCDF file.
    Parameters
    ----------
    clim_dir: str
        The directory were the geoTIFFs of the precipitation climatology are
        stored.
    logger: logging instance
        Logger for log messages, passed on from the main.py script.

    Returns
    ------
    climatology_filepath: str (filepath)
        The filepath to the climatogy file, either downloaded or already stored.
    """
    # List to store the data arrays
    data_arrays = []

    # Loop through each month and read the corresponding TIFF file
    for month in range(1, 13):
        logger.info("Month is: %s", {str(month)})
        file_path = os.path.join(clim_dir, f"wc2.1_30s_prec_{month:02d}.tif")
        with rioxarray.open_rasterio(file_path, band_as_variable=True).rename(
            {"band_1": "P"}
        ) as data:
            data_arrays.append(data.P)

    # Combine all the data arrays along a new time dimension
    logger.info("Concatenating the individual months to one file, this can take a while..")
    combined_data = xr.concat(data_arrays, dim="month")
    combined_data = combined_data.assign_coords(month=("month", np.arange(1, 13, 1)))
    ds_combined = combined_data.to_dataset()

    # Store as netCDF
    logger.info("Converting done, store as netCDF")
    compression_settings = {
        "zlib": True,
        "complevel": 4,  # Compression level (1-9), higher means more compression
    }
    ds_combined.to_netcdf(
        os.path.join(clim_dir, "wc2.1_30s_prec_new.nc"),
        encoding={var: compression_settings for var in ds_combined.data_vars},
    )
    ds_combined.close()

    # Return the climatology filepath to which the netCDF was saved
    return os.path.join(clim_dir, "wc2.1_30s_prec_new.nc")
