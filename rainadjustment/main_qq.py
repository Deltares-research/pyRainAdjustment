import argparse
import logging
import os
import time
import glob

import numpy as np
import xarray as xr
from xarray import Dataset

from utils.xml_config_parser import parse_run_xml
from utils.io import obtain_gauge_information, check_dimensions


def preprocess_FEWS_netCDF(input_ds):
    '''
    Function to be applied in the xr.open_mfdataset call to preprocess individual files
    Returns
    -------

    '''
    ds = input_ds.assign_coords({'step': ('time', input_ds.time.values - input_ds.time.values[0])})
    ds = ds.swap_dims({'time': 'step'})
    ds = ds.rename({'time': 'valid_time'})
    return ds


def create_qmapping_factors(grid_values, obs_values):
    '''
    Past correctie toe op distributie van tijdstap observaties.
    Parameters
    ----------
    grid any hindcast period with any timestep
    obs

    Returns

    -------

    '''
    grid_q = np.nanquantile(grid_values.flatten(), q=np.linspace(0, 1, 201))
    obs_q = np.nanquantile(obs_values, q=np.linspace(0, 1, 201))
    correction_factors = obs_q / grid_q
    res = np.array([correction_factors, grid_q, obs_q])
    return res


def apply_qmap_correction(forecast: float, historical_data: np.ndarray[:], cor_factor: np.ndarray[:]):
    '''
    Return quantile mapping factors based on historical forecasts
    Parameters
    ----------
    forecast: value for a specific station
    historical_data: array containing historic forecast distribution for the percentiles
    cor_factor: 1D-array containing correction factors for the percentiles


    Returns
    -------

    '''
    correction = np.interp(forecast, historical_data, cor_factor)
    apply_corr = np.where(correction > 1000, 1000, correction)
    return np.array([correction, forecast * apply_corr])


def Q_mapping_gridref(init_month: int, logger, preprocess_files=False):
    if preprocess_files == True:
        files = glob.glob(f'./input/historic_forecasts/*.nc')
        grid_hist = xr.open_mfdataset(files, preprocess=preprocess_FEWS_netCDF)
        grid_hist.to_netcdf('./input/combined_historic_forecasts.nc')
    else:
        grid_hist = xr.load_dataset('./input/combined_historic_forecasts.nc')
    grid_hist = check_dimensions(grid_hist, logger=logger)

    grid_hist['lon'] = np.round(grid_hist.lon.values, decimals=2)
    grid_hist['lat'] = np.round(grid_hist.lat.values, decimals=2)
    select_timestamps = grid_hist.analysis_time.dt.month.values == init_month
    grid_hist = grid_hist.isel({'analysis_time': select_timestamps})
    grid_hist = grid_hist.isel({'analysis_time': grid_hist.analysis_time.dt.day.values < 27})
    print(grid_hist.valid_time.values)

    ref_hist = xr.load_dataset('./input/reference_rainfall.nc')
    ref_hist = check_dimensions(ref_hist, logger=logger)
    ref_hist = ref_hist.sel({'time': np.unique(grid_hist.valid_time.values.flatten())})
    ref_hist['lon'] = np.round(ref_hist.lon.values, decimals=2)
    ref_hist['lat'] = np.round(ref_hist.lat.values, decimals=2)

    q_list = []
    grid_hist = grid_hist.compute()
    for step_it, leadtime in enumerate(grid_hist.step.values):
        print(step_it)
        grid_at_step = grid_hist.isel({'step': step_it})
        ref_hist_step = ref_hist.sel({'time': np.unique(grid_hist.valid_time.values.flatten())})
        factors = xr.apply_ufunc(
            create_qmapping_factors,
            grid_at_step.P,
            ref_hist_step.P,
            input_core_dims=[['analysis_time'], ['time']],
            output_core_dims=[['variable', 'percentile']],
            vectorize=True,
        )
        q_list.append(factors)

    q_array = xr.concat(q_list, dim='step')
    q_array = q_array.assign_coords(percentile=np.linspace(0, 1, 201))
    q_array = q_array.to_dataset(dim='variable')
    q_array = q_array.rename({0: 'correction_factors', 1: 'forecast_P_at_q', 2: 'obs_P_at_q'})

    print(q_array.correction_factors.values)
    q_array.to_netcdf(f'./intermediate/grid_quantile_correction_{init_month}.nc')
    return q_array


def Q_mapping_correction_factors(grid_frcst: Dataset, grid_clim_path):
    climatology = xr.load_dataset(grid_clim_path)
    factors = climatology.correction_factors
    distr = climatology.forecast_P_at_q

    corrections = xr.apply_ufunc(
        apply_qmap_correction,
        grid_frcst.P,
        distr,
        factors,
        input_core_dims=[[], ['percentile'], ['percentile']],
        output_core_dims=[['variable']],
        vectorize=True
    )
    return corrections


def main_qq(derive_correction_factors=False):
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

    # Parse known global args first
    global_args = global_parser.parse_known_args()[0]

    # Set up the logger
    if not os.path.isdir(os.path.join(work_dir, "logs")):
        os.mkdir(os.path.join(work_dir, "logs"))
    logfn = os.path.join(work_dir, "logs", f"log_meteo_rain_gauge_adjustment.txt")
    logging.basicConfig(filename=logfn,
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
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
            config_xml = parse_run_xml('input/adjustment_settings.xml')
        except Exception as exception:
            logger.exception(exception, exc_info=True)

    # The actual work
    if derive_correction_factors:
        for iMonth in range(3, 4):
            if not os.path.exists(f'./intermediate/grid_quantile_correction_{iMonth}.nc'):
                # try:
                Q_mapping_gridref(iMonth, logger=logger, preprocess_files=True)
                # except:
                #     logger.warning('Could not compute correction factors in quantile mapping for month'
                #                    f' index {iMonth}.')

    # Fetch the forecast
    forecast = xr.load_dataset('./input/forecast.nc')
    forecast = preprocess_FEWS_netCDF(forecast)
    month_int = forecast.analysis_time.dt.month.values[0]
    forecast = check_dimensions(forecast, logger=logger)

    forecast = forecast.isel({'analysis_time': forecast.analysis_time.dt.month == int(month_int)})  ## Select March
    forecast['lon'] = np.round(forecast.lon.values, decimals=2)
    forecast['lat'] = np.round(forecast.lat.values, decimals=2)

    print('month: ', month_int)
    corrections = Q_mapping_correction_factors(forecast, f'./intermediate/grid_quantile_correction_{month_int}.nc')
    print(corrections)
    corrections = corrections.to_dataset(dim='variable')
    print(corrections)
    corrections = corrections.rename({0: 'adjustment_factor', 1: 'P'})
    print(corrections)
    corrections = corrections.swap_dims({'step': 'valid_time'})
    corrections = corrections.rename({'valid_time': 'time'})
    # data = data.expand_dims({'analysis_time': [data.analysis_time.values]})
    corrections.to_netcdf(f'./output/corrected_forecast.nc')
    return corrections


if __name__ == '__main__':
    # For 1:12
    main_qq(derive_correction_factors=True)