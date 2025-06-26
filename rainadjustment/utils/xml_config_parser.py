# -*- coding: utf-8 -*-
"""
Function to read xml runfile from Delft-FEWS, which are used as initial
settings for pyRainAdjustment.

Available functions:
- parse_run_xml
"""

from typing import Any
import xml
from xml import dom
from xml.dom import minidom


def parse_run_xml(xml_file: str) -> dict[str, Any]:
    """Function to read the xml settings from Delft-FEWS, which are used as
    initial settings for radar rainfall nowcasting with pysteps.

    Parameters
    ----------
    xml_file: string
        The location and filename of the xml-file which contains the nowcast
        settings for pysteps

    Returns at dictionary containing (if present in the XML file)
    ------------------------------------
    work_dir: string
        The working directory for the adjustment.
    threshold: float
        The threshold value used. If a gauge or grid cell is below this value,
        it is not used in the adjustment procedure. For the additive
        adjustment_method, this should always be zero. Defaults to 0.0.
    max_change_factor: float
        Maximum change (both increase and decrease) of the adjusted gridded
        rainfall per grid cell point. The values should be provided as float
        of the factor (e.g. 2.0 - max. two times smaller or bigger than
        orginal). Defaults to None.
    nearest_cells_to_use: int
        The number of grid cells around (and including) the grid cell that
        corresponds to the rain gauge location. Typically 9 cells are used to
        account for rain drift due to wind effects. Defaults to 9.
    min_gauges: int
        Minimum number of gauges that should be present in order to apply
        the rain gauge adjustment method. Often, 5 gauges are used. Defaults
        to 5.
    kriging_n_gauges_to_use: int
        The maximum number of neighbouring gauges the kriging algorithm uses
        per gauge. If the total number of gauges is smaller than this value,
        the total number of gauges is used in pyRainAdjustment. Defaults to
        12.
    downscaling_factor: int
        The factor by which the gridded precipitation should be downscaled.
        For instance, a factor of 2 indicates that the downscaled precipitation
        will have a grid resolution that is two times finer than the original
        grid resolution.
    adjustment_method: string
        The used adjustment method. Options are: MFB, Additive, Multiplicative,
        Mixed. Defaults to None.
    statistical_function: str
        Statistical function to find the value of the nearest cells that is
        compared with the gauge value. Options are median, mean and best.
        Defaults to median.
    interpolation_method: str
        The interpolation method that should be used. An interpolation method
        from https://docs.wradlib.org/en/latest/ipol.html should be provided.
        Defaults to Idw (inverse distance weighting).
    clim_filepath: str
        The filepath to the monthly climatology netCDF file.
    qq_filepath: str
        The filepath to the quantile mapping fators.
    variogram_model: str
        The variogram model used for the Kriging interpolation. Defaults to
        "standard".
    derive_qmapping_factors: bool
        Setting to derive the quantile mapping factors or not. Defaults to False.
    qmapping_month: int | None
        The month for which the quantile mapping factors should be derived.
        Defaults to None.
    leadtime_specific_factors: bool
        Setting to derive lead-time specific qq correction factors (when True) or
        one factor for all leadtimes (when False). Defaults to False.
    gridded_reference_product: bool
        Setting to indicate whether the reference rainfall for the quantile mapping
        procedure is a gridded product (True) or contains point observations from
        rain gauges (False).
    """
    # Set the indir + filename of the used xml file
    input_xml = xml_file

    # Open the xml
    doc = xml.dom.minidom.parse(input_xml)

    # Get XML elements
    work_dir = doc.getElementsByTagName("workDir")[0].firstChild.nodeValue

    # Initialize the variables
    threshold = 0.0
    max_change_factor = None
    nearest_cells_to_use = 9
    min_gauges = 5
    kriging_n_gauges_to_use = 12
    downscaling_factor = None
    adjustment_method = None
    statistical_function = "median"
    interpolation_method = "Idw"
    clim_filepath = None
    qq_filepath = None
    variogram_model = "standard"
    derive_qmapping_factors = False
    qmapping_month = None
    leadtime_specific_factors = False
    gridded_reference_product = True

    output_dict = {
        "work_dir": work_dir,
        "threshold": threshold,
        "max_change_factor": max_change_factor,
        "nearest_cells_to_use": nearest_cells_to_use,
        "min_gauges": min_gauges,
        "kriging_n_gauges_to_use": kriging_n_gauges_to_use,
        "downscaling_factor": downscaling_factor,
        "adjustment_method": adjustment_method,
        "statistical_function": statistical_function,
        "interpolation_method": interpolation_method,
        "clim_filepath": clim_filepath,
        "qq_filepath": qq_filepath,
        "variogram_model": variogram_model,
        "derive_qmapping_factors": derive_qmapping_factors,
        "qmapping_month": qmapping_month,
        "leadtime_specific_factors": leadtime_specific_factors,
        "gridded_reference_product": gridded_reference_product,
    }

    properties = doc.getElementsByTagName("float")
    for prop in properties:
        output_dict.update({prop.attributes["key"].value: float(prop.getAttribute("value"))})

    properties = doc.getElementsByTagName("int")
    for prop in properties:
        output_dict.update({prop.attributes["key"].value: int(prop.getAttribute("value"))})

    properties = doc.getElementsByTagName("string")
    for prop in properties:
        output_dict.update({prop.attributes["key"].value: prop.getAttribute("value")})

    properties = doc.getElementsByTagName("bool")
    for prop in properties:
        key = prop.attributes["key"].value
        value_str = prop.getAttribute("value").strip().lower()
        value = value_str == "true"
        output_dict[key] = value

    return output_dict
