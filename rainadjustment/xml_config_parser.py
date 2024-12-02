# -*- coding: utf-8 -*-
"""
Last modified 01-10-2024
@authors: Ruben Imhoff (Deltares) and Mees Radema (Deltares)

Function to read xml runfile from Delft-FEWS, which are used as initial
settings for radar rainfall nowcasting with pysteps.

The initial settings are used in the pysteps run script.
"""

import xml
from xml import dom
from xml.dom import minidom


# --------------------------------------------------------------------------- #
# The function
# --------------------------------------------------------------------------- #


def parse_run_xml(xml_file):
    """Function to read the xml settings from Delft-FEWS, which are used as
    initial settings for radar rainfall nowcasting with pysteps.

    Parameters
    ----------
    xml_file: string
        The location and filename of the xml-file which contains the nowcast
        settings for pysteps

    Returns
    -------
    work_dir: string
        The working directory for the adjustment.
    adjustment_method: string
        The used adjustment method. Options are: MFB, Additive, Multiplicative,
        Mixed. Defaults to None.
    nearest_cells_to_use: int
        The number of grid cells around (and including) the grid cell that
        corresponds to the rain gauge location. Typically 9 cells are used to
        account for rain drift due to wind effects. Defaults to 9.
    statistical_function: str
        Statistical function to find the value of the nearest cells that is
        compared with the gauge value. Options are median, mean and best.
        Defaults to median.
    min_gauges: int
        Minimum number of gauges that should be present in order to apply
        the rain gauge adjustment method. Often, 5 gauges are used. Defaults
        to 5.
    kriging_n_gauges_to_use: int
        The maximum number of neighbouring gauges the kriging algorithm uses
        per gauge. If the total number of gauges is smaller than this value,
        the total number of gauges is used in pyRainAdjustment. Defaults to
        12.
    threshold: float
        The threshold value used. If a gauge or grid cell is below this value,
        it is not used in the adjustment procedure. For the additive
        adjustment_method, this should always be zero. Defaults to 0.0.
    max_change_factor: float
        Maximum change (both increase and decrease) of the adjusted gridded
        rainfall per grid cell point. The values should be provided as float
        of the factor (e.g. 2.0 - max. two times smaller or bigger than
        orginal). Defaults to None.
    interpolation_method: str
        The interpolation method that should be used. An interpolation method
        from https://docs.wradlib.org/en/latest/ipol.html should be provided.
        Defaults to Idw (inverse distance weighting).
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
    adjustment_method = None
    statistical_function = "median"
    interpolation_method = "Idw"

    # Get int properties
    properties = doc.getElementsByTagName("float")
    for prop in properties:
        if prop.attributes["key"].value == "threshold":
            threshold = float(prop.getAttribute("value"))
        if prop.attributes["key"].value == "max_change_factor":
            max_change_factor = float(prop.getAttribute("value"))

    # Get int properties
    properties = doc.getElementsByTagName("int")
    for prop in properties:
        if prop.attributes["key"].value == "nearest_cells_to_use":
            nearest_cells_to_use = int(prop.getAttribute("value"))
        if prop.attributes["key"].value == "min_gauges":
            min_gauges = int(prop.getAttribute("value"))
        if prop.attributes["key"].value == "kriging_n_gauges":
            kriging_n_gauges_to_use = int(prop.getAttribute("value"))

    # Get str properties
    properties = doc.getElementsByTagName("string")
    for prop in properties:
        if prop.attributes["key"].value == "adjustment_method":
            adjustment_method = prop.getAttribute("value")
        if prop.attributes["key"].value == "statistical_function":
            statistical_function = prop.getAttribute("value")
        if prop.attributes["key"].value == "interpolation_method":
            interpolation_method = prop.getAttribute("value")

    output_dict = {
        "work_dir": work_dir,
        "threshold": threshold,
        "max_change_factor": max_change_factor,
        "nearest_cells_to_use": nearest_cells_to_use,
        "min_gauges": min_gauges,
        "kriging_n_gauges_to_use": kriging_n_gauges_to_use,
        "adjustment_method": adjustment_method,
        "statistical_function": statistical_function,
        "interpolation_method": interpolation_method,
    }

    return output_dict
