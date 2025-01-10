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
        Mixed.
    nearest_cells_to_use: int
        The number of grid cells around (and including) the grid cell that 
        corresponds to the rain gauge location. Typically 9 cells are used to
        account for rain drift due to wind effects.
    statistical_function: str
        Statistical function to find the value of the nearest cells that is
        compared with the gauge value. Options are median, mean and best.
    min_gauges: int
        Minimum number of gauges that should be present in order to apply
        the rain gauge adjustment method. Often, 5 gauges are used.
    threshold: float
        The threshold value used. If a gauge or grid cell is below this value,
        it is not used in the adjustment procedure. For the additive
        adjustment_method, this should always be zero. 
    max_change_factor: float
        Maximum change (both increase and decrease) of the adjusted gridded
        rainfall per grid cell point. The values should be provided as float
        of the factor (e.g. 2.0 - max. two times smaller or bigger than
        orginal).
    """
    # Set the indir + filename of the used xml file
    input_xml = xml_file
    
    # Open the xml
    doc = xml.dom.minidom.parse(input_xml)
    
    # Get XML elements
    work_dir = doc.getElementsByTagName("workDir")[0].firstChild.nodeValue

    output_dict = {}
    properties = doc.getElementsByTagName("float")
    for prop in properties:
        output_dict.update({prop.attributes["key"].value: float(prop.getAttribute("value"))})

    properties = doc.getElementsByTagName("int")
    for prop in properties:
        output_dict.update({prop.attributes["key"].value: int(prop.getAttribute("value"))})

    properties = doc.getElementsByTagName("string")
    for prop in properties:
        output_dict.update({prop.attributes["key"].value: prop.getAttribute("value")})

    return output_dict
