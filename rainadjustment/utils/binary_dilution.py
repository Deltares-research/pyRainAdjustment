# -*- coding: utf-8 -*-
"""
Functionality to let a central grid slowly smoothly transition into a larger outside grid.

Available functons:
- apply_smooth_dilated_mask
- __compute_smooth_dilated_mask
"""
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt


def apply_smooth_dilated_mask(
    config_xml: dict[str, Any],
    input_array: npt.NDArray[np.float64],
    background_array: npt.NDArray[np.float64],
):
    """
    Parameters
    ----------
    config_xml: dict
        Dictionary containing the adjustment settings.
    input_array: ndarray(float)
        The input array that should be smoothed into the background array
        at the edges.
    background_array: ndarray(float)
        The background array containing the edge values.


    Returns
    ------
    output_array: ndarray(float)
        The adjusted input array after the dilated mask smoothing has been
        applied.
    """
    smoothing_distance = config_xml["smooth_edge_values_range"]

    # Set edge cells to np.nan
    input_array[:smoothing_distance, :] = np.nan  # Top edge
    input_array[-smoothing_distance:, :] = np.nan  # Bottom edge
    input_array[:, :smoothing_distance] = np.nan  # Left edge
    input_array[:, -smoothing_distance:] = np.nan  # Right edge

    # First get the nan indices based on the smoothing_distance
    nan_indices = np.isnan(input_array)

    # Compute the smooth dilated mask
    new_mask = __compute_smooth_dilated_mask(
        nan_indices,
        max_padding_size_in_px=smoothing_distance,
    )

    # Ensure mask values are between 0 and 1
    mask_background = np.clip(new_mask, 0, 1)
    mask_input_array = np.clip(1 - new_mask, 0, 1)

    # Handle NaNs in precip_forecast_new and precip_forecast_new_mod_only by setting NaNs to 0 in the blending step
    input_array = np.nan_to_num(input_array, nan=0.0)
    background_array = np.nan_to_num(background_array, nan=0.0)

    # Perform the blending of radar and model inside the radar domain using a weighted combination
    output_array = np.nansum(
        [
            mask_background * background_array,
            mask_input_array * input_array,
        ],
        axis=0,
    )

    return output_array


def __compute_smooth_dilated_mask(
    original_mask,
    max_padding_size_in_px=0,
    gaussian_kernel_size=9,
    inverted=False,
    non_linear_growth_kernel_sizes=False,
):
    """
    Compute a smooth dilated mask using Gaussian blur and dilation with varying kernel sizes.

    Parameters
    ----------
    original_mask : array_like
        Two-dimensional boolean array containing the input mask.
    max_padding_size_in_px : int
        The maximum size of the padding in pixels. Default is 100.
    gaussian_kernel_size : int, optional
        Size of the Gaussian kernel to use for blurring, this should be an uneven number. This option ensures
        that the nan-fields are large enough to start the smoothing. Without it, the method will also be applied
        to local nan-values in the radar domain. Default is 9, which is generally a recommended number to work
        with.
    inverted : bool, optional
        Typically, the smoothed mask works from the outside of the radar domain inward, using the
        max_padding_size_in_px. If set to True, it works from the edge of the radar domain outward
        (generally not recommended). Default is False.
    non_linear_growth_kernel_sizes : bool, optional
        If True, use non-linear growth for kernel sizes. Default is False.

    Returns
    -------
    final_mask : array_like
        The smooth dilated mask normalized to the range [0,1].
    """
    if max_padding_size_in_px < 0:
        raise ValueError("max_padding_size_in_px must be greater than or equal to 0.")

    # Check if gaussian_kernel_size is an uneven number
    assert gaussian_kernel_size % 2

    # Convert the original mask to uint8 numpy array and invert if needed
    array_2d = np.array(original_mask, dtype=np.uint8)
    if inverted:
        array_2d = np.bitwise_not(array_2d)

    # Rescale the 2D array values to 0-255 (black or white)
    rescaled_array = array_2d * 255

    # Apply Gaussian blur to the rescaled array
    blurred_image = cv2.GaussianBlur(
        rescaled_array, (gaussian_kernel_size, gaussian_kernel_size), 0
    )

    # Apply binary threshold to negate the blurring effect
    _, binary_image = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY)

    # Define kernel sizes
    if non_linear_growth_kernel_sizes:
        lin_space = np.linspace(0, np.sqrt(max_padding_size_in_px), 10)
        non_lin_space = np.power(lin_space, 2)
        kernel_sizes = list(set(non_lin_space.astype(np.uint8)))
    else:
        kernel_sizes = np.linspace(0, max_padding_size_in_px, 10, dtype=np.uint8)

    # Process each kernel size
    final_mask = np.zeros_like(binary_image, dtype=np.float64)
    for kernel_size in kernel_sizes:
        if kernel_size == 0:
            dilated_image = binary_image
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated_image = cv2.dilate(binary_image, kernel)

        # Convert the dilated image to a binary array
        _, binary_array = cv2.threshold(dilated_image, 128, 1, cv2.THRESH_BINARY)
        final_mask += binary_array

    final_mask = final_mask / final_mask.max()

    return final_mask
