# Adjustment Methods in Hindcasting Mode
Reaching better resolution through downscaling is not the only step to improve gridded precipitation products for hydrological purposes. As most gridded precipitation products are heavily biased, generally through a combination of systematic and random errors, bias adjustment procedures are a crucial additional step to improve precipitation products for hydrological purposes. When rain gauge observations are available, for instance in hindcasting or state-updating mode in an operational system, bias adjustment is a little more straightforward as gridded precipitation products can be adjusted using the gauge observations. As part of this project, we have implemented three different bias adjustment methods. The methods vary from simple, uniform corrections (mean field bias, see section 2.2.1) to more advanced geospatial adjustment methods that a much higher rain gauge density in real time (see section 2.2.3).

## Mean Field Bias (MFB) Adjustment
The Mean Field Bias (MFB) adjustment method provides a spatially uniform multiplicative adjustment factor that can be applied to adjust the gridded rainfall field (see the figure below for an example fr Australia). It uses all available rain gauges and corresponding grid cells to calculate an adjustment factor for the entire grid as follows:

$$
F_{MFB} = \frac{\sum_{n=1}^{N} G(i_n, j_n)}{\sum_{n=1}^{N} P_{orig}(i_n, j_n)}
$$

Here:
- $$G(i_n, j_n)$$ is the rain gauge sum for gauge *n* at location $$(i_n, j_n)$$ for the used time interval.
- $$P_{orig}(i_n, j_n)$$ is the precipitation value at the corresponding grid cell of the original, unadjusted gridded precipitation product.

Once calculated, $$F_{MFB}$$ is multiplied with $$P_{orig}$$ to get the adjusted output precipitation grid.

To ensure $$F_{MFB}$$ does not blow up when $$\sum_{n=1}^{N} P_{orig}(i_n, j_n)$$ is either very small or zero, we typically work with a threshold for the entire sum of both the gauges and corresponding grid cells, or a threshold per gauge–grid cell couple.

The advantage of the MFB method is that the uniform adjustment makes the method independent of the number of available rain gauges, offering a robust method in operations. However, the spatially uniform application makes it insensitive to spatial variations in the gridded precipitation product quality — for instance, due to the large extent of the domain, specific local weather phenomena, or local topography.

![image](https://github.com/user-attachments/assets/a1e64f5e-a1c2-4ce3-bcd0-3a6bde122f89)
Example of mean field bias adjustment of the ERA5 reanlaysis data over Southeastern Australia for 2024-11-28. Shown are a) the original QPE from ERA5, b) the adjustment factor and c) the adjusted ERA5 QPE.

## Spatial Adjustment Methods
To not only adjust in time but also in space, we implemented **spatial adjustment methods** in `pyRainAdjustment`. With these methods, an adjustment factor per rain–grid cell pair is calculated, and subsequently, all point-based adjustment factors are interpolated back onto the original grid of the gridded precipitation product. The interpolated gridded adjustment factor is then used to adjust the gridded precipitation field in space for that time step (see the figure below for an example over Southeastern Australia).

In `pyRainAdjustment`, we implemented four different interpolation schemes that can be used for this step:

- Nearest neighbour
- Linear
- Inverse-distance weighting
- Ordinary kriging

The adjustment method can be calculated through a **multiplicative**, **additive**, or **mixed** approach.

### Multiplicative Approach

This resembles the MFB adjustment but works per gauge–grid cell pair instead of over the sum of all of them simultaneously:

$$
F(i,j) = \frac{G(i,j)}{P_{orig}(i,j)}
$$

Subsequently, all local values of $$F_{MFB}(i,j)$$ are interpolated back onto the original grid, and the final adjusted precipitation grid is calculated as:

$$
P_{adj}(m,n) = F_{MFB}(m,n) \cdot P_{orig}(m,n)
$$

where $$(m,n)$$ is the grid cell at row *m* and column *n*.

### Additive Approach

This calculates an **error** instead of an adjustment factor between the gauge–grid cell pair:

$$
E(i,j) = G(i,j) - P_{orig}(i,j)
$$

After interpolating all $$E(i,j)$$ onto the original grid, the final adjusted precipitation grid is calculated as:

$$
P_{adj}(m,n) = 
\begin{cases}
0.0, & \text{if } P_{orig}(m,n) + E(m,n) < 0 \\
P_{orig}(m,n) + E(m,n), & \text{otherwise}
\end{cases}
$$

### Mixed Error Model

The **mixed error model** assumes that both a multiplicative and additive error can be present and tries to correct for that as follows (Bronstert et al., 2010):

$$
G(i,j) = P_{orig}(i,j) \cdot (1 + \delta) + \varepsilon
$$

Where:
- $$\delta$$ is the multiplicative error
- $$\varepsilon$$ is the additive error

To determine both $$\delta$$ and $$\varepsilon$$, a **least-squares estimation** is made, assuming both are independent and normally distributed. They are calculated as:

$$
\varepsilon = \frac{G(i,j) - P_{orig}(i,j)}{P_{orig}(i,j)^2 + 1}
$$

$$
\delta = \frac{G(i,j) - \varepsilon}{P_{orig}(i,j)} - 1
$$

Both $$\delta$$ and $$\varepsilon$$ are then interpolated onto the original grid, and the final adjusted precipitation grid is calculated as:

$$
P_{adj}(m,n) = P_{orig}(m,n) \cdot (1 + \delta(m,n)) + \varepsilon(m,n)
$$

![image](https://github.com/user-attachments/assets/1bf51276-24e3-4ae6-b1fb-5988c590df53)
Example of a spatial mixed error model to adjust the ERA5 reanlaysis data over Southeastern Australia for 2024-11-28. Shown are a) the original QPE from ERA5, b) the adjustment factor, interpolated onto the original grid with an inverse-distance weighting approach. and c) the adjusted ERA5 QPE.

## Kriging with External Drift
The most advanced geospatial adjustment technique implemented in `pyRainAdjustment` is **kriging with external drift (KED)**. Ordinary kriging is an interpolation technique in which the prediction at grid cell $$(m,n)$$ is based on the weighted sum of all surrounding rain gauge values:

$$
P(m,n) = \sum_{n=1}^{N} \lambda_n \cdot G(i_n, j_n)
$$

Where:
- $$\lambda_n$$ is the weight for the *n*th gauge location.
- $$G(i_n, j_n)$$ is the gauge value at location $$(i_n, j_n)$$.

In **inverse-distance weighting (IDW)**, the weight depends on the distance of the gauge to the grid cell. In **ordinary kriging**, however, the weight is determined using a **covariance model** in combination with the location of the target grid cell.

To estimate the covariance model, a **semi-variogram model** is used to determine the relationship between the semi-variance of the data and the distance between data points. In `pyRainAdjustment`, the method can:
- Automatically derive this semi-variogram model (assuming a spherical model), or
- Use a standard exponential model (`1.0 * e^10,000 + 0.0`), similar to the approach in the widely-used `wradlib` Python package (Heistermann et al., 2013).
For more information on kriging, refer to Wackernagel (2003).

In **Kriging with External Drift (KED)**, a spatially correlated residual (the external drift term) is added to the ordinary kriging equation. This ensures that the estimated precipitation grid follows the structure of the external drift when interpolating the gauge values to the grid. The advantage of this approach over the aforementioned adjustment methods is that KED can be seen as a more hybrid approach in which the gauge values are the starting point and the gridded rainfall is assumed to capture the spatial patterns better, which is then used in the interpolation (kriging) step. This results in an end product that can contain values different from those found in the gauge and gridded precipitation data, with patterns that still mimic the gridded precipitation data but adjusted for the observed values measured by the gauges (see the figure below). This method is generally considered one of the best adjustment methods for gridded rainfall product (Goudenhoofdt and Delobbe, 2009), but has as disadvantage that its calculation time is longer and that it needs a high density of rain gauges that are available in real time, which generally limits the operational applicability of this method.

![image](https://github.com/user-attachments/assets/aa667def-87de-4bf4-9dc3-9ed01b23881c)
Example of kriging with external drift adjustment of the ERA5 reanlaysis data over Southeastern Australia for 2024-11-28. Shown are a) the original QPE from ERA5, b) the adjustment factor and c) the adjusted ERA5 QPE.

## Usage in Delft-FEWS
Examples of how to configure these options in Delft-FEWS are provided in the folder `config` and this is further explained in [/config/README.md](https://github.com/Deltares-research/pyRainAdjustment/tree/main/config/README.md). For a configuration example of the adjustment methods, have a look at [/config/ModuleConfigFiles/ProcessRainGaugeAdjustment.xml](https://github.com/Deltares-research/pyRainAdjustment/tree/main/config/ModuleConfigFiles/ProcessRainGaugeAdjustment.xml). The different adjustment methods can be provided to the Delft-FEWS model adapter (General Adapter) by providing the property key `adjustment_method`, which can be one of the following methods: `MFB`, `additive`, `multiplicative`, `mixed`, `KED` and `quantile_mapping`.

## References
Bronstert, A., E. Zehe, and S. Itzerott. ‘Operationelle Abfluss- Und Hochwasservorhersage in Quellgebieten - OPAQUE : Abschlussbericht ; Laufzeit Des Vorhabens: 01.06.2006 Bis 31.03.2010’, 2010. https://doi.org/10.2314/GBV:65175206X.

Goudenhoofdt, E., and L. Delobbe. ‘Evaluation of Radar-Gauge Merging Methods for Quantitative Precipitation Estimates’. Hydrology and Earth System Sciences 13, no. 2 (18 February 2009): 195–203. https://doi.org/10.5194/hess-13-195-2009.

Heistermann, M., S. Jacobi, and T. Pfaff. ‘Technical Note: An Open Source Library for Processing Weather Radar Data (Wradlib)’. Hydrology and Earth System Sciences 17, no. 2 (28 February 2013): 863–71. https://doi.org/10.5194/hess-17-863-2013.

Wackernagel, Hans. ‘Ordinary Kriging’. In Multivariate Geostatistics: An Introduction with Applications, edited by Hans Wackernagel, 79–88. Berlin, Heidelberg: Springer, 2003. https://doi.org/10.1007/978-3-662-05294-5_11. 

