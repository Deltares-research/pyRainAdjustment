# Downscaling methods in pyRainAdjustment
Most precipitation downscaling methods are either climatology based or statistics based, with the field of artificial intelligence slowly providing an alternative to these conventional approaches. In pyRainAdjustment, a relatively simple but robust climatology-based [downscaling method](https://github.com/Deltares-research/pyRainAdjustment/tree/main/rainadjustment/functions/downscaling.py) is implemented. 

The method uses monthly climatology data to downscale the original precipitation grid to a predefined higher-resolution grid. This takes place in a couple of steps:
1. The high-resolution climatology data is reprojected with an averaging resampling scheme to the coarser spatial grid the original precipitation data is on. This way, the high-resolution data gets averaged out over the coarser grid cells.
2. The climatology data on the coarser grid is then downscaled with a nearest neighbour reprojection to the finer desired output grid, which preserves the data on the coarse grid, but presents it on the finer output grid. This product is regarded as the ‘coarse grid’. 
3. The high-resolution climatology data is subsequently also directly reprojected to the output grid with an averaging resampling scheme in case the output grid is coarser than the original grid resolution of the climatology data. This is regarded as the ‘fine grid’ and preserves the high-resolution climatology data on the output data as well as possible.
4. On the output grid, a downscaling factor is calculated per grid cell, which is calculated as:
`f_{downscaling} =  P_{fine}/P_{coarse}`,
 with `P_{fine}` the climatology data on the ‘fine grid’ and `P_{coarse}` the climatology data on the ‘coarse grid’. 
5. Finally, the original gridded precipitation data is downscaled to the finer resolution by first reprojecting it with a nearest neighbour transformation to the finer grid and subsequently multiplying it with `f_{downscaling}` to correct it with the climatology-based downscaling factor.

As a standard climatology dataset, we recommend using the WorldClim2 dataset by Fick & Hijmans (2017), which is a monthly dataset containing global precipitation climatology data on a 1-km resolution. The figure below illustrates the use of the introduced downscaling method with the WorldClim2 dataset by downscaling daily ERA5 reanalysis data from a 30-km resolution to a 3-km resolution. The result clearly provides more resolution and starts to show the effect of the topography on the precipitation distribution, which is also present in the WorldClim2 dataset. However, the individual original ERA5 grid cells remain somewhat visible, which is something that a relatively simple downscaling method cannot overcome.

When no climatology product is available, pyRainAdjustment will automatically download and prepare the WorldClim2 monthly precipitation climatology on 1-km resolution (Fick and Hijmans, 2017). This climatology is placed in the 'clim' sub folder of the pyRainAdjustment module in Delft-FEWS, so in principle this step only has to take place once (and takes approximately 10 minutes).  

![image](https://github.com/user-attachments/assets/70dbf453-a88d-4c65-87ec-368cd7adc3fe)
Example of climatology-based downscaling of daily ERA5 reanalysis data over Southeastern Australia on 2024-11-30. Shown are (a) the original spatial resolution (approximately 30 km) to (b) a resolution of 3 km. 

## References
Fick, Stephen E., and Robert J. Hijmans. ‘WorldClim 2: New 1-Km Spatial Resolution Climate Surfaces for Global Land Areas’. International Journal of Climatology 37, no. 12 (2017): 4302–15. https://doi.org/10.1002/joc.5086.
