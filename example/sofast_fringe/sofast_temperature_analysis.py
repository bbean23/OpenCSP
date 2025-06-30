"""Python script that provides an example of how to process SOFAST data
of the Sandia temperature chamber experiment. In this experiment, a CSP
mirror was placed in a large temerature chamber. An installation of SOFAST
was set up outside of the temperature chamber so that when the chamber
doors were opened, SOFAST could immediately take a measurement. This was
done for a range of temperatures from +5C to -50C. This example will walk
a user through how to automatically process such data, how to automatically
plot slope maps, and how to generate slope difference plots to show the
change in slope from a reference temerature.

To run, the base directory containing the calibration and measurement data
must be defined and passed in as the first argument:
```
python sofast_temperature_analysis.py "path/to/base/dir"
```
"""

import sys
import glob
import os
from os.path import join, basename, exists, dirname
import re

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.csp.MirrorAbstract import MirrorAbstract
import opencsp.common.lib.csp.visualize_orthorectified_image as vis
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
import opencsp.common.lib.tool.log_tools as lt


def example_run_temperature_analysis(dir_working: str, dir_save: str) -> None:
    """
    Runs a temperature analysis example using SOFAST. This function processes multiple
    SOFAST measurements of the same mirror at different temperatures by utilizing
    calibration data and measurement files organized in a specified directory structure.

    Parameters
    ----------
    dir_working : str
        The path to the working directory that contains the required subdirectories:

            - `calibration/` Should contain calibration files including:

                - camera.h5
                - facet_definition.json
                - image_projection.h5
                - screen_shape.h5
                - spatial_orientation.h5

            - `measurements/` Should contain measurement files named in the format
              `measurement_XXC.h5`, where XX represents the analysis temperature.
    dir_save : str
        The directory where all output figures will be saved. This directory will be created
        if it does not already exist.

    Notes
    -----
    The function performs the following steps:

        1. Checks for the existence of the input and output directories.
        2. Loads all calibration files necessary for processing the measurements.
        3. Characterizes all measurement files by calibrating fringe images and processing
           them with the SOFAST algorithm.
        4. Calculates the slope deviation of the mirrors from a reference temperature of
           20°C and generates visualizations of the slope magnitude and differences,
           saving them as PNG files in the specified output directory.

    The expected directory structure is as follows:

    ```
    dir_working/
    ├── calibration/
    │   ├── camera.h5
    │   ├── facet_definition.json
    │   ├── image_projection.h5
    │   ├── screen_shape.h5
    │   └── spatial_orientation.h5
    ├── measurements/
    │   ├── measurement_05C.h5
    │   ├── measurement_10C.h5
    │   ├── measurement_20C.h5
    │   ├── measurement_30C.h5
    │   ├── measurement_40C.h5
    │   └── measurement_50C.h5
    └── output/
    ```

    This function is intended for use in scientific research and analysis involving
    temperature-dependent measurements of optical surfaces.
    """

    # 1. Check input/output directory exist
    # =====================================
    # Check input directory exists
    if not exists(dir_working):
        raise FileNotFoundError(f'Could not find: {dir_working}')

    # Define file paths
    dir_calibration = join(dir_working, 'calibration')
    dir_measurement = join(dir_working, 'measurements')

    # Create save directory if needed
    if not exists(dir_save):
        os.mkdir(dir_save)

    # Set up logging
    lt.logger()

    # 2. Load all calibration files
    # =============================
    file_camera = join(dir_calibration, 'camera.h5')
    file_display = join(dir_calibration, 'screen_shape.h5')
    file_orientation = join(dir_calibration, 'spatial_orientation.h5')
    file_facet = join(dir_calibration, 'facet_definition.json')

    camera = Camera.load_from_hdf(file_camera)
    display = Display.load_from_hdf(file_display)
    orientation = SpatialOrientation.load_from_hdf(file_orientation)
    facet_data = DefinitionFacet.load_from_json(file_facet)
    files_measurement = glob.glob(join(dir_measurement, 'measurement*.h5'))

    # 3. Characterize all measurement files
    # =====================================
    # Define common surface definition (parabolic surface) to use with all measurements
    surface = Surface2DParabolic(initial_focal_lengths_xy=(300.0, 300.0), robust_least_squares=True, downsample=10)

    # Create empty list to hold all calculated mirror objects
    mirrors: list[MirrorAbstract] = []
    # Create empty list to hold analysis temperatures
    temperatures: list[str] = []

    # Loop through measurement files and process with SOFAST
    for file_measurement in tqdm(files_measurement, 'Processing SOFAST measurement files', colour='green'):
        # Extract analysis temperature from filename
        filename = basename(file_measurement)
        temperature = float(re.match(r'measurement_(\d+)C.h5', filename).group(1))
        temperatures.append(temperature)
        # Load measurement files
        calibration = ImageCalibrationScaling.load_from_hdf(file_measurement)
        measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)
        # Calibrate fringes
        measurement.calibrate_fringe_images(calibration)
        # Instantiate sofast object
        sofast = Sofast(measurement, orientation, camera, display)
        # Process
        sofast.process_optic_singlefacet(facet_data, surface)
        # Get mirror object
        mirrors.append(sofast.get_optic().mirror)

    # 4. Calculate slope deviation from 20°C reference temperature
    # ==============================================================
    # Define common plotting parameters
    resolution = 0.005  # meters, sampling resolution of mirror surface
    quiver_density = 0.05  # meters, the sampling spacing of quiver arrows
    clim_magnitude = 10  # mrad, for slope magnitude plots
    clim_difference = 5  # mrad, for slope difference plots
    quiver_scale_magnitude = 70  # Quiver arrow scale factor
    quiver_scale_difference = 70  # Quiver arrow scale factor
    fig_size = (10, 10)

    # Get reference temperature mirror object
    idx_ref = temperatures.index(20)  # Reference temperature is 20°C
    mirror_ref = mirrors[idx_ref]
    # Get reference mirror x/y slope array in mrad, shape (2, m, n) array
    slope_ref, x_vec, y_vec = mirror_ref.get_orthorectified_slope_array(resolution)  # radians
    slope_ref = slope_ref * 1000  # mrad
    # Calculate reference slope magnitude in mrad, shape (m, n) array
    slope_magnitude_ref = np.sqrt(np.sum(slope_ref**2, axis=0))

    # Loop through all mirrors and calculate slope deviation from reference temperature
    for mirror, temperature in tqdm(
        zip(mirrors, temperatures, strict=True), 'Plotting SOFAST images', len(temperatures), colour='blue'
    ):
        # Visualize slope magnitude of current mirror
        fig, ax = plt.subplots(figsize=fig_size)
        mirror.plot_orthorectified_slope(
            resolution, "magnitude", clim_magnitude, ax, quiver_density, quiver_scale_magnitude
        )
        ax.set_title(f'Slope Magnitude: {temperature}C')
        ax.grid(True)
        fig.savefig(join(dir_save, f"SlopeMagnitude_{temperature:02.0f}C.png"))
        plt.close(fig)
        # Get current mirror slope map
        slope_current = mirror.get_orthorectified_slope_array(resolution)[0] * 1000  # mrad, shape (2, m, n) array
        slope_magnitude_current = np.sqrt(np.sum(slope_current**2, axis=0))  # mrad, shape (m, n) array
        # Calculate slope difference from reference mirror
        slope_difference = slope_current - slope_ref  # mrad, x/y slopes, shape (2, m, n) array)
        slope_difference_magnitude = slope_magnitude_current - slope_magnitude_ref  # mrad, shape (m, n) array
        # Plot x slope difference image
        fig, ax = plt.subplots(figsize=fig_size)
        vis.plot_orthorectified_image(
            slope_difference_magnitude,
            axis=ax,
            cmap='seismic',
            extent=mirror_ref.region.aabbox(),
            clims=[-clim_difference, clim_difference],
            cmap_title='(mrad)',
        )
        vis.add_quivers(
            slope_difference[0], slope_difference[1], x_vec, y_vec, quiver_density, ax, quiver_scale_difference
        )
        ax.set_title(f'Slope Difference from Nominal: {temperature}C')
        ax.grid(True)
        fig.savefig(join(dir_save, f"SlopeDifference_{temperature:02.0f}C.png"))
        plt.close(fig)


if __name__ == '__main__':
    path = sys.argv[1]
    save_path = join(dirname(__file__), 'data/output/temperature_analysis')
    example_run_temperature_analysis(path, save_path)
