import os
from os.path import join

import matplotlib

import opencsp
from opencsp.app.sofast.lib.visualize_setup import visualize_setup
from opencsp.common.lib.deflectometry.Display import Display
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.Measurement import Measurement
from opencsp.common.lib.deflectometry.EnsembleData import EnsembleData
from opencsp.common.lib.deflectometry.FacetData import FacetData
from opencsp.app.sofast.lib.Sofast import Sofast
from opencsp.common.lib.deflectometry.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlMirror as rcm


def example_driver():
    """Example SOFAST script

    Performs processing of previously collected Sofast data of multi facet mirror ensemble.
        1. Loads saved "multi-facet" SOFAST collection data
        2. Processes data with SOFAST (without using facet file)
        3. Prints best-fit parabolic focal lengths
        4. Plots slope magnitude, physical setup

    """
    # Define sample data directory
    sample_data_dir = join(
        os.path.dirname(opencsp.__file__), 'test/data/sofast_measurements/'
    )

    # Directory setup
    file_measurement = join(sample_data_dir, 'measurement_ensemble.h5')
    file_camera = join(sample_data_dir, 'camera.h5')
    file_display = join(sample_data_dir, 'display_distorted_2d.h5')
    file_calibration = join(sample_data_dir, 'calibration.h5')
    file_facet = join(sample_data_dir, 'Facet_lab_6x4.json')
    file_ensemble = join(sample_data_dir, 'Ensemble_lab_6x4.json')

    # Define save dir
    dir_save = join(os.path.dirname(__file__), 'data/output/facet_ensemble')
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    # Load data
    camera = Camera.load_from_hdf(file_camera)
    display = Display.load_from_hdf(file_display)
    measurement = Measurement.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)
    ensemble_data = EnsembleData.load_from_json(file_ensemble)

    # Define facet data
    facet_data = [FacetData.load_from_json(file_facet)] * ensemble_data.num_facets

    # Define surface data (plano)
    # surface_data = [dict(surface_type='plano', robust_least_squares=False, downsample=20)] * ensemble_data.num_facets
    # Define surface data (parabolic)
    surface_data = [
        dict(
            surface_type='parabolic',
            initial_focal_lengths_xy=(100.0, 100.0),
            robust_least_squares=False,
            downsample=20,
        )
    ] * ensemble_data.num_facets

    # Calibrate fringes
    measurement.calibrate_fringe_images(calibration)

    # Instantiate sofast object
    sofast = Sofast(measurement, camera, display)

    # Update search parameters
    sofast.params.mask_hist_thresh = 0.83
    sofast.params.perimeter_refine_perpendicular_search_dist = 10.0
    sofast.params.facet_corns_refine_frac_keep = 1.0
    sofast.params.facet_corns_refine_perpendicular_search_dist = 3.0
    sofast.params.facet_corns_refine_step_length = 5.0

    # Process
    sofast.process_optic_multifacet(facet_data, ensemble_data, surface_data)

    # Calculate focal length from parabolic fit
    for idx in range(sofast.num_facets):
        if surface_data[idx]['surface_type'] == 'parabolic':
            surf_coefs = sofast.data_characterization_facet[idx]['surf_coefs_facet']
            focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
            print('Parabolic fit focal lengths:')
            print(f'  X {focal_lengths_xy[0]:.3f} m')
            print(f'  Y {focal_lengths_xy[1]:.3f} m')

    # Get optic representation
    ensemble: FacetEnsemble = sofast.get_optic()

    # Generate plots
    figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
    mirror_control = rcm.RenderControlMirror(
        centroid=True, surface_normals=True, norm_res=1
    )
    axis_control_m = rca.meters()

    # Visualize setup
    fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, title='')
    spatial_ori: SpatialOrientation = sofast.data_geometry_facet[0][
        'spatial_orientation'
    ]
    visualize_setup(
        display,
        camera,
        spatial_ori.v_screen_optic_screen,
        spatial_ori.r_optic_screen,
        ax=fig_record.axis,
    )
    fig_record.save(dir_save, 'physical_setup_layout', 'png')

    # Plot scenario
    fig_record = fm.setup_figure_for_3d_data(
        figure_control, axis_control_m, title='Facet Ensemble'
    )
    ensemble.draw(fig_record.view, mirror_control)
    fig_record.axis.axis('equal')
    fig_record.save(dir_save, 'facet_ensemble', 'png')

    # Plot slope map
    fig_record = fm.setup_figure(figure_control, axis_control_m, title='')
    ensemble.plot_orthorectified_slope(res=0.002, clim=7, axis=fig_record.axis)
    fig_record.save(dir_save, 'slope_magnitude', 'png')

    # Save data (optionally)
    sofast.save_data_to_hdf(f'{dir_save}/data_multifacet.h5')


if __name__ == '__main__':
    example_driver()
