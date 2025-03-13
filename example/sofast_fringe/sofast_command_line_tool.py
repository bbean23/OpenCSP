from os.path import join
import sys

from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.SofastInterface import (
    SofastInterface,
    SofastCommonRunData,
    SofastFixedRunData,
    SofastCommonProcessData,
    SofastFringeProcessData,
    SofastFixedProcessData,
)
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import ImageAcquisition
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.RegionXY import RegionXY
import opencsp.common.lib.tool.log_tools as lt


def example_sofast_command_line_tool(dir_cal: str, dir_save: str):
    # Set up OpenCSP logger
    lt.logger(join(dir_save, "log.txt"), lt.log.WARN)

    # Define image acquisition
    image_acquisition_in = ImageAcquisition(instance=0)  # First camera instance found
    image_acquisition_in.frame_size = (1626, 1236)  # Set frame size
    image_acquisition_in.gain = 230  # Set gain (higher=faster/more noise, lower=slower/less noise)

    # Define image projection
    file_image_projection = join(dir_cal, "image_projection_optics_lab_landscape_square.h5")
    image_projection = ImageProjection.load_from_hdf(file_image_projection)
    image_projection.display_data.image_delay_ms = 200  # define projector-camera delay

    # Setup and run Sofast Interface
    inter = SofastInterface(image_acquisition_in)

    # Initialize Sofast Fringe system
    fringes = Fringes.from_num_periods(4, 4)
    inter.data_sofast_common_run = SofastCommonRunData(
        measure_point_optic=Vxyz((0, 0, 0)), dist_optic_screen=10, name_optic="Some Optic"
    )
    inter.initialize_sofast_fringe(fringes)

    # Initialize Sofast Fixed system
    inter.data_sofast_fixed_run = SofastFixedRunData(origin=Vxy((100, 200)))
    inter.initialize_sofast_fixed()

    # Initialize Sofast processing data
    file_facet = join(dir_cal, "facet_NSTTF.json")
    file_camera = join(dir_cal, "camera_sofast_optics_lab_landscape_2025_02.h5")
    file_ori = join(dir_cal, "spatial_orientation_optics_lab_landscape.h5")
    file_display_shape = join(dir_cal, "display_shape_optics_lab_landscape_square_distorted_3d_100x100.h5")
    file_dot_locs = join(dir_cal, "dot_locations_optics_lab_landscape_square_width3_space6.h5")
    surface_fringe = Surface2DParabolic((100, 100), False, 10)
    surface_fixed = Surface2DParabolic((100, 100), False, 1)

    inter.data_sofast_common_proccess = SofastCommonProcessData(
        facet_definition=DefinitionFacet.load_from_json(file_facet),
        camera=Camera.load_from_hdf(file_camera),
        spatial_orientation=SpatialOrientation.load_from_hdf(file_ori),
    )
    inter.data_sofast_fringe_process = SofastFringeProcessData(
        display_shape=DisplayShape.load_from_hdf(file_display_shape), surface_2d=surface_fringe
    )
    inter.data_sofast_fixed_process = SofastFixedProcessData(
        fixed_pattern_dot_locs=DotLocationsFixedPattern.load_from_hdf(file_dot_locs), surface_2d=surface_fixed
    )

    # Set up plotting
    shape = RegionXY.rectangle(1.5)
    focal_length = 100
    inter.plotting.optic_reference = MirrorParametric.generate_symmetric_paraboloid(focal_length, shape)
    inter.plotting.options_ray_trace_vis.to_plot = False
    inter.plotting.options_file_output.number_in_name = False
    inter.plotting.options_file_output.to_save = True

    # Set up save paths
    inter.paths.dir_save_fixed = dir_save
    inter.paths.dir_save_fringe = dir_save
    inter.paths.dir_save_fringe_calibration = dir_save

    # Run
    inter.run_cli()


if __name__ == "__main__":
    dir_cal_in = sys.argv[1]
    dir_save_in = sys.argv[2]
    example_sofast_command_line_tool(dir_cal_in, dir_save_in)
