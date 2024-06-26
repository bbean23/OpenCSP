"""Unit test suite to test the System class
"""
import os

import pytest

import opencsp
from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.System import System
from opencsp.app.sofast.test.ImageAcquisition_no_camera import ImageAcquisition
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection


@pytest.mark.no_xvfb
def test_System():
    # Get test data location
    base_dir = os.path.join(
        os.path.dirname(opencsp.__file__), 'test/data/sofast_measurements'
    )

    # Create fringe object
    periods_x = [0.9, 3.9]
    periods_y = [15.9, 63.9]
    F = Fringes(periods_x, periods_y)

    # Instantiate image projection class
    im_proj = ImageProjection.load_from_hdf_and_display(
        os.path.join(base_dir, 'general/image_projection_test.h5')
    )

    # Instantiate image acquisition class
    im_aq = ImageAcquisition()

    # Set camera settings
    im_aq.frame_size = (100, 80)
    im_aq.frame_rate = 7
    im_aq.exposure_time = 300000
    im_aq.gain = 230

    # Create system class
    system = System(im_proj, im_aq)

    # Load fringes
    system.load_fringes(F, 0)

    # Define functions to put in system queue
    def f1():
        system.capture_mask_and_fringe_images(system.run_next_in_queue)

    def f2():
        system.close_all()

    # Load function in queue
    system.set_queue([f1, f2])

    # Run
    system.run()


if __name__ == '__main__':
    test_System()
