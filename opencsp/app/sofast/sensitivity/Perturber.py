import copy
import functools
import numpy as np

from opencsp.app.sofast.lib.Configuration import Configuration
import opencsp.app.sofast.sensitivity.ErrorSources as es


class Perturber(Configuration):
    def __init__(self,
                 base_config: Configuration,
                 errors: es.ErrorSources = None,
                 camera_characterization_images_dir: str = None,
                 aruco_marker_calibration_images_dir: str = None):
        errors = errors if errors is not None else es.ErrorSources()

        # clear the base instance's cache
        self.base_config.reset()

        self.base_config = base_config
        self._errors = errors
        self.camera_characterization_images_dir = camera_characterization_images_dir
        self.aruco_marker_calibration_images_dir = aruco_marker_calibration_images_dir

    @property
    def errors(self):
        return copy.deepcopy(self._errors)

    @errors.setter
    def errors(self, new_errors: es.ErrorSources):
        self._errors = new_errors
        self.base_config.reset()
        self.reset()

    @functools.cached_property
    def measurement(self):
        measurement = self.base_config.measurement
        rng = None

        if self._errors.gain_nose != 0:
            rng = rng or np.random.default_rng()
            cam_min = np.min(self.calibration.camera_values)
            cam_max = np.max(self.calibration.camera_values)
            noise_scale = self._errors.gain_nose * (cam_max - cam_min)
            fringe_dims = measurement.fringe_images.shape[0:2]

            for i in range(measurement.num_fringe_ims):
                noise = rng.standard_normal(fringe_dims) * noise_scale
                measurement.fringe_images[i] += noise

        return measurement

    @functools.cached_property
    def camera(self):
        return self.base_config.camera

    @functools.cached_property
    def display(self):
        return self.base_config.display

    @functools.cached_property
    def calibration(self):
        return self.base_config.calibration

    @functools.cached_property
    def ensemble_data(self):
        return self.base_config.ensemble_data

    @functools.cached_property
    def facet_data(self):
        return self.base_config.facet_data
