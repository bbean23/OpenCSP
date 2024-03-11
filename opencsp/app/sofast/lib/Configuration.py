import copy
import functools
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet

from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.app.sofast.lib.ImageCalibrationGlobal import ImageCalibrationGlobal
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.common.lib.camera.Camera import Camera
import opencsp.common.lib.tool.log_tools as lt


class Configuration():
    def __init__(self,
                 file_measurement_dir_name_ext: str,  # fringe images, ex 'measurement_facet.h5'
                 file_camera_dir_name_ext: str,  # measured camera distortion, ex 'camera.h5'
                 file_display_dir_name_ext: str,  # 'display_distorted_2d.h5'
                 file_calibration_dir_name_ext: str = None,  # 'calibration.h5'
                 file_ensemble_data_dir_name_ext: str = None,
                 file_facet_data_dir_name_ext: str | list[str] = None,  # 'Facet_NSTTF.json'
                 surface_data: list[dict[str, any]] = None):
        """ Class to hold all configuration parameters for a SOFAST measurement.

        Parameters
        ----------
        file_measurement_dir_name_ext : str
            Path to the measurement hdf5 file, containing the mask and fringe
            images.
        file_camera_dir_name_ext: str
            Path to the sofast camera hdf5 file, containing the camera
            characterization values.
        file_display_dir_name_ext: str
            Path to the display hdf5 file, containing the screen shape, and
            screen-camera rotation and translation.
        file_calibration_dir_name_ext: str
            Path to the calibration hdf5 file, containing the projector values
            and camera response.
        file_ensemble_data_dir_name_ext: str
            Path to the ensemble json file, containing TODO.
            ensemble_data.
        file_facet_data_dir_name_ext: str
            Path to the facet json file, containing the approximate facet
            corners and center locations.
        surface_data: list[dict[str,any]]
            Defines surface fitting parameters, one dict per facet. See
            SlopeSolver documentation for more information.
        """
        from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe  # import here to avoid a circular loop

        self._file_measurement_dir_name_ext = file_measurement_dir_name_ext
        self._file_camera_dir_name_ext = file_camera_dir_name_ext
        self._file_display_dir_name_ext = file_display_dir_name_ext
        self._file_calibration_dir_name_ext = file_calibration_dir_name_ext
        self._file_ensemble_data_dir_name_ext = file_ensemble_data_dir_name_ext
        self._file_facet_data_dir_name_ext = file_facet_data_dir_name_ext
        self._surface_data = surface_data or []

        # check input for errors
        if self._file_facet_data_dir_name_ext != None:
            # validate the facet data
            if not self._validate_num_facets():
                lt.error_and_raise(ValueError, 'Error in Sofast Configuration(): ' +
                                   f'Given length of facet data is {len(self.facet_data):d} ' +
                                   f'but ensemble_data expects {self.ensemble_data.num_facets:d} facets.')
            # validate the surface data
            for sd in self._surface_data:
                ProcessSofastFringe._check_surface_data(sd)

    @functools.cached_property
    def measurement(self):
        """ Loads and returns the measurement instance. Always returns the same instance. """
        return MeasurementSofastFringe.load_from_hdf(self._file_measurement_dir_name_ext)

    @functools.cached_property
    def camera(self):
        """ Loads and returns the camera instance. Always returns the same instance. """
        return Camera.load_from_hdf(self._file_camera_dir_name_ext)

    @functools.cached_property
    def display(self):
        """ Loads and returns the display instance. Always returns the same instance. """
        return DisplayShape.load_from_hdf(self._file_display_dir_name_ext)

    @functools.cached_property
    def calibration(self):
        """ Loads and returns the calibration instance. Always returns the same instance. """
        try:
            return ImageCalibrationGlobal.load_from_hdf(self._file_calibration_dir_name_ext)
        except:
            return ImageCalibrationScaling.load_from_hdf(self._file_calibration_dir_name_ext)

    @functools.cached_property
    def ensemble_data(self):
        """ Loads and returns the ensemble instance. Always returns the same instance. """
        return DefinitionEnsemble.load_from_json(self._file_ensemble_data_dir_name_ext)

    @functools.cached_property
    def facet_data(self):
        """ Loads and returns the facet_data instance(s). Always returns the same instance. """
        file_facet_data_dir_name_ext = self._file_facet_data_dir_name_ext

        # Load one or more facet files
        if isinstance(file_facet_data_dir_name_ext, str):
            file_facet_data_dir_name_ext = [file_facet_data_dir_name_ext]
        facets = list(map(DefinitionFacet.load_from_json, file_facet_data_dir_name_ext))

        return facets

    @functools.cached_property
    def surface_data(self):
        return copy.deepcopy(self._surface_data)

    def reset(self):
        """ Clear the various cached properties of this instance. The next time
        the properties are retrieved they will be recomputed. """
        del self.measurement
        del self.camera
        del self.display
        del self.calibration
        del self.ensemble_data
        del self.facet_data
        del self.surface_data

    def _validate_num_facets(self):
        # attempt to load the facets
        facet_data = None
        try:
            facet_data = self.facet_data
        except:
            pass

        # attempt to load the ensemble data
        ensemble_data = None
        try:
            ensemble_data = self.ensemble_data
        except:
            pass

        # check that we have the right number of facets
        if facet_data != None and ensemble_data != None:
            if len(facet_data) != ensemble_data.num_facets:
                return False
        return True
