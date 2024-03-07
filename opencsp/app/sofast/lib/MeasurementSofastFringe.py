"""Measurement class for SofastFringe
"""
import datetime as dt

import numpy as np

from opencsp.app.sofast.lib.ImageCalibrationAbstract import ImageCalibrationAbstract
import opencsp.common.lib.geometry.LoopXY as lxy
import opencsp.common.lib.geometry.Vxy as vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as hdf5_tools
import opencsp.common.lib.tool.log_tools as lt


class MeasurementSofastFringe:
    """SofastFringe measurement data class that contains captured images
    and metadata about the measurement.
    """

    def __init__(
        self,
        mask_images: np.ndarray,
        fringe_images: np.ndarray,
        fringe_periods_x: np.ndarray,
        fringe_periods_y: np.ndarray,
        measure_point: Vxyz,
        optic_screen_dist: float,
        date: dt.datetime,
        name: str = '',
    ) -> 'MeasurementSofastFringe':
        """
        A measurement contains 2 (MxN) mask images, n (MxN) images of
        horizontal/vertical fringes.

        Parameters
        ----------
        mask_images : ndarray
            MxNx2 frame array.
        fringe_images : ndarray
            MxNxN frame array. Y fringes first
        fringe_periods_x/y : 1d array
            Periods used to generate x/y fringes, fractional screens.
        measure_point : Vxyz
            Location of measure point, meters.
        optic_screen_dist : float
            Optic-screen distance, meters.
        date : datetime
            Collection date/time.
        name : str
            Name or serial number of measurement.

        """
        # Check mask image size
        if mask_images.shape[2] != 2 or np.ndim(mask_images) != 3:
            raise ValueError(
                f'Two mask images needed, but {mask_images.shape[2]} given.'
            )

        # Save input measurement data
        self.mask_images = mask_images
        self.fringe_images = fringe_images
        self.fringe_periods_x = fringe_periods_x
        self.fringe_periods_y = fringe_periods_y
        self.measure_point = measure_point
        self.optic_screen_dist = optic_screen_dist
        self.date = date
        self.name = name

        # Save calculations
        self.image_shape_xy = np.flip(self.mask_images.shape[:2])
        self.image_shape_yx = self.mask_images.shape[:2]
        self.phase_shifts = int(4)
        self.num_y_ims = self.fringe_periods_y.size * self.phase_shifts
        self.num_x_ims = self.fringe_periods_x.size * self.phase_shifts
        self.num_fringe_ims = self.fringe_images.shape[2]
        # Check number of input fringes
        if (self.num_y_ims + self.num_x_ims) != self.num_fringe_ims or np.ndim(
            fringe_images
        ) != 3:
            raise ValueError(
                f'Incorrect number of fringe images given. Fringe images shape = {fringe_images.shape}.'
            )

        # Instantiate calibration objected fringes
        self._fringe_images_calibrated = None

    def __repr__(self) -> str:
        return 'MeasurementSofastFringe: { ' + self.name + ' }'

    @property
    def fringe_images_y(self) -> np.ndarray:
        """Returns raw y-only fringes"""
        return self.fringe_images[..., : self.num_y_ims]

    @property
    def fringe_images_x(self) -> np.ndarray:
        """Returns raw x-only fringes"""
        return self.fringe_images[..., self.num_y_ims:]

    @property
    def fringe_images_calibrated(self) -> np.ndarray:
        """Returns calibrated fringes"""
        if self._fringe_images_calibrated is None:
            raise ValueError('Fringe images have not been calibrated.')

        return self._fringe_images_calibrated

    @property
    def fringe_images_y_calibrated(self) -> np.ndarray:
        """Returns calibrated y-only fringes"""
        return self.fringe_images_calibrated[..., : self.num_y_ims]

    @property
    def fringe_images_x_calibrated(self) -> np.ndarray:
        """Returns calibrated x-only fringes"""
        return self.fringe_images_calibrated[..., self.num_y_ims:]

    def calibrate_fringe_images(
        self, calibration: ImageCalibrationAbstract, **kwargs
    ) -> None:
        """
        Performs brightness level calibration on the raw captured fringes.

        Parameters
        ----------
        calibration : ImageCalibrationAbstract
            Image Calibration object.
        **kwargs
            Other keyword arguments to pass into ImageCalibration object
            "apply_to_images" method.

        """
        if not isinstance(calibration, ImageCalibrationAbstract):
            raise ValueError(
                'Input calibration must be instance of ImageCalibrationAbstract.'
            )

        self._fringe_images_calibrated = calibration.apply_to_images(self, **kwargs)

    def custom_mask_region(self, keep_region: lxy.LoopXY):
        """ Sets all pixels in the mask images that are NOT in the given region to 0.

        Sometimes the mask images have poor contrast between the optic and the
        surroundings. This can happen, for example, if the reflectivity of the
        optic and the surrounding are similar, or if the ambient light is too
        high relative to the projector. In this cases, it might be necessary to
        assist the computer vision by clipping out a specific region that
        isolates the optic in the mask images.

        Parameters
        ----------
        keep_region : lxy.LoopXY
            A region within the mask image to keep. Pixels outside this region
            will be set to 0 (black).
        """
        # get some values about the mask images
        height, width = self.mask_images.shape[0], self.mask_images.shape[1]

        # sanity check: does the region fit the image?
        for point in keep_region.vertices:
            if point.x < 0 or point.x > width or point.y < 0 or point.y > height:
                lt.warn("Warning in Measurement.custom_mask_region(): " +
                        "at least one vertex of the keep_region is outside the bounds of the mask images " +
                        f"(point x/y: {point.x}/{point.y}, mask image width/height: {width}/{height})")
                break

        # any pixels matching "is_outside_region" will be set to 0 (black)
        is_inside_region = np.zeros((height, width), dtype=np.bool_)
        is_outside_region = None

        # For each column of the mask, determine if those pixels lie inside the
        # border. Update the is_inside_region array with those truth values.
        y_indicies = list(range(height))
        for i in range(width):
            x_indicies = [i] * height
            pnts = vxy.Vxy([x_indicies, y_indicies])
            col_mask = keep_region.is_inside_or_on_border(pnts)
            is_inside_region[:, i] = col_mask

        # clear the pixels not contained in the region
        is_outside_region = np.logical_not(is_inside_region)
        self.mask_images[is_outside_region] = 0

    @classmethod
    def load_from_hdf(cls, file) -> 'MeasurementSofastFringe':
        """
        Loads from HDF file

        Parameters
        ----------
        file : string
            HDF file to load

        """
        # Load grid data
        datasets = [
            'MeasurementSofastFringe/mask_images',
            'MeasurementSofastFringe/fringe_images',
            'MeasurementSofastFringe/fringe_periods_x',
            'MeasurementSofastFringe/fringe_periods_y',
            'MeasurementSofastFringe/measure_point',
            'MeasurementSofastFringe/optic_screen_dist',
            'MeasurementSofastFringe/date',
            'MeasurementSofastFringe/name',
        ]
        kwargs = hdf5_tools.load_hdf5_datasets(datasets, file)

        kwargs['measure_point'] = Vxyz(kwargs['measure_point'])
        kwargs['date'] = dt.datetime.fromisoformat(kwargs['date'])

        return cls(**kwargs)

    def save_to_hdf(self, file) -> None:
        """
        Saves to HDF file

        Parameters
        ----------
        file : string
            HDF file to save

        NOTE: Collection date is saved as string in iso-format.
        """
        datasets = [
            'MeasurementSofastFringe/mask_images',
            'MeasurementSofastFringe/fringe_images',
            'MeasurementSofastFringe/fringe_periods_x',
            'MeasurementSofastFringe/fringe_periods_y',
            'MeasurementSofastFringe/measure_point',
            'MeasurementSofastFringe/optic_screen_dist',
            'MeasurementSofastFringe/date',
            'MeasurementSofastFringe/name',
        ]
        data = [
            self.mask_images,
            self.fringe_images,
            self.fringe_periods_x,
            self.fringe_periods_y,
            self.measure_point.data.squeeze(),
            self.optic_screen_dist,
            self.date.isoformat(),
            self.name,
        ]

        # Save data
        hdf5_tools.save_hdf5_datasets(data, datasets, file)
