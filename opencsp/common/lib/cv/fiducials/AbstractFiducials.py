from abc import ABC, abstractmethod
from typing import Callable

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.geometry.Vxyz as v3
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class AbstractFiducials(ABC):
    """
    A collection of markers (such as an ArUco board) that is used to orient the camera relative to observed objects
    in the scene. It is suggested that each implementing class be paired with a complementary locator method or
    SpotAnalysisImageProcessor.
    """

    def __init__(self, style: rcps.RenderControlPointSeq = None, pixels_to_meters: Callable[[p2.Pxy], v3.Vxyz] = None):
        """
        Parameters
        ----------
        style : RenderControlPointSeq, optional
            How to render this fiducial when using the default
            :py:meth:`render_to_plot` method. By default rcps.default().
        pixels_to_meters : Callable[[p2.Pxy], v3.Vxyz], optional
            Conversion function to get the physical point in space for the given x/y position information. Used in the
            default self.scale implementation. A good implementation of this function will correct for many factors such
            as relative camera position and camera distortion. For extreme accuracy, this will also account for
            non-uniformity in the target surface. Defaults to a simple 1 meter per pixel model.
        """
        self.style = style if style is not None else rcps.default()
        self.pixels_to_meters = pixels_to_meters

    @abstractmethod
    def get_bounding_box(self, index=0) -> reg.RegionXY:
        """The X/Y bounding box(es) of this instance, in pixels."""

    @property
    @abstractmethod
    def origin(self) -> p2.Pxy:
        """The origin point(s) of this instance, in pixels."""

    @property
    @abstractmethod
    def rotation(self) -> scipy.spatial.transform.Rotation:
        """
        The pointing of the normal vector(s) of this instance.
        This is relative to the camera's reference frame, where x is positive
        to the right, y is positive down, and z is positive in (away from the
        camera).

        This can be used to describe the forward transformation from the
        camera's perspective. For example, an aruco marker whose origin is in
        the center of the image and is facing towards the camera could have the
        rotation::

            Rotation.from_euler('y', np.pi)

        If that same aruco marker was also placed upside down, then it's
        rotation could be::

            Rotation.from_euler(
                'yz',
                [ [np.pi, 0],
                  [0,     np.pi] ]
            )

        Not that this just describes rotation, and not the translation. We call
        the rotation and translation together the orientation.
        """

    @property
    @abstractmethod
    def size(self) -> list[float]:
        """The scale(s) of this fiducial, in pixels, relative to its longest axis.
        For example, if the fiducial is a square QR-code and is oriented tangent
        to the camera, then the scale will be the number of pixels from one
        corner to the other."""  # TODO is this a good definition?

    @property
    def scale(self) -> list[float]:
        """
        The scale(s) of this fiducial, in meters, relative to its longest axis.
        This can be used to determine the distance and rotation of the
        fiducial relative to the camera.
        """
        ret = []

        for i in range(len(self.origin)):
            bb = self.get_bounding_box(i)
            left_px, right_px, bottom_px, top_px = bb.loops[0].axis_aligned_bounding_box()
            top_left_m = self.pixels_to_meters(p2.Pxy([left_px, top_px]))
            bottom_right_m = self.pixels_to_meters(p2.Pxy([right_px, bottom_px]))
            scale = (bottom_right_m - top_left_m).magnitude()[0]
            ret.append(scale)

        return ret

    def _render(self, axes: matplotlib.axes.Axes):
        """
        Called from render(). The parameters are always guaranteed to be set.
        """
        axes.scatter(
            self.origin.x,
            self.origin.y,
            linewidth=self.style.linewidth,
            marker=self.style.marker,
            s=self.style.markersize,
            color=self.style.markerfacecolor,
            edgecolor=self.style.markeredgecolor,
        )

    def render_to_plot(self, axes: matplotlib.axes.Axes = None):
        """
        Renders this fiducial to the active matplotlib.pyplot plot.

        The default implementation uses plt.scatter().

        Parameters
        ----------
        axes: matplotlib.axes.Axes, optional
            The plot to render to. Uses the active plot if None. Default is None.
        """
        if axes is None:
            axes = plt.gca()
        self._render(axes)

    def render_to_figure(self, fig_record: rcfr.RenderControlFigureRecord, image: np.ndarray = None):
        """
        Renders a visual representation of this fiducial to the given fig_record.

        The given image should have already been rendered to the figure record
        if it is set. If this has been called from :py:meth:`render_to_image`
        then image is guaranteed to be set.

        Parameters
        ----------
        fig_record : rcfr.RenderControlFigureRecord
            The record to render with. Most render methods should be available
            via fig_record.view.draw_*().
        image : np.ndarray, optional
            The image that was already rendered to the figure record, or None if
            there hasn't been an image rendered or that data just isn't
            available. By default None, or the image passed in to
            :py:meth:`render_to_image` if being called from that method.

        Raises
        ------
        NotImplementedError
            If this method hasn't been implemented yet in one of the child
            classes of :py:class:`AbstractFiducials`.
        """
        raise NotImplementedError

    def render_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Renders this fiducial to the a new image on top of the given image.

        The default implementation creates a new matplotlib plot, and then
        renders to it with either :py:meth:`render_to_figure` or
        :py:meth:`render_to_plot`, depending on which has been implemented.
        """
        # Create the figure to plot to
        (height_px, width_px), nchannel = it.dims_and_nchannels(image)
        figsize = rcfg.RenderControlFigure.pixel_resolution_inches(width_px, height_px)
        figure_control = rcfg.RenderControlFigure(
            tile=False, figsize=figsize, grid=False, draw_whitespace_padding=False
        )
        view_spec_2d = vs.view_spec_im()

        fig_record = fm.setup_figure_for_3d_data(
            figure_control,
            rca.image(draw_axes=False, grid=False),
            view_spec_2d,
            equal=False,
            name=self.__class__.__name__,
            code_tag=f"{__file__}.render_to_image()",
        )

        try:
            # A portion of this code is from:
            # https://stackoverflow.com/questions/35355930/figure-to-image-as-a-numpy-array

            # Prepare the image
            fig_record.view.imshow(image)

            # render
            try:
                self.render_to_figure(fig_record, image)
            except NotImplementedError:
                self.render_to_plot(fig_record.axis)

            # Convert back to a numpy array
            new_image = fig_record.to_array()
            new_image = new_image.astype(image.dtype)

            # Return the updated image
            return new_image

        except Exception as ex:
            lt.error("Error in AbstractFiducials.render_to_image(): " + repr(ex))
            raise

        finally:
            plt.close(fig_record.figure)
