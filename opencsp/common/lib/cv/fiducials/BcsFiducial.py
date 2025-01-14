import matplotlib.axes
import matplotlib.patches
import numpy as np
import scipy.spatial

from opencsp.common.lib.cv.fiducials.AbstractFiducials import AbstractFiducials
import opencsp.common.lib.geometry.LoopXY as loop
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.Vxyz as v3
import opencsp.common.lib.render_control.RenderControlBcs as rcb
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr


class BcsFiducial(AbstractFiducials):
    def __init__(
        self, origin_px: p2.Pxy, radius_px: float, style: rcb.RenderControlBcs = None, pixels_to_meters: float = 0.1
    ):
        """
        Fiducial for indicating where the BCS target is in an image.

        Parameters
        ----------
        origin_px : Pxy
            The center point of the BCS target, in pixels
        radius_px : float
            The radius of the BCS target, in pixels
        style : RenderControlBcs, optional
            The rendering style, by default None
        pixels_to_meters : float, optional
            A simple conversion method for how many meters a pixel represents, for use in scale(). by default 0.1
        """
        super().__init__(style=style)
        self.origin_px = origin_px
        self.radius_px = radius_px
        self.pixels_to_meters = pixels_to_meters

    def get_bounding_box(self, index=0) -> reg.RegionXY:
        x1, x2 = self.origin.x[0] - self.radius_px, self.origin.x[0] + self.radius_px
        y1, y2 = self.origin.y[0] - self.radius_px, self.origin.y[0] + self.radius_px
        return reg.RegionXY(loop.LoopXY.from_rectangle(x1, y1, x2 - x1, y2 - y1))

    @property
    def origin(self) -> p2.Pxy:
        return self.origin_px

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        raise NotImplementedError("rotation is not yet implemented for PointFiducials")

    @property
    def size(self) -> list[float]:
        return [self.radius_px * 2]

    @property
    def scale(self) -> list[float]:
        return [self.size * self.pixels_to_meters]

    def render_to_figure(self, fig: rcfr.RenderControlFigureRecord, image: np.ndarray, include_label=False):
        label = self.get_label(include_label)

        if self.style.linestyle is not None:
            circ = matplotlib.patches.Circle(
                self.origin.data.tolist(),
                self.radius_px,
                color=self.style.color,
                linestyle=self.style.linestyle,
                linewidth=self.style.linewidth,
                fill=False,
                label=label,
            )
            fig.view.axis.add_patch(circ)
            label = None

        if self.style.marker is not None:
            fig.view.draw_pq(([self.origin.x], [self.origin.y]), style=self.style, label=label)
            label = None
