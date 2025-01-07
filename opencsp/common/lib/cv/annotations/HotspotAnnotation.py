from opencsp.common.lib.cv.annotations.PointAnnotations import PointAnnotations
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class HotspotAnnotation(PointAnnotations):
    def __init__(self, style: rcps.RenderControlPointSeq = None, point: p2.Pxy = None):
        if style is None:
            style = rcps.default(color=color.magenta(), marker='x', markersize=12)
        super().__init__(style, point)

    def __str__(self):
        return f"HotspotAnnotation<{self.points.x[0]},{self.points.y[0]}>"
