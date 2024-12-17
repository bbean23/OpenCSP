import copy
import numpy as np

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import ViewCrossSectionImageProcessor
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render.Color as color


class ViewCustomCrossSectionImageProcessor(ViewCrossSectionImageProcessor):
    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)

        self.no_sun_image: np.ndarray = None

    def pre_visualize(
        self, operable: SpotAnalysisOperable, cs_loc: tuple[int, int], cropped_region: tuple[int, int, int, int]
    ) -> int:
        if self.no_sun_image is not None:
            # get the cropped no-sun image
            no_sun_image = self.no_sun_image.copy()
            cx1, cy1, cx2, cy2 = cropped_region
            no_sun_image = no_sun_image[cy1:cy2, cx1:cx2, ...]

            # get the render styles for the no-sun image
            vstyle = copy.copy(self.vertical_style)
            hstyle = copy.copy(self.horizontal_style)
            vstyle.set_color(color.yellow())
            hstyle.set_color(color.plot_colors["purple"])

            # add the no-sun cross sections to the plots
            label = "No Sun"
            ret = self._draw_cross_section(no_sun_image, cs_loc, cropped_region, vstyle, hstyle, label, label)
            return ret

        else:
            return 0

    def post_visualize(
        self, operable: SpotAnalysisOperable, cs_loc: tuple[int, int], cropped_region: tuple[int, int, int, int]
    ) -> int:
        return 0
