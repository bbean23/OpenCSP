import copy
from typing import Callable

import matplotlib.axes
import matplotlib.backend_bases
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
import opencsp.common.lib.cv.image_reshapers as ir
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractVisualizationImageProcessor import (
    AbstractVisualizationImageProcessor,
)
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class ViewCrossSectionImageProcessor(AbstractVisualizationImageProcessor):
    """
    Interprets the current image as a 2D cross section and either displays it,
    or if interactive it displays the plot and waits on the next press of the
    "enter" key.

    This visualization uses either one or two windows to display the cross
    sections, depending on the initialization parameters.
    """

    def __init__(
        self,
        cross_section_location: tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]],
        label: str | rca.RenderControlAxis = 'Light Intensity',
        single_plot: bool = True,
        interactive: bool | Callable[[SpotAnalysisOperable], bool] = False,
        crop_to_threshold: int | None = None,
        y_range: tuple[int, int] = None,
        extra_source_images: list[CacheableImage] | Callable[[SpotAnalysisOperable], list[CacheableImage]] = None,
    ):
        """
        Parameters
        ----------
        cross_section_location : tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]]
            The (x, y) pixel location to take cross sections through.
        label : str | rca.RenderControlAxis, optional
            The label to use for the window title, by default 'Cross Section at
            [cross_section_location]'
        single_plot : bool, optional
            If True, then draw both the horizational and vertical cross section
            graphs on the same plot. If False, then use two separate plots.
            Default is True.
        interactive : bool | Callable[[SpotAnalysisOperable], bool], optional
            If True then the spot analysis pipeline is paused until the user
            presses the "enter" key, by default False
        crop_to_threshold : int | None, optional
            Crops the input image horizontally and vertically to the first/last
            values >= the given threshold. This crop is based on the
            cross_section_location and is done before the cross section is
            measured. This is useful when trying to inspect hot spots where the
            interesting values are limited to a small portion of the image. None
            to not crop the image. By default None.
        y_range : tuple[int, int] | None, optional
            Set the y-range of the cross-section plots. None to not constrain
            the y-axis range with this parameter. Default is None.
        extra_source_images: list[CacheableImage] | Callable[[SpotAnalysisOperable], list[CacheableImage]] | None
            Additional images to draw the cross section for in the
            visualizations, or None to not draw the cross section for any
            additional images. The cross section of these additional images will
            be drawn under the plot of the operable's primary image. Default is None.
        """
        label_for_name = "" if label.strip() == "" else "_" + label
        super().__init__(interactive, self.__class__.__name__ + label_for_name)

        self.cross_section_location = cross_section_location
        self.label = label
        self.single_plot = single_plot
        self.crop_to_threshold = crop_to_threshold
        self.y_range = y_range
        self.extra_source_images = extra_source_images

        # initialize certain visualization values
        self.horizontal_style = rcps.RenderControlPointSeq(color=color.magenta(), linewidth=2, marker='None')
        self.vertical_style = rcps.RenderControlPointSeq(color=color.plot_colors["brown"], linewidth=2, marker='None')

        # declare future values
        self.view_specs: list[dict]
        self.rc_axises: list[rca.RenderControlAxis]
        self.fig_records: list[rcfr.RenderControlFigureRecord]
        self.views: list[v3d.View3d]
        self.axes: list[matplotlib.axes.Axes]
        self.plot_titles: list[str]

    @property
    def num_figures(self) -> int:
        if self.single_plot:
            return 2
        else:
            return 3

    def _init_figure_records(
        self, render_control_figure: rcfg.RenderControlFigure
    ) -> list[rcfr.RenderControlFigureRecord]:
        self.view_specs = []
        self.rc_axises = []
        self.fig_records = []
        self.views = []
        self.axes = []
        self.plot_titles = []

        setup_figure = lambda rc_axis, view_spec, name: fm.setup_figure(
            render_control_figure,
            rc_axis,
            view_spec,
            equal=False,
            number_in_name=False,
            name=name,
            title="",
            code_tag=f"{__file__}._init_figure_records()",
        )

        if self.single_plot:
            plot_titles = ["Image"]
        else:
            plot_titles = ["Image", "Horizontal CS: ", "Vertical CS: "]

        for plot_title in plot_titles:
            if plot_title == "Image":
                rc_axis = rca.RenderControlAxis()
                view_spec = vs.view_spec_pq()
                fig_record = setup_figure(rc_axis, view_spec, "Cross Sections")

            else:
                if self.single_plot:
                    rc_axis = rca.RenderControlAxis(x_label='index', y_label='value')
                    name_suffix = ""
                else:
                    if "Horizontal" in plot_title:
                        rc_axis = rca.RenderControlAxis(x_label='x', y_label='value')
                        name_suffix = " (Horizontal)"
                    else:
                        rc_axis = rca.RenderControlAxis(x_label='y', y_label='value')
                        name_suffix = " (Vertical)"

                view_spec = vs.view_spec_xy()
                fig_record = setup_figure(rc_axis, view_spec, self.label + name_suffix)

            self.view_specs.append(view_spec)
            self.rc_axises.append(rc_axis)
            self.fig_records.append(fig_record)
            self.views.append(fig_record.view)
            self.axes.append(fig_record.figure.gca())
            self.plot_titles.append(plot_title)

        return self.fig_records

    def _visualize_operable(
        self, operable: SpotAnalysisOperable, is_last: bool
    ) -> tuple[list[CacheableImage], list[rcfr.RenderControlFigureRecord]]:
        image = operable.primary_image.nparray
        width, height = image.shape[1], image.shape[0]

        # get the cross section pixel location
        if isinstance(self.cross_section_location, tuple):
            cs_loc = self.cross_section_location
        else:
            cs_loc = self.cross_section_location(operable)
        if cs_loc is None:
            return None, None
        cross_sec_x, cross_sec_y = cs_loc

        # subselect a piece of the image based on the crop threshold
        y_start, y_end, x_start, x_end = 0, height, 0, width
        cs_cropped_x, cs_cropped_y = cross_sec_x, cross_sec_y
        cropped_width, cropped_height = width, height
        if self.crop_to_threshold is not None:
            y_start, y_end, x_start, x_end = it.range_for_threshold(image, self.crop_to_threshold)

            # check that this cropped range contains the cross section target
            if cross_sec_x >= x_start and cross_sec_x < x_end:
                if cross_sec_y >= y_start and cross_sec_y < y_end:
                    image = image[y_start:y_end, x_start:x_end]
                    cs_cropped_x, cs_cropped_y = cross_sec_x - x_start, cross_sec_y - y_start
                    cropped_width, cropped_height = x_end - x_start, y_end - y_start

        # matplotlib puts the origin in the bottom left instead of the top left
        cs_cropped_y_mlab = cropped_height - cs_cropped_y

        # Clear the previous plot
        for fig_record in self.fig_records:
            fig_record.clear()

        # Update the title
        for plot_title_prefix, fig_record in zip(self.plot_titles, self.fig_records):
            fig_record.title = plot_title_prefix + operable.best_primary_nameext

        # Get the additional images to get the cross section for
        additional_images: list[CacheableImage] = []
        if self.extra_source_images is not None:
            if isinstance(self.extra_source_images, list):
                additional_images = self.extra_source_images
            else:
                additional_images = self.extra_source_images(operable)

        for source_image in ["primary"] + additional_images:
            # get the 'np_image' and the plot label
            is_primary = source_image == "primary"
            if is_primary:
                plot_label = "Primary " if len(additional_images) > 0 else ""
                np_image = image
                hstyle = self.horizontal_style
                vstyle = self.vertical_style
            else:
                _, plot_label, _ = ft.path_components(source_image.source_path)
                plot_label += " "
                np_image = source_image.nparray
                hstyle = copy.deepcopy(self.horizontal_style)
                hstyle.color = color.green()
                vstyle = copy.deepcopy(self.horizontal_style)
                vstyle.color = color.blue()

            # Get the cross sections
            v_cross_section = np_image[:, cs_cropped_x : cs_cropped_x + 1].squeeze().tolist()
            v_p_list = list(range(len(v_cross_section)))
            h_cross_section = np_image[cs_cropped_y : cs_cropped_y + 1, :].squeeze().tolist()
            h_p_list = list(range(len(h_cross_section)))

            if self.single_plot:
                # Align the cross sections so that the intersect point overlaps
                if cs_cropped_x < cs_cropped_y:
                    diff = cs_cropped_x - cs_cropped_y
                    v_p_list = [i + diff for i in v_p_list]
                if cs_cropped_y < cs_cropped_x:
                    diff = cs_cropped_y - cs_cropped_x
                    h_p_list = [i + diff for i in h_p_list]
            else:
                # Translate the cross sections plots to their actual locations
                v_p_list = [i + y_start for i in v_p_list]
                h_p_list = [i + x_start for i in h_p_list]

            # Draw the image plot
            if is_primary:
                i_view = self.views[0]
                np_image = ir.false_color_reshaper(np_image, 255)
                i_view.draw_image(np_image, (0, 0), (cropped_width, cropped_height))
                i_view.draw_pq_list([(cs_cropped_x, 0), (cs_cropped_x, cropped_height)], style=vstyle)
                i_view.draw_pq_list([(0, cs_cropped_y_mlab), (cropped_width, cs_cropped_y_mlab)], style=hstyle)

            # Draw the new cross sections plot using the same axes
            v_view = self.views[1]
            h_view = self.views[1]
            if not self.single_plot:
                v_view = self.views[2]
            v_view.draw_pq_list(
                zip(v_p_list, v_cross_section), style=vstyle, label=plot_label + "Vertical Cross Section"
            )
            h_view.draw_pq_list(
                zip(h_p_list, h_cross_section), style=hstyle, label=plot_label + "Horizontal Cross Section"
            )

            # explicitly set the y-axis range
            if self.y_range is not None:
                h_view.y_limits = self.y_range
                v_view.y_limits = self.y_range

        # draw
        for view in self.views:
            legend = (self.single_plot) or (len(additional_images) > 0)
            view.show(block=False, legend=legend)

        return [], self.fig_records

    def close_figures(self):
        for view in self.views:
            with et.ignored(Exception):
                view.close()

        self.view_specs.clear()
        self.rc_axises.clear()
        self.fig_records.clear()
        self.views.clear()
        self.axes.clear()
        self.plot_titles.clear()


if __name__ == "__main__":
    from opencsp.common.lib.cv.CacheableImage import CacheableImage

    row = np.arange(100)
    rows = np.repeat(row, 100, axis=0).reshape(100, 100)
    # array([[ 0,  0,  0, ...,  0,  0,  0],
    #        [ 1,  1,  1, ...,  1,  1,  1],
    #        [ 2,  2,  2, ...,  2,  2,  2],
    #        ...,
    #        [97, 97, 97, ..., 97, 97, 97],
    #        [98, 98, 98, ..., 98, 98, 98],
    #        [99, 99, 99, ..., 99, 99, 99]])
    cacheable_rows = CacheableImage(rows, source_path=__file__)

    processor = ViewCrossSectionImageProcessor((50, 50), single_plot=False, interactive=True, crop_to_threshold=20)
    processor.process_operable(SpotAnalysisOperable(cacheable_rows))
