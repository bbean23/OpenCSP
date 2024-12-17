import copy
from typing import Callable

import matplotlib.axes
import matplotlib.backend_bases
import numpy as np

from contrib.common.lib.cv.spot_analysis.PixelLocation import PixelLocation
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

    Custom rendering code can be added by extending this class and overriding
    the :py:meth:`pre_visualize` and :py:meth:`post_visualize` functions.
    """

    def __init__(
        self,
        cross_section_location: (
            Callable[[SpotAnalysisOperable], tuple[int, int]] | tuple[int, int] | str | PixelLocation
        ) = None,
        label: str | rca.RenderControlAxis = 'Light Intensity',
        single_plot: bool = True,
        interactive: bool | Callable[[SpotAnalysisOperable], bool] = False,
        crop_to_threshold: int | None = None,
        y_range: tuple[int, int] = None,
    ):
        """
        Parameters
        ----------
        cross_section_location : Callable[[SpotAnalysisOperable], tuple[int, int]] | tuple[int, int] | str | PixelLocation
            The (x, y) pixel location to take cross sections through.
        label : str | rca.RenderControlAxis, optional
            The label to use for the window title, by default 'Light Intensity
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
        """
        label_for_name = "" if label.strip() == "" else "_" + label
        super().__init__(interactive, self.__class__.__name__ + label_for_name)

        # validate input
        if cross_section_location is None:
            cross_section_location = "center"

        self.cross_section_location = PixelLocation(cross_section_location)
        self.label = label
        self.single_plot = single_plot
        self.crop_to_threshold = crop_to_threshold
        self.y_range = y_range

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

    @property
    def _figure_records(self) -> tuple[rcfr.RenderControlFigureRecord, rcfr.RenderControlFigureRecord]:
        """The vertical and horizontal figure records. They might be the same instance."""
        v_fig_record = self.fig_records[1]
        h_fig_record = self.fig_records[1]
        if not self.single_plot:
            v_fig_record = self.fig_records[2]
        return v_fig_record, h_fig_record

    def pre_visualize(
        self, operable: SpotAnalysisOperable, cs_loc: tuple[int, int], cropped_region: tuple[int, int, int, int]
    ) -> int:
        """
        An overrideable method that is called before the main cross section
        drawing code. The user can extend the ViewCrossSectionImageProcessor
        class to implement additional drawing code with this method and the
        :py:meth:`post_visualize` method.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable being rendered.
        cs_loc : tuple[int, int]
            The x/y location of the (cropped) image that the cross section goes through.
        cropped_region : tuple[int, int, int, int]
            The cropped [left,top,right,bottom] region if crop_to_threshold is
            not None, or the full image region if it is None.

        Returns
        -------
        num_additional_graphs: int
            How many additional graphs this method rendered to either of the fig_record plots.

        Notes
        -----
        The figure records to draw to can be obtained with :py:attr:`_fig_records`.
        """
        return 0

    def post_visualize(
        self,
        operable: SpotAnalysisOperable,
        cs_loc: tuple[int, int],
        cropped_region: tuple[int, int, int, int],
        v_fig_record: rcfr.RenderControlFigureRecord,
        h_fig_record: rcfr.RenderControlFigureRecord,
    ) -> int:
        """
        An overrideable method that is called after the main cross section
        drawing code. The user can extend the ViewCrossSectionImageProcessor
        class to implement additional drawing code with this method and the
        :py:meth:`pre_visualize` method.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable being rendered.
        cs_loc : tuple[int, int]
            The x/y location of the (cropped) image that the cross section goes through.
        cropped_region : tuple[int, int, int, int]
            The cropped [left,top,right,bottom] region if crop_to_threshold is
            not None, or the full image region if it is None.

        Returns
        -------
        num_additional_graphs: int
            How many additional graphs this method rendered to either of the fig_record plots.

        Notes
        -----
        The figure records to draw to can be obtained with :py:attr:`_fig_records`.
        """
        return 0

    def _draw_cross_section(
        self,
        np_image: np.ndarray,
        cs_loc: tuple[int, int],
        cropped_region: tuple[int, int, int, int],
        vstyle: rcps.RenderControlPointSeq = None,
        hstyle: rcps.RenderControlPointSeq = None,
        vlabel: str = None,
        hlabel: str = None,
    ) -> int:
        """
        Draws the cross sections onto the vertical and horizontal plots using
        the View3d :py:meth:`draw_pq_list` method.

        Parameters
        ----------
        np_image : np.ndarray
            The image to grab the cross section data from. Should already be cropped according to the cropped_region.
        cs_loc : tuple[int, int]
            The cross section location, for the np_image, with the cropped_region offset already applied.
        cropped_region : tuple[int, int, int, int]
            The [left,top,right,bottom] edges of the crop as was already applied
            to the input np_image. The size of np_image should match this size.
        vstyle : rcps.RenderControlPointSeq, optional
            Style to draw the vertical cross section with, by default :py:attr:`vertical_style`
        hstyle : rcps.RenderControlPointSeq, optional
            Style to draw the horizontal cross section with, by default :py:attr:`horizontal_style`
        vlabel: str, optional
            The label to apply to the vertical graph, by default "Vertical Cross Section"
        hlabel: str, optional
            The label to apply to the horizontal graph, by default "Horizontal Cross Section"

        Returns
        -------
        int
            _description_
        """
        # get default values
        if hstyle is None:
            hstyle = self.horizontal_style
        if vstyle is None:
            vstyle = self.vertical_style
        if hlabel is None:
            hlabel = "Horizontal Cross Section"
        if vlabel is None:
            vlabel = "Vertical Cross Section"

        # Get the cross sections
        cs_loc_x, cs_loc_y = cs_loc[0], cs_loc[1]
        v_cross_section = np_image[:, cs_loc_x : cs_loc_x + 1].squeeze().tolist()
        v_p_list = list(range(len(v_cross_section)))
        h_cross_section = np_image[cs_loc_y : cs_loc_y + 1, :].squeeze().tolist()
        h_p_list = list(range(len(h_cross_section)))

        if self.single_plot:
            # Align the cross sections so that the intersect point overlaps
            if cs_loc_x < cs_loc_y:
                diff = cs_loc_x - cs_loc_y
                v_p_list = [i + diff for i in v_p_list]
            if cs_loc_y < cs_loc_x:
                diff = cs_loc_y - cs_loc_x
                h_p_list = [i + diff for i in h_p_list]
        else:
            # Translate the cross sections plots to their actual locations
            crop_left = cropped_region[0]
            crop_top = cropped_region[1]
            v_p_list = [i + crop_top for i in v_p_list]
            h_p_list = [i + crop_left for i in h_p_list]

        # Draw the cross section graphs
        v_fig_record, h_fig_record = self._figure_records
        v_fig_record.view.draw_pq_list(zip(v_p_list, v_cross_section), style=vstyle, label=vlabel)
        h_fig_record.view.draw_pq_list(zip(h_p_list, h_cross_section), style=hstyle, label=hlabel)

        if self.single_plot:
            return 2
        else:
            return 1

    def _visualize_operable(
        self, operable: SpotAnalysisOperable, is_last: bool
    ) -> tuple[list[CacheableImage], list[rcfr.RenderControlFigureRecord]]:
        np_image = operable.primary_image.nparray
        width, height = np_image.shape[1], np_image.shape[0]

        # get the cross section pixel location
        cs_loc = self.cross_section_location.get_location(operable)
        if cs_loc is None:
            return None, None
        cs_loc = (int(np.round(cs_loc[0])), int(np.round(cs_loc[1])))
        cross_sec_x, cross_sec_y = cs_loc

        # subselect a piece of the image based on the crop threshold
        y_start, y_end, x_start, x_end = 0, height, 0, width
        cs_cropped_x, cs_cropped_y = cross_sec_x, cross_sec_y
        cropped_width, cropped_height = width, height
        if self.crop_to_threshold is not None:
            y_start, y_end, x_start, x_end = it.range_for_threshold(np_image, self.crop_to_threshold)

            # check that this cropped range contains the cross section target
            if cross_sec_x >= x_start and cross_sec_x < x_end:
                if cross_sec_y >= y_start and cross_sec_y < y_end:
                    np_image = np_image[y_start:y_end, x_start:x_end]
                    cs_cropped_x, cs_cropped_y = cross_sec_x - x_start, cross_sec_y - y_start
                    cropped_width, cropped_height = x_end - x_start, y_end - y_start
        cropped_region = tuple([x_start, y_start, x_end, y_end])
        cs_loc_cropped = tuple([cs_cropped_x, cs_cropped_y])

        # matplotlib puts the origin in the bottom left instead of the top left
        cs_cropped_y_mlab = cropped_height - cs_cropped_y

        # Clear the previous plot
        for fig_record in self.fig_records:
            fig_record.clear()

        # Update the title
        for plot_title_prefix, fig_record in zip(self.plot_titles, self.fig_records):
            fig_record.title = plot_title_prefix + operable.best_primary_nameext

        # get the horizontal and vertical figure records
        v_fig_record, h_fig_record = self._figure_records

        # get the style
        hstyle = self.horizontal_style
        vstyle = self.vertical_style

        # Draw the image w/ cross section line overlays
        i_view = self.views[0]
        false_color_image = ir.false_color_reshaper(np_image, 255)
        i_view.draw_image(false_color_image, (0, 0), (cropped_width, cropped_height))
        i_view.draw_pq_list([(cs_cropped_x, 0), (cs_cropped_x, cropped_height)], style=vstyle)
        i_view.draw_pq_list([(0, cs_cropped_y_mlab), (cropped_width, cs_cropped_y_mlab)], style=hstyle)

        # Pre-plot call to the custom renderer
        graphs_per_plot_cnt = 0
        graphs_per_plot_cnt += self.pre_visualize(operable, cs_loc_cropped, cropped_region)

        # Draw the cross sections graphs using the same axes
        graphs_per_plot_cnt += self._draw_cross_section(np_image, cs_loc, cropped_region)

        # Post-plot call to the custom renderer
        graphs_per_plot_cnt += self.post_visualize(operable, cs_loc_cropped, cropped_region)

        # draw
        for view in self.views:
            legend = graphs_per_plot_cnt > 1
            view.show(block=False, legend=legend)

        # explicitly set the y-axis range
        if self.y_range is not None:
            h_fig_record.view.axis.set_ylim(self.y_range)
            v_fig_record.view.axis.set_ylim(self.y_range)

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
