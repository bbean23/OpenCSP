from abc import ABC, abstractmethod
import copy
import dataclasses
from typing import Callable
import weakref

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class AbstractVisualizationImageProcessor(AbstractSpotAnalysisImageProcessor, ABC):
    """
    An AbstractSpotAnalysisImageProcessor that is used to generate visualizations.

    By convention subclasses are named "View*ImageProcessor" (their name starts
    with "View" and ends with "ImageProcessor"). Note that subclasses should not
    implement their own _execute() methods, but should instead implement
    num_figures, init_figure_records(), visualize_operable(), and
    close_figures().

    The visualizations that these processors create can be used either for
    debugging or monitoring, depending on the value of the "interactive"
    initialization parameter.

    VisualizationCoordinator
    ------------------------
    Certain elements of the visualization are handled by the
    VisualizationCoordinator, including at least:

        - tiled layout of visualization windows
        - user interaction that is common to all visualization windows

    The life cycle for this class is::

        - __init__()
        - register_visualization_coordinator()*
        - num_figures()*
        - init_figure_records()*
        - process()
        -     _execute()
        -     visualize_operable()*
        - close_figures()*

    In the above list, one star "*" indicates that this method is called by the
    coordinator.

    Examples
    --------
    An example class that simply renders operables as an image might be implemented as::

        class ViewSimpleImageProcessor(AbstractVisualizationImageProcessor):
            def __init__(self, name, interactive):
                super().__init__(name, interactive)

                self.figure_rec: RenderControlFigureRecord = None

            @property
            def num_figures(self):
                return 1

            def init_figure_records(self, render_control_fig):
                self.fig_record = fm.setup_figure(
                    render_control_fig,
                    rca.image(),
                    equal=False,
                    name=self.name,
                    code_tag=f"{__file__}.init_figure_records()",
                )
                return [self.fig_record]

            def visualize_operable(self, operable, is_last):
                image = operable.primary_image.nparray
                self.fig_record.view.imshow(image)
                return [self.fig_record]

            def close_figures(self):
                with exception_tools.ignored(Exception):
                    self.fig_record.close()
                self.fig_record = None
    """

    def __init__(
        self,
        interactive: bool | Callable[[SpotAnalysisOperable], bool],
        base_image_selector: str | ImageType = None,
        name: str = None,
        accepts_external_figure_record: bool = False,
    ):
        """
        Parameters
        ----------
        interactive : bool | Callable[[SpotAnalysisOperable], bool], optional
            If True then the spot analysis pipeline is paused until the user presses the "enter" key, by default False
        base_image_selector : ImageType, optional
            Which image to draw the visualization on top of. The latest
            available of the given type is used. Can also be one of None,
            'Visualization', or 'Algorithm'. Default is the latest primary
            image.
        name : str
            Passed through to AbstractSpotAnalysisImageProcessor.__init__()
        accepts_external_figure_record : bool
            True if the visualization has been designed to allow
            :py:class:`RenderControlFigureRecord` as the base upon which to draw.
            False to limit base_images in :py:meth:`visualize_operable` to just
            :py:class:`CacheableImage`. Default is False.
        """
        # import here to avoid circular dependencies
        from opencsp.common.lib.cv.spot_analysis.VisualizationCoordinator import VisualizationCoordinator

        super().__init__(name)

        # validate arguments
        if isinstance(base_image_selector, str):
            acceptable_values = ["visualization", "algorithm"]
            if base_image_selector.lower() == "algorithm":
                base_image_selector = ImageType.ALGORITHM
            elif base_image_selector.lower() == "visualization":
                base_image_selector = ImageType.VISUALIZATION
            else:
                lt.error_and_raise(
                    ValueError,
                    "Error in AbstractVisualizationImageProcessor(): "
                    + f"base_image_selector must be either an ImageType or one of {acceptable_values}, "
                    + f"but it is '{base_image_selector}'",
                )

        # register arguments
        self.interactive = interactive
        self.base_image_selector: ImageType = base_image_selector
        """
        Determines the image returned from
        :py:meth:`_get_image_for_visualizing`. Typically this will be one of
        None/ImageType.PRIMARY or 'visualization'.
        """
        self.accepts_external_figure_record = accepts_external_figure_record
        """
        If True, then the :py:class:`VisualizationCoordinator` will attempt to use the
        figure record of a previous image processor instead of the corresponding cacheable
        image. For base_image_selector = 'visualization' this will be the previous
        AbstractVisualizationImageProcessor.
        """

        # internal values
        self.visualization_coordinator: VisualizationCoordinator = None
        """
        The coordinator registered with this instance through
        register_visualization_coordinator(). If None, then it is assumed that
        we should draw the visualization during the _execute() method.
        """
        self.initialized_figure_records = False
        """ True if init_figure_records() has been called, False otherwise. """
        self.figure_records: list[rcfr.RenderControlFigureRecord] = []
        """
        The records returned by :py:meth:`init_figure_records` in :py:meth:`_init_figure_records`.
        """

    @property
    @abstractmethod
    def num_figures(self) -> int:
        """
        How many figure windows this instance intends to create. Must be
        available at all times after this instance has been initialized.
        """
        pass

    @abstractmethod
    def init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
        """
        Initializes the figure windows (via figure_management.setup_figure*) for
        this instance and returns the list of initialized figures. The length of
        this list ideally should match what was previously returned for
        num_figures.

        Parameters
        ----------
        render_control_fig : rcf.RenderControlFigure
            The render controller to use during figure setup.

        Returns
        -------
        figures: list[rcfr.RenderControlFigureRecord]
            The list of newly created figure windows.
        """
        pass

    def _get_figure_record_for_visualizing(
        self, operable: SpotAnalysisOperable
    ) -> rcfr.RenderControlFigureRecord | None:
        """
        Chooses the figure record from the most recent AbstractVisualizationImageProcessor
        as the base upon which to render the current visualization.

        If the most recent AbstractVisualizationImageProcessor is not a good target for drawing on,
        such as when there is no previous AbstractVisualizationImageProcessor in the operable's
        history, then this method returns None instead.
        """
        # find the most recent visualization processor in the history, if there is one
        previous_operables, previous_image_processor = operable.previous_operables

        # Check if there are any previous visualization processors
        if previous_operables and isinstance(previous_image_processor, AbstractVisualizationImageProcessor):
            # Get the figure record from the previous image processor
            figure_records = previous_image_processor.figure_records

            # Return the figure record if it exists, otherwise return None
            if figure_records is not None and len(figure_records) > 0:
                return figure_records[0]
            else:
                return None
        else:
            # There are no previous visualization processors
            return None

    def _get_image_for_visualizing(self, operable: SpotAnalysisOperable) -> CacheableImage:
        """
        Chooses one of the operable's images to use to draw visualizations on
        top of based on self.:py:attr:`base_image_selector`. The returned value
        will be passed through to :py:meth:`visualize_operable` as the
        base_image.
        """
        err_src = "Error in AbstractVisualizationImageProcessor._get_image_for_visualization(): "
        if self.base_image_selector is None or self.base_image_selector == ImageType.PRIMARY:
            return operable.primary_image
        elif self.base_image_selector == ImageType.VISUALIZATION:
            try:
                return list(operable.visualization_images.values())[-1][0]
            except IndexError:
                lt.error_and_raise(
                    IndexError,
                    err_src
                    + "failed to get the latest visualization image! "
                    + "Maybe there isn't a previous visualization image processor?",
                )
        elif self.base_image_selector == ImageType.ALGORITHM:
            try:
                return list(operable.algorithm_images.values())[-1][0]
            except IndexError:
                lt.error_and_raise(
                    IndexError,
                    err_src
                    + "failed to get the latest algorithm image! "
                    + "Maybe there isn't a previous image processor, or "
                    + "maybe it didn't produce any algorithm images?",
                )
        elif self.base_image_selector in [
            ImageType.REFERENCE,
            ImageType.NULL,
            ImageType.COMPARISON,
            ImageType.BACKGROUND_MASK,
        ]:
            try:
                return operable.supporting_images[self.base_image_selector]
            except IndexError:
                lt.error_and_raise(
                    IndexError, err_src + f"the operable does not have a {self.base_image_selector} image!"
                )
        else:
            lt.error_and_raise(
                RuntimeError,
                err_src
                + f"unknown base_image_selector of type {type(self.base_image_selector)}: "
                + f"{self.base_image_selector}",
            )

    def _get_base_for_visualizing(
        self, operable: SpotAnalysisOperable
    ) -> CacheableImage | rcfr.RenderControlFigureRecord:
        # attempt to use a RenderControlFigureRecord for visualizing
        if self.base_image_selector is not None:
            if self.base_image_selector == ImageType.ALGORITHM:
                if self.accepts_external_figure_record:
                    fig_record = self._get_figure_record_for_visualizing(operable)
                    if fig_record is not None:
                        return fig_record

        # fall back on using an image for visualizing
        return self._get_image_for_visualizing(operable)

    def prepare_for_visualization(
        self, operable: SpotAnalysisOperable, is_last: bool, base_image: CacheableImage | rcfr.RenderControlFigureRecord
    ):
        """
        Called just prior to :py:meth:`visualize_operable`. The default
        implementation of this method simply clears the figure records
        returned in :py:meth:`init_figure_records`.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable object being processed.
        is_last : bool
            A flag indicating whether this is the last operable to be processed.
        base_image : CacheableImage | rcfr.RenderControlFigureRecord
            The base image to be used for visualization.
        """
        if len(self.figure_records) == 0:
            if not self.initialized_figure_records:
                lt.error_and_raise(
                    RuntimeError,
                    "Programmer error in AbstractVisualizationImageProcessor.prepare_for_visualization(): "
                    + "expected _init_figure_records() to have been called before this method, "
                    + f"but {self.initialized_figure_records=}!",
                )
            else:
                lt.error_and_raise(
                    RuntimeError,
                    "Programmer error in AbstractVisualizationImageProcessor.prepare_for_visualization(): "
                    + "expected _init_figure_records() to have been called before this method, "
                    + f"but {len(self.figure_records)=}!",
                )

        # clear the previously plotted results
        for fig_record in self.figure_records:
            fig_record.clear()

    @abstractmethod
    def visualize_operable(
        self, operable: SpotAnalysisOperable, is_last: bool, base_image: CacheableImage | rcfr.RenderControlFigureRecord
    ) -> list[CacheableImage | rcfr.RenderControlFigureRecord]:
        """
        Updates the figures for this instance with the data from the given operable.

        The implementing visualization image processor has the option of
        returning visualizations as cacheable images, figure records, or a mix
        of both.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable to draw the visualization for.
        is_last : bool
            True if this is the last operable to be drawn by this processor.
        base_image : CacheableImage | rcfr.RenderControlFigureRecord
            The base image on which to draw the visualization. Value is
            determined by :py:attr:`base_image_selector` and retrieved with
            :py:meth:`_get_image_for_visualizing`.

            If self.:py:attr:`accepts_external_figure_record` is False, then
            this will never be a figure record and will always be a
            cacheable image.

        Returns
        -------
        visualizations: list[CacheableImage|rcfr.RenderControlFigureRecord]
            Visualizations from this image processor as cacheable images or as
            figure records. Empty list if there aren't any.
        """
        pass

    def show_visualization(self, figure_records: list[rcfr.RenderControlFigureRecord]):
        """
        Call ".view.show()" on all the figure_records.
        Called immediately after :py:meth:`visualize_operable`.

        A common reason to override this method is to call show with the parameter legend=True.
        The default implementation calls show with the parameter block=False.

        Parameters
        ----------
        figure_records : list[rcfr.RenderControlFigureRecord]
            The figure records returned from :py:meth:`visualize_operable`.
        """
        for fig_record in figure_records:
            fig_record.view.show(block=False)

    def close_figures(self):
        """
        Closes all visualization windows created by this instance.

        The default implementation closes the view of all the :py:attr:`figure_records`
        and then clears the figure_records list.
        """
        for fig_record in self.figure_records:
            with et.ignored(Exception):
                fig_record.view.close()

        self.figure_records.clear()

    @property
    def has_visualization_coordinator(self) -> bool:
        """
        True if this instance is registered with a visualization coordinator.
        False otherwise.
        """
        return self.visualization_coordinator is not None

    def register_visualization_coordinator(self, coordinator):
        """
        Registers the given coordinator with this visualization processor instance.

        Parameters
        ----------
        coordinator : VisualizationCoordinator
            The coordinator that is registering against this instance.
        """
        # Note: no type hint for coordinator to avoid a circular import dependency
        self.visualization_coordinator = coordinator

    def _init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
        """
        Called by the registered coordinator to create any necessary
        visualization windows. If there is no registered coordinator by the time
        _execute is called, then this method will be evaluated by this instance
        internally.

        Parameters
        ----------
        render_control_fig : rcf.RenderControlFigure
            The controller to use with figure_management.setup_figure*

        Returns
        -------
        list[rcfr.RenderControlFigureRecord]
            The list of newly created visualization windows.
        """
        ret = self.init_figure_records(render_control_fig)
        self.initialized_figure_records = True
        self.figure_records = copy.copy(ret)
        return ret

    @staticmethod
    def default_render_control_figure_for_operable(operable: SpotAnalysisOperable):
        """
        Create a default render control figure for the given operable.

        This static method generates a render control figure based on the dimensions
        and number of channels of the primary image associated with the provided
        `SpotAnalysisOperable`. This default figure does not use tiling and does not
        include the typical matplotlib whitespace padding.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            An instance of `SpotAnalysisOperable` containing the primary image
            for which the render control figure is to be created.

        Returns
        -------
        rcf.RenderControlFigure
            A configured render control figure that can be used for visualizing
            the operable's primary image.

        Notes
        -----
        - The figure size is determined based on the pixel dimensions of the
          primary image.
        """
        # ChatGPT 4o-mini assisted with generating this docstring, reviewed by a human
        (height_px, width_px), nchannel = it.dims_and_nchannels(operable.primary_image.nparray)
        figsize = rcf.RenderControlFigure.pixel_resolution_inches(width_px, height_px)
        figure_control = rcf.RenderControlFigure(tile=False, figsize=figsize, grid=False, draw_whitespace_padding=False)
        return figure_control

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        """
        Execute the visualization process for the given operable.

        This method performs the visualization of the provided `SpotAnalysisOperable`.
        It checks for the presence of a visualization coordinator and either visualizes
        the operable through the coordinator or directly if no coordinator is available.
        The method also manages the visualization images associated with the operable.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable instance to be visualized. It may contain visualization
            images that will be updated during the execution.
        is_last : bool
            A flag indicating whether this is the last operable to be processed.
            This may affect how the visualization is handled.

        Returns
        -------
        list[SpotAnalysisOperable]
            A list containing the updated `SpotAnalysisOperable` instance with
            the visualization images included.

        Notes
        -----
        - If a visualization coordinator is present, the operable is visualized
          through it. If not, the visualization is performed immediately.
        - The method initializes figure records if they have not been set up
          previously.
        - The visualization images are copied and updated to ensure that the
          original operable remains unchanged.
        """
        # ChatGPT 4o-mini assisted with generating this docstring, reviewed by a human
        ret: SpotAnalysisOperable = None

        if self.has_visualization_coordinator:
            # Visualize the operable and block (if interactive).
            op_with_vis = self.visualization_coordinator.visualize(self, operable, is_last)
            if op_with_vis is not None:
                ret = op_with_vis
            else:
                ret = dataclasses.replace(operable)
        else:
            # no coordinator for synchronized visualization, always visualize
            # the operable immediately
            if not self.initialized_figure_records:
                # Create the figure to plot to
                render_control = self.default_render_control_figure_for_operable(operable)
                self._init_figure_records(render_control)
            new_visualizations = self._visualize_operable(operable, is_last)

            # get the visualization images list
            visualization_images = copy.copy(operable.visualization_images)
            if self not in visualization_images:
                visualization_images[self] = []
            else:
                visualization_images[self] = copy.copy(visualization_images[self])
            visualization_images[self] += new_visualizations

            # if interactive, then wait for any button to be pressed
            # TODO check for enter key to be pressed, specifically
            if self.interactive:
                self.figure_records[0].figure.waitforbuttonpress(60 * 60)

            # update the return value
            ret = dataclasses.replace(operable, visualization_images=visualization_images)

        return [ret]

    def _visualize_operable(self, operable: SpotAnalysisOperable, is_last: bool) -> list[CacheableImage]:
        """
        Calls :py:meth:`visualize_operable` and collects the visualziation images.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable to visualize.
        is_last : bool
            True if this is the last operable that this method will be evaluated for.

        Returns
        -------
        list[CacheableImage]
            This processor's visualizations.
        """
        # get the image to render onto
        base_image = self._get_base_for_visualizing(operable)

        # clear the previous visualization
        if isinstance(base_image, rcfr.RenderControlFigureRecord):
            # don't clear a record that is being used as the base
            pass
        else:
            self.prepare_for_visualization(operable, is_last, base_image)

        # visualize the operable
        visualizations = self.visualize_operable(operable, is_last, base_image)

        # show the figure records
        visualizations_as_records = [
            fig_record for fig_record in visualizations if isinstance(fig_record, rcfr.RenderControlFigureRecord)
        ]
        if len(visualizations_as_records) > 0:
            self.show_visualization(visualizations_as_records)
        else:
            self.show_visualization(self.figure_records)

        # verify the returned type
        if not isinstance(visualizations, list):
            raise TypeError(
                f"Error in {self.name}.visualize_operable(): "
                + "should have returned a list of visualizations but instead returned a "
                + str(type(visualizations))
            )
        for i, visualization in enumerate(visualizations):
            if not (
                isinstance(visualization, CacheableImage) or isinstance(visualization, rcfr.RenderControlFigureRecord)
            ):
                raise TypeError(
                    f"Error in {self.name}.visualize_operable(): "
                    + "should have returned a list of CacheableImage and RenderControlFigureRecord, but "
                    + f"visualization {i} is a "
                    + str(type(visualizations))
                )

        # build the list of visualization images
        all_vis_images: list[CacheableImage] = []
        for cacheable_or_figure_rec in visualizations:
            if isinstance(cacheable_or_figure_rec, CacheableImage):
                all_vis_images.append(cacheable_or_figure_rec)

            else:
                # get the figure as an numpy array, using the standard 8 inches figure height
                np_image = cacheable_or_figure_rec.to_array(8.0)

                # add the image
                cacheable_image = CacheableImage(np_image)
                all_vis_images.append(cacheable_image)

        return all_vis_images
