from abc import ABC, abstractmethod
import copy
import dataclasses
from typing import Callable

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.tool.log_tools as lt


class AbstractVisualizationImageProcessor(AbstractSpotAnalysisImageProcessor, ABC):
    """
    An AbstractSpotAnalysisImageProcessor that is used to generate visualizations.

    By convention subclasses are named "View*ImageProcessor" (their name starts
    with "View" and ends with "ImageProcessor"). Note that subclasses should not
    implement their own _execute() methods, but should instead implement
    num_figures, _init_figure_records(), visualize_operable(), and
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

            def _init_figure_records(self, render_control_fig):
                self.fig_record = fm.setup_figure(
                    render_control_fig,
                    rca.image(),
                    equal=False,
                    name=self.name,
                    code_tag=f"{__file__}._init_figure_records()",
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
        """
        # import here to avoid circular dependencies
        from opencsp.common.lib.cv.spot_analysis.VisualizationCoordinator import VisualizationCoordinator

        super().__init__(name)

        # validate arguments
        if isinstance(base_image_selector, str):
            acceptable_values = ["visualization", "algorithm"]
            if base_image_selector.lower() not in acceptable_values:
                lt.error_and_raise(
                    ValueError,
                    "Error in AbstractVisualizationImageProcessor(): "
                    + f"base_image_selector must be either an ImageType or one of {acceptable_values}, "
                    + "but it is '{base_image_selector}'",
                )

        # register arguments
        self.interactive = interactive
        self.base_image_selector = base_image_selector

        # internal values
        self.visualization_coordinator: VisualizationCoordinator = None
        """
        The coordinator registered with this instance through
        register_visualization_coordinator(). If None, then it is assumed that
        we should draw the visualization during the _execute() method.
        """
        self.initialized_figure_records = False
        """ True if init_figure_records() has been called, False otherwise. """
        self.pending_operables: list[SpotAnalysisOperable] = []
        """
        Operables that this instance has attempted to visualize, but has been
        blocked by the coordinator when asking is_time_to_visualize().
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
    def _init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
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

    def _get_image_for_visualizing(self, operable: SpotAnalysisOperable) -> CacheableImage:
        if self.base_image_selector is None or self.base_image_selector == ImageType.PRIMARY:
            return operable.primary_image
        elif isinstance(self.base_image_selector, str):
            if self.base_image_selector.lower() == 'visualization':
                return list(operable.visualization_images.values())[-1][0]
            elif self.base_image_selector.lower() == 'algorithm':
                return list(operable.algorithm_images.values())[-1][0]
            else:
                lt.error_and_raise(
                    RuntimeError,
                    "Error in AbstractVisualizationImageProcessor._get_image_for_visualization(): "
                    + f"unknown base_image_selector string value '{self.base_image_selector}'",
                )
        elif self.base_image_selector in [
            ImageType.REFERENCE,
            ImageType.NULL,
            ImageType.COMPARISON,
            ImageType.BACKGROUND_MASK,
        ]:
            return operable.supporting_images[self.base_image_selector]
        else:
            lt.error_and_raise(
                RuntimeError,
                "Error in AbstractVisualizationImageProcessor._get_image_for_visualization(): "
                + f"unknown base_image_selector of type {type(self.base_image_selector)}: {self.base_image_selector}",
            )

    @abstractmethod
    def _visualize_operable(
        self, operable: SpotAnalysisOperable, is_last: bool, base_image: CacheableImage
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
        base_image : CacheableImage
            The base image on which to draw the visualization. Value is
            determined by :py:attr:`base_image_selector` and retrieved with
            :py:meth:`_get_image_for_visualizing`.

        Returns
        -------
        visualizations: list[CacheableImage|rcfr.RenderControlFigureRecord]
            Visualizations from this image processor as cacheable images or as
            figure records. Empty list if there aren't any.
        """
        pass

    @abstractmethod
    def close_figures(self):
        """
        Closes all visualization windows created by this instance.
        """
        pass

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

    def init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
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
        ret = self._init_figure_records(render_control_fig)
        self.initialized_figure_records = True
        return ret

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        ret: SpotAnalysisOperable = None

        if self.has_visualization_coordinator:
            # let the visualization coordinator determine when we visualize an operable
            if self.visualization_coordinator.is_time_to_visualize(self, operable, is_last):
                # visualize the operable now
                self.pending_operables.append(operable)
                op_with_vis = self.visualization_coordinator.visualize(self, operable, is_last)
                if op_with_vis is not None:
                    ret = op_with_vis
                else:
                    ret = dataclasses.replace(operable)
            else:
                # remember this operable, to be visualized later synchronously
                # with the other visualization image processors
                ret = dataclasses.replace(operable)
                self.pending_operables.append(ret)
        else:
            # no coordinator for synchronized visualization, always visualize
            # the operable immediately
            if not self.initialized_figure_records:
                self.init_figure_records(rcf.RenderControlFigure(tile=False))
            self.pending_operables.append(operable)
            new_visualizations = self.visualize_operable(operable, is_last)

            # get the visualization images list
            visualization_images = copy.copy(ret.visualization_images)
            if self not in visualization_images:
                visualization_images[self] = []
            else:
                visualization_images[self] = copy.copy(visualization_images[self])
            visualization_images[self] += new_visualizations

            # update the return value
            ret = dataclasses.replace(operable, visualization_images=visualization_images)

        return [ret]

    def visualize_operable(self, operable: SpotAnalysisOperable, is_last: bool) -> list[CacheableImage]:
        """
        Calls _visualize_operable() and registers the visualizations as
        algorithm_images on the given operable.

        Visualizes all pending operables that either are the given operable or
        are ancestors of the given operable.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable to visualize, or the decendent of the operable(s) to
            visualize. Should either be one of self.pending_operables, or a
            decendent of one or more pending operables.
        is_last : bool
            True if this is the last operable that this method will be evaluated for.

        Returns
        -------
        list[CacheableImage]
            This processor's visualizations.
        """
        # Get the list of pending operables to visualize
        operables_to_visualize: list[SpotAnalysisOperable] = []
        if self.has_visualization_coordinator:
            operables_to_visualize, self.pending_operables = self.visualization_coordinator._select_pending_operables(
                self.pending_operables, operable
            )
        else:
            for pending_operable in copy.copy(self.pending_operables):
                if (pending_operable == operable) or (pending_operable.is_ancestor_of(operable)):
                    operables_to_visualize.append(pending_operable)
                    self.pending_operables.remove(pending_operable)

        # Visualize the matching operables
        all_vis_images: list[CacheableImage] = []
        for i, vis_operable in enumerate(operables_to_visualize):
            operable_is_last = is_last and i == len(operables_to_visualize) - 1
            base_image = self._get_image_for_visualizing(vis_operable)
            visualizations = self._visualize_operable(vis_operable, operable_is_last, base_image)

            # build the list of visualization images
            for cacheable_or_figure_rec in visualizations:
                if isinstance(cacheable_or_figure_rec, CacheableImage):
                    all_vis_images.append(cacheable_or_figure_rec)

                else:
                    # get the figure as an numpy array
                    np_image = cacheable_or_figure_rec.to_array()

                    # add the image
                    cacheable_image = CacheableImage(np_image)
                    all_vis_images.append(cacheable_image)

        return all_vis_images
