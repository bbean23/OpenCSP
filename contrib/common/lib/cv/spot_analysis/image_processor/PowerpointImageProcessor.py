import copy
import enum
import os

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import AbstractVisualizationImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.render.lib.PowerpointImage as pi
import opencsp.common.lib.render.PowerpointSlide as ps
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlPowerpointPresentation as rcpp
import opencsp.common.lib.render_control.RenderControlPowerpointSlide as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class PowerpointImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Saves the results of the spot analysis image processing pipeline as a PowerPoint deck.
    """

    def __init__(
        self,
        save_dir: str,
        save_name: str,
        overwrite=False,
        operable_title_slides_todo=False,
        processors_per_slide: list[list[AbstractSpotAnalysisImageProcessor]] | None = None,
    ):
        super().__init__()

        # register some of the input arguments
        self.save_dir = save_dir
        self.save_name = save_name if save_name.lower().endswith(".pptx") else save_name + ".pptx"
        self.dest_path_name_ext = ft.norm_path(os.path.join(self.save_dir, self.save_name))

        # validate input
        if not ft.directory_exists(self.save_dir):
            lt.error_and_raise(
                FileNotFoundError,
                "Error in PowerpointImageProcessor.__init__(): "
                + f"destination directory \"{self.save_dir}\" does not exist!",
            )
        if ft.file_exists(self.dest_path_name_ext):
            if not overwrite:
                lt.error_and_raise(
                    FileExistsError,
                    "Error in PowerpointImageProcessor.__init__(): "
                    + f"destination file \"{self.dest_path_name_ext}\" already exists!",
                )
        if processors_per_slide is not None:
            if not hasattr(processors_per_slide, "__iter__"):
                lt.error_and_raise(
                    ValueError,
                    "Error in PowerpointImageProcessor.__init__(): "
                    + f"\"processors_per_slide\" is a {type(processors_per_slide)}, but should be a list of lists",
                )
            else:
                for i, processor_set in enumerate(processors_per_slide):
                    if not hasattr(processor_set, "__iter__") or isinstance(
                        processor_set, AbstractSpotAnalysisImageProcessor
                    ):
                        lt.error_and_raise(
                            ValueError,
                            "Error in PowerpointImageProcessor.__init__(): "
                            + f"\"processors_per_slide[{i}]\" is a {type(processor_set)}, but should be a lists of image processors!",
                        )
                    for j, processor in enumerate(processor_set):
                        if not isinstance(processor, AbstractSpotAnalysisImageProcessor):
                            lt.error_and_raise(
                                ValueError,
                                "Error in PowerpointImageProcessor.__init__(): "
                                + f"\"processors_per_slide[{i}][{j}]\" is a {type(processor)}, but should be an AbstractSpotAnalysisImageProcessor!",
                            )

        # register the rest of the input arguments
        self.overwrite = overwrite
        self.operable_title_slides = operable_title_slides_todo
        self.processors_per_slide = processors_per_slide

        # internal values
        self.is_first_operable = True
        self.presentation = rcpp.RenderControlPowerpointPresentation()
        self.slide_control = rcps.RenderControlPowerpointSlide()

    def _add_operable_title_slide(self, operable: SpotAnalysisOperable):
        # check if we should have title slides
        if not self.operable_title_slides:
            return

        # create a title slide and add it to the presentation
        slide = ps.PowerpointSlide.template_title(operable.best_primary_pathnameext, "", self.slide_control)
        self.presentation.add_slide(slide)

    def _get_processors_in_order(self, operable: SpotAnalysisOperable) -> list[AbstractSpotAnalysisImageProcessor]:
        previous_operables = operable.previous_operables[0]
        previous_processor: AbstractSpotAnalysisImageProcessor = operable.previous_operables[1]

        if previous_operables is None or previous_processor is None:
            return []

        else:
            ret = self._get_processors_in_order(previous_operables[0])
            ret.append(previous_processor)
            return ret

    def _populate_processor_images_dict(
        self,
        operable: SpotAnalysisOperable,
        processor_images_dict: dict[AbstractSpotAnalysisImageProcessor, list[CacheableImage]],
        include_processors: list[AbstractSpotAnalysisImageProcessor],
        recursion_index=0,
    ):
        """
        Follows the chain of operables backwards to get a list of all images
        that affected the given operable.
        """

        ############################################
        # Get images related to the current operable
        ############################################

        # get the values for one step back
        previous_operables = operable.previous_operables[0]
        """ The operables that preceeded this current operable """
        previous_processor: AbstractSpotAnalysisImageProcessor = operable.previous_operables[1]
        """ The processor that produced this current operable """
        if previous_operables is None or previous_processor is None:
            return
        if True:  # one operable per processor, to avoid a spanning tree for many-to-one processors
            previous_operables = previous_operables[:1]

        # make sure we have a list to add to
        if previous_processor not in processor_images_dict:
            processor_images_dict[previous_processor] = []

        # Add the visualization images for this step
        found_algo_images = False
        if previous_processor in operable.visualization_images:
            for image in operable.visualization_images[previous_processor]:
                processor_images_dict[previous_processor].append(image)
                found_algo_images = True

        # If we didn't find any images, then add the primary image for this step
        if not found_algo_images:
            processor_images_dict[previous_processor].append(operable.primary_image)

        # add the supporting images for this step
        # TODO

        ######################################
        # Get images for the operable's parent
        ######################################

        # repeat for the previous step in the operable chain
        for previous_operable in previous_operables:
            self._populate_processor_images_dict(
                previous_operable, processor_images_dict, include_processors, recursion_index=recursion_index + 1
            )

        #################################################################
        # Special Case: processors that have special visualization images
        #################################################################

        # add missing visualization images
        if recursion_index == 0:
            vis_processors = filter(
                lambda proc: isinstance(proc, AbstractVisualizationImageProcessor), processor_images_dict.keys()
            )
            for processor in vis_processors:
                processor_images_dict[processor] = operable.visualization_images[processor]

        ######################################################
        # Get images that explain how determinations were made
        ######################################################

        if self.is_first_operable:
            for processor in operable.algorithm_images:
                algorithm_images = operable.algorithm_images[processor]
                if len(algorithm_images) == 0:
                    continue

                if processor not in processor_images_dict:
                    processor_images_dict[processor] = []
                processor_images_dict[processor] += algorithm_images

        ###############################
        # Limit to the given processors
        ###############################

        # remove any processors that don't match the limited types
        for processor in list(processor_images_dict.keys()):
            if processor not in include_processors:
                del processor_images_dict[processor]

        ###############################
        # Remove duplicate images
        ###############################

        for processor in processor_images_dict:
            keep_images: list[CacheableImage] = []
            for cacheable in processor_images_dict[processor]:
                if cacheable not in keep_images:
                    keep_images.append(cacheable)
            processor_images_dict[processor] = keep_images

    def save(self):
        """Saves this presentation out to disk"""
        self.presentation.save(self.dest_path_name_ext, self.overwrite)

    def fetch_input_operable(self):
        """Get the next operable. If an unexpected exception gets thrown, then
        panic save the presentation as it exists up to this point."""
        try:
            return super().fetch_input_operable()
        except Exception as ex:
            if not isinstance(ex, StopIteration):
                self.save()
            raise

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        # add a title slide for the operable, as necessary
        self._add_operable_title_slide(operable)

        # build the slides for this operable
        processor_images_dict_template: dict[AbstractSpotAnalysisImageProcessor, list[CacheableImage]] = {}

        # Initialize the processor_images dict. Initializing it in this way will
        # will preserve processor ordering when the dictionary is filled.
        all_processors = self._get_processors_in_order(operable)
        for processor in all_processors:
            processor_images_dict_template[processor] = []

        # check if this should be for all processors
        processors_per_slide = self.processors_per_slide
        if processors_per_slide is None:
            processors_per_slide = [[processor] for processor in all_processors]

        # add one slide for each set of processors
        for processor_set in processors_per_slide:
            processor_images_dict = copy.copy(processor_images_dict_template)

            # get the images per processor
            self._populate_processor_images_dict(operable, processor_images_dict, processor_set)

            # get the images
            images_list: list[tuple[AbstractSpotAnalysisImageProcessor, CacheableImage]] = []
            for processor in processor_images_dict:
                for image in processor_images_dict[processor]:
                    images_list.append((processor, image))

            # prepare the slide
            n_rows, n_cols = rcf.RenderControlFigure.num_tiles_4x3aspect(len(images_list))
            slide = ps.PowerpointSlide.template_content_grid(n_rows, n_cols, self.slide_control)

            # add information to the slide
            slide.set_title(operable.best_primary_nameext)
            for processor, image in images_list:
                if processor.name == "HotspotImageProcessor":
                    pass
                slide.add_image(pi.PowerpointImage(image.nparray, caption=processor.name))

            # add the slide to the presentation
            slide.save_and_bake()
            self.presentation.add_slide(slide)

        # save the powerpoint presentation
        if is_last:
            self.save()

        self.is_first_operable = False
        return [operable]
