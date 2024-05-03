import copy
import enum
import os

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import AbstractVisualizationImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
import opencsp.common.lib.render.lib.PowerpointImage as pi
import opencsp.common.lib.render.PowerpointSlide as ps
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlPowerpointPresentation as rcpp
import opencsp.common.lib.render_control.RenderControlPowerpointSlide as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class PowerpointImageProcessor(AbstractSpotAnalysisImagesProcessor):
    """
    Saves the results of the spot analysis image processing pipeline as a PowerPoint deck.
    """

    def __init__(
        self,
        save_dir: str,
        save_name: str,
        overwrite=False,
        operable_title_slides_todo=False,
        processors_per_slide: list[list[AbstractSpotAnalysisImagesProcessor]] = None,
    ):
        super().__init__(self.__class__.__name__)

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

        # register the rest of the input arguments
        self.overwrite = overwrite
        self.operable_title_slides = operable_title_slides_todo
        self.processors_per_slide = processors_per_slide

        # internal values
        self.presentation = rcpp.RenderControlPowerpointPresentation()
        self.slide_control = rcps.RenderControlPowerpointSlide()

    def _add_operable_title_slide(self, operable: SpotAnalysisOperable):
        # check if we should have title slides
        if not self.operable_title_slides:
            return

        # create a title slide and add it to the presentation
        slide = ps.PowerpointSlide.template_title(operable.best_primary_pathnameext, "", self.slide_control)
        self.presentation.add_slide(slide)

    def _get_processors_in_order(self, operable: SpotAnalysisOperable) -> list[AbstractSpotAnalysisImagesProcessor]:
        previous_operables = operable.previous_operables[0]
        previous_processor: AbstractSpotAnalysisImagesProcessor = operable.previous_operables[1]

        if previous_operables is None or previous_processor is None:
            return []

        else:
            ret = self._get_processors_in_order(previous_operables[0])
            ret.append(previous_processor)
            return ret

    def _fill_processor_images_dict(
        self,
        operable: SpotAnalysisOperable,
        processor_images_dict: dict[AbstractSpotAnalysisImagesProcessor, set[CacheableImage]],
        include_processors: list[AbstractSpotAnalysisImagesProcessor],
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
        previous_processor: AbstractSpotAnalysisImagesProcessor = operable.previous_operables[1]
        """ The processor that produced this current operable """
        if previous_operables is None or previous_processor is None:
            return
        if True:  # one operable per processor, to avoid a spanning tree for many-to-one processors
            previous_operables = previous_operables[:1]

        # make sure we have a list to add to
        if previous_processor not in processor_images_dict:
            processor_images_dict[previous_processor] = set()

        # Add the algorithm images for this step
        found_algo_images = False
        if previous_processor in operable.algorithm_images:
            for image in operable.algorithm_images[previous_processor]:
                processor_images_dict[previous_processor].add(image)
                found_algo_images = True

        # If we didn't find any images, then add the primary image for this step
        if not found_algo_images:
            processor_images_dict[previous_processor].add(operable.primary_image)

        # add the supporting images for this step
        # TODO

        ######################################
        # Get images for the operable's parent
        ######################################

        # repeat for the previous step in the operable chain
        for previous_operable in previous_operables:
            self._fill_processor_images_dict(
                previous_operable, processor_images_dict, include_processors, recursion_index=recursion_index + 1
            )

        ########################################
        # Special Case: visualization processors
        ########################################

        # add missing visualization images
        if recursion_index == 0:
            visualization_processors = filter(
                lambda proc: isinstance(proc, AbstractVisualizationImageProcessor), processor_images_dict.keys()
            )
            for processor in visualization_processors:
                if processor not in operable.algorithm_images:
                    continue
                processor_images_dict[processor] = operable.algorithm_images[processor]

        ###############################
        # Limit to the given processors
        ###############################

        # remove any processors that don't match the limited types
        for processor in list(processor_images_dict.keys()):
            if processor not in include_processors:
                del processor_images_dict[processor]

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        # add a title slide for the operable, as necessary
        self._add_operable_title_slide(operable)

        # build the slides for this operable
        processor_images_dict_template: dict[AbstractSpotAnalysisImagesProcessor, set[CacheableImage]] = {}

        # Initialize the processor_images dict. Initializing it in this way will
        # will preserve processor ordering.
        all_processors = self._get_processors_in_order(operable)
        for processor in all_processors:
            processor_images_dict_template[processor] = set()

        # add one slide for each set of processors
        for processor_set in self.processors_per_slide:
            processor_images_dict = copy.copy(processor_images_dict_template)

            # get the images per processor
            self._fill_processor_images_dict(operable, processor_images_dict, processor_set)

            # get the images
            images_list: list[tuple[AbstractSpotAnalysisImagesProcessor, CacheableImage]] = []
            for processor in processor_images_dict:
                for image in processor_images_dict[processor]:
                    images_list.append((processor, image))

            # prepare the slide
            n_rows, n_cols = rcf.RenderControlFigure.num_tiles_4x3aspect(len(images_list))
            slide = ps.PowerpointSlide.template_content_grid(n_rows, n_cols, self.slide_control)

            # add information to the slide
            slide.set_title(operable.best_primary_nameext)
            for processor, image in images_list:
                slide.add_image(pi.PowerpointImage(image.nparray, caption=processor.name))

            # add the slide to the presentation
            slide.save_and_bake()
            self.presentation.add_slide(slide)

        # save the powerpoint presentation
        if is_last:
            self.presentation.save(self.dest_path_name_ext, self.overwrite)

        return [operable]
