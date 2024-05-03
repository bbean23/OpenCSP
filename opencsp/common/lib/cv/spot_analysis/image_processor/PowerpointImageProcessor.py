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


class SlideDivision(enum.Enum):
    PER_OPERABLE = 0
    PER_IMAGE = 1
    PER_PROCESSOR = 2


class PowerpointImageProcessor(AbstractSpotAnalysisImagesProcessor):
    """
    Saves the results of the spot analysis image processing pipeline as a PowerPoint deck.

    The saved results can include any of:

        - the primary image for the current operable
        - the results from the operables at each of the previous processors
        - the supporting images associated with each operable
        - the algorithm images associated with each operable

    The powerpoint deck can formatted with:

        - one slide per operable [TODO]
        - one slide per N images [TODO]
        - one slide per processor

    Other options:

        - Insert title slides between operable
          (for each current operable add a title slide)
        - Only include a list of processor types
        - Only include the first operable for processors that have many-to-one or
          many-to-many operable result
    """

    def __init__(
        self,
        save_dir: str,
        save_name: str,
        overwrite=False,
        all_preceeding_processors=True,
        include_primary_image=True,
        include_supporting_images_todo=False,
        include_algorithm_images=False,
        slide_division_todo=SlideDivision.PER_PROCESSOR,
        images_per_division=1,
        operable_title_slides=False,
        include_processor_types: list[type[AbstractSpotAnalysisImagesProcessor]] = None,
        one_operable_per_processor_todo=True,
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
        if not include_primary_image and not include_supporting_images_todo and not include_algorithm_images:
            lt.error_and_raise(
                ValueError,
                "Error in PowerpointImageProcessor.__init__(): "
                + f"Must include at least one type of image from the spot analysis operables.",
            )

        # register the rest of the input arguments
        self.overwrite = overwrite
        self.all_preceeding_processors = all_preceeding_processors
        self.include_primary_image = include_primary_image
        self.include_supporting_images_todo = include_supporting_images_todo
        self.include_algorithm_images = include_algorithm_images
        self.slide_division = slide_division_todo
        self.images_per_division = images_per_division
        self.operable_title_slides = operable_title_slides
        self.include_processor_types = include_processor_types
        self.one_operable_per_processor = one_operable_per_processor_todo

        # check for currently unsupported options
        if self.slide_division == SlideDivision.PER_IMAGE or self.slide_division == SlideDivision.PER_OPERABLE:
            lt.error_and_raise(
                NotImplementedError, f"The value {self.slide_division} for slide_division is not currently supported"
            )
        if not self.one_operable_per_processor:
            lt.error_and_raise(
                NotImplementedError,
                f"The value {self.one_operable_per_processor} for one_operable_per_processor is not currently supported",
            )

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

    def _get_per_processor_images(
        self,
        operable: SpotAnalysisOperable,
        processor_images: dict[AbstractSpotAnalysisImagesProcessor, list[CacheableImage]],
        recursion_index=0
    ):
        """
        Follows the chain of operables backwards to get a list of all images
        that affected the given operable.
        """
        # get the values for one step back
        previous_operables = operable.previous_operables[0]
        previous_processor: AbstractSpotAnalysisImagesProcessor = operable.previous_operables[1]
        if previous_operables is None or previous_processor is None:
            return
        if self.one_operable_per_processor:
            previous_operables = previous_operables[:1]

        # make sure we have a list to add to
        if previous_processor not in processor_images:
            processor_images[previous_processor] = []

        # add the primary image for this step
        if self.include_primary_image:
            processor_images[previous_processor].append(operable.primary_image)

        # add the supporting images for this step
        # TODO

        # add the algorithm images for this step
        if self.include_algorithm_images:
            if previous_processor in operable.algorithm_images:
                for image in operable.algorithm_images[previous_processor]:
                    processor_images[previous_processor].append(image)

        # go back another step
        for previous_operable in previous_operables:
            self._get_per_processor_images(previous_operable, processor_images, recursion_index=recursion_index+1)

        # add missing visualization images
        if recursion_index == 0:
            visualization_processors = filter(lambda proc: isinstance(
                proc, AbstractVisualizationImageProcessor), processor_images.keys())
            for processor in visualization_processors:
                if not self.include_algorithm_images:
                    continue
                if processor not in operable.algorithm_images:
                    continue
                for algorithm_image in operable.algorithm_images[processor]:
                    if algorithm_image not in processor_images[processor]:
                        processor_images[processor].append(algorithm_image)

        # remove any processors that don't match the limited types
        if self.include_processor_types is not None:
            for processor in list(processor_images.keys()):
                found = False
                for processor_type in self.include_processor_types:
                    if isinstance(processor, processor_type):
                        found = True
                        break
                if not found:
                    del processor_images[processor]

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        # add a title slide for the operable, as necessary
        self._add_operable_title_slide(operable)

        # build the slides for this operable
        if self.slide_division == SlideDivision.PER_IMAGE:
            raise NotImplementedError

        elif self.slide_division == SlideDivision.PER_OPERABLE:
            raise NotImplementedError

        elif self.slide_division == SlideDivision.PER_PROCESSOR:
            processor_images: dict[AbstractSpotAnalysisImagesProcessor, list[CacheableImage]] = {}

            # Initialize the processor_images dict. Initializing it in this will
            # will preserve processor ordering.
            all_processors = self._get_processors_in_order(operable)
            for processor in all_processors:
                processor_images[processor] = []

            # get the images per processor
            self._get_per_processor_images(operable, processor_images)
            processors_with_images = filter(lambda proc: len(processor_images[proc]) > 0, processor_images.keys())
            processor_images = {processor: processor_images[processor] for processor in processors_with_images}

            # add one slide per processor
            for processor in processor_images:
                cacheable_images = processor_images[processor]
                n_rows, n_cols = rcf.RenderControlFigure.num_tiles_4x3aspect(len(cacheable_images))

                slide = ps.PowerpointSlide.template_content_grid(n_rows, n_cols, self.slide_control)
                slide.set_title(processor.name)
                for cacheable_image in cacheable_images:
                    slide.add_image(pi.PowerpointImage(cacheable_image.nparray))
                slide.save_and_bake()

                self.presentation.add_slide(slide)

        # save the powerpoint presentation
        if is_last:
            self.presentation.save(self.dest_path_name_ext, self.overwrite)

        return [operable]
