import copy
import dataclasses
import multiprocessing
import multiprocessing.pool
import os
import pickle

import cv2 as cv
import matplotlib.patches
import numpy as np
import numpy.typing as npt
from PIL import Image
import scipy.ndimage
import scipy.signal

from opencsp.common.lib.cv.CacheableImage import CacheableImage
import opencsp.common.lib.cv.image_reshapers as ir
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class InpaintImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Fills in shadows in images caused by known patterns. This uses a mask to
    determine the valid/invalid pixels, and fills in the invalid pixels from the
    neighboring valid pixels.

    In order to not propogate noise from the edges of the mask, it is suggested
    that this image processor be evaluated after a blurring tool such as
    ConvolutionImageProcessor.
    """

    def __init__(self, mask_image_path_name_ext: str, cache_dir: str = None):
        super().__init__()

        # validate inputs
        mask_image_path_name_ext = ft.norm_path(mask_image_path_name_ext)
        if not ft.file_exists(mask_image_path_name_ext):
            lt.error_and_raise(
                FileNotFoundError,
                "Error in MaskInfillImageProcessor(): " + f"mask image file {mask_image_path_name_ext} does not exist!",
            )

        # register inputs
        self.mask_image_path_name_ext = mask_image_path_name_ext
        self.cache_dir = cache_dir

        # in-memory cached values
        self.mask: npt.NDArray[np.uint8] = None
        """ The mask image. 1 for pixels that should be filled in. 0 for pixels
        that should be kept from the image being evaluated. """
        self.sized_mask: npt.NDArray[np.uint8] = None
        """ The mask image, scaled to the same width and height as the image being evaluated. """
        self.contour_pixels: npt.NDArray[np.uint8] = None
        """ The pixels that define the contour of the mask. 1 where the valid
        pixels meet the masked pixels. """
        self.num_marching_edges_iterations: int = None
        """ How many iterations of marching edges it takes to cover the entire mask. """
        self.extended_corner_masks: dict[str, npt.NDArray[np.uint8]] = {}
        """
        Extended versions of the sized mask. The masked area is extended to the
        top-left ("tl"), top-right ("tr"), bottom-right ("br"), and bottom-left
        ("bl"), so that the marching edges from cv.inpaint() overlap.
        """
        self.corner_mask_scales: dict[str, npt.NDArray[np.float32]] = {}
        """
        How much to weigh the result from a top-left ("tl"), top-right ("tr"),
        bottom-right ("br"), and bottom-left ("bl") inpaint, as a simple way of
        interpolating the result from the marching edges algorithm.
        """

        # algorithm images
        self.example_inpainted_image: CacheableImage = None

        # debugging
        self.setup_debug_images()

    def setup_debug_images(self, tile_array=(3, 2)):
        # debugging
        self.axis_control = rca.image(grid=False)
        self.figure_control = rcfg.RenderControlFigure(tile=True, tile_array=(3, 2))
        self.view_spec_2d = vs.view_spec_im()
        self.debug_fig_records = []

    def draw_debug_image(self, image: np.ndarray, title: str, block: bool):
        image = image.copy()
        fig_record = fm.setup_figure(
            self.figure_control, self.axis_control, self.view_spec_2d, title=title, code_tag=f"{__file__}"
        )
        self.debug_fig_records.append(fig_record)
        fig_record.view.imshow(image)
        if block:
            fig_record.view.show(block=True)
            for fig_record in self.debug_fig_records:
                fig_record.close()
            self.debug_fig_records.clear()
        else:
            fig_record.view.show(block=False)

    def get_mask(self) -> npt.NDArray[np.uint8]:
        """
        Get the mask of areas that need to be filled in. 1 for pixels that
        need to be filled in. 0 for pixels that are valid in the source
        image.
        """
        if self.mask is not None:
            return self.mask

        # load the mask image
        mask_image = np.array(Image.open(self.mask_image_path_name_ext))
        mask_image = mask_image.squeeze()
        if mask_image.ndim > 2:
            mask_image = mask_image[:, :, 0]

        # get the mask from this value
        self.mask = np.zeros_like(mask_image)
        self.mask[np.where(mask_image == 0)] = 255

        return self.mask

    def get_sized_mask(self, shape: tuple[int, int]) -> npt.NDArray[np.uint8]:
        """
        Get the mask of areas that need to be filled in, resized to match the
        given shape. 1 for pixels that need to be filled in. 0 for pixels
        that are valid in the source image.
        """
        # check for a cached version
        width, height = shape[1], shape[0]
        if self.sized_mask is not None:
            sized_mask_width, sized_mask_height = self.sized_mask.shape[1], self.sized_mask.shape[0]
            if (sized_mask_width == width) and (sized_mask_height == height):
                return self.sized_mask
        else:
            self.example_inpainted_image = None
            # TODO reset everything

        # get the unsized mask
        mask = self.get_mask()
        mask_width, mask_height = mask.shape[1], mask.shape[0]
        if (mask_width == width) and (mask_height == height):
            self.sized_mask = mask
            return self.sized_mask

        # resize the mask to the given shape
        self.sized_mask = np.zeros((height, width), dtype=np.uint8)
        fx, fy = 0, 0  # determined automatically
        interpolation = cv.INTER_NEAREST_EXACT
        self.sized_mask = cv.resize(mask, (width, height), self.sized_mask, fx, fy, interpolation)
        lt.debug(f"In get_sized_mask: {mask.shape=}, {shape=}, {self.sized_mask.shape=}")

        return self.sized_mask

    def get_explainer_images(self, sized_mask: npt.NDArray[np.uint8]) -> list[CacheableImage]:
        if self.example_inpainted_image is not None:
            scale_images = list(self.corner_mask_scales.values())
            scales_max = np.max([np.max(scale_image) for scale_image in scale_images])
            scales_explainers = [
                CacheableImage.from_single_source(
                    ir.false_color_reshaper((scale_image / scales_max * 255.0).astype(np.uint8))
                )
                for scale_image in scale_images
            ]
            return [self.example_inpainted_image, *scales_explainers]
        width, height = sized_mask.shape[1], sized_mask.shape[0]

        # draw images using figure_management
        axis_control = rca.image(grid=False, draw_axes=False)
        figure_control = rcfg.RenderControlFigure(draw_whitespace_padding=False)

        #####################################
        # build the all-point influence image
        #####################################

        # get the example image, for demonstrating filling
        stripes = np.zeros((height, width, 3), dtype=np.uint8)
        color_wheel = [color.red(), color.green(), color.blue()]
        for i in range(0, width, 50):
            start, end = i, np.min([width, i + 50])
            stripes[:, start:end] = color_wheel[i % 3].rgb_255()

        # show the fully filled sized_mask region
        stripes_filled = self._fill_image(stripes, sized_mask)
        fig_record = fm.setup_figure(
            figure_control,
            axis_control,
            vs.view_spec_im(),
            name=f"{self.name} Explainer Images",
            code_tag=f"{__file__}",
            equal=False,
        )
        fig_record.view.imshow(stripes_filled)
        self.example_inpainted_image = CacheableImage.from_single_source(fig_record.to_array())
        fig_record.close()

        # debugging
        if False:
            self.setup_debug_images((1, 1))
            self.draw_debug_image(self.example_inpainted_image.nparray)

        return self.get_explainer_images(sized_mask)

    def get_contour(self, mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if self.contour_pixels is not None:
            return self.contour_pixels

        # get a mask image that is 0 where the mask is, and 1 otherwise
        inverse_mask = np.ones_like(mask)
        inverse_mask[np.where(mask > 0)] = 0

        # Add a 1-pixel border to our inverse mask. We do this so that when we
        # find the contours, and the contours include all the edge pixels, we
        # can ignore said edge pixels by trimming down to the original size.
        width, height = mask.shape[1], mask.shape[0]
        buffered_inverse_mask = np.ones((height + 2, width + 2), dtype=np.uint8)
        buffered_inverse_mask[1 : height + 1, 1 : width + 1] = inverse_mask

        # find the contours
        zeros = np.zeros_like(buffered_inverse_mask)
        contours, _ = cv.findContours(buffered_inverse_mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        buffered_contour_pixels = cv.drawContours(zeros, contours, -1, 255)

        # remove edge pixels
        self.contour_pixels = buffered_contour_pixels[1 : height + 1, 1 : width + 1]

        return self.contour_pixels

    def get_num_marching_edges_iterations(self, sized_mask: npt.NDArray[np.uint8]) -> int:
        # check for in-memory cached values
        if self.num_marching_edges_iterations is not None:
            return self.num_marching_edges_iterations

        # get initial values
        valid_pixels = np.ones(sized_mask.shape, dtype=np.float32)
        valid_pixels[np.where(sized_mask > 0)] = 0

        # iterate until we've filled the entire masked area
        self.num_marching_edges_iterations = 0
        while np.min(valid_pixels) == 0:
            new_valid_pixels: npt.NDArray[np.float32]
            """ Pixels that are a part of this iteration, and only this iteration. """

            # get one iteration of convolution
            new_valid_pixels = scipy.ndimage.gaussian_filter(valid_pixels.copy(), sigma=1, radius=1, mode='constant')

            # limit to the pixels from just this iteration
            new_valid_pixels_test = np.logical_and(new_valid_pixels > 0, valid_pixels == 0)
            iteration_pixels = np.where(new_valid_pixels_test)

            # get ready for the next iteration
            valid_pixels[iteration_pixels] = 1.0
            self.num_marching_edges_iterations += 1

        lt.info(
            "In InpaintImageProcessor.get_num_marchin_edges_iterations: "
            + f"mask width/height: {self.num_marching_edges_iterations}"
        )

        return self.num_marching_edges_iterations

    def _load_extended_corner_masks_from_cache(self, sized_mask: npt.NDArray[np.uint8]) -> bool:
        if self.cache_dir is None:
            return False

        # check that the cached sized_mask exists and matches the given sized_mask
        cached_sized_mask_file = ft.join(self.cache_dir, f"{self.name}-cached_sized_mask.npy")
        if not ft.file_exists(cached_sized_mask_file):
            return False
        cached_sized_mask = np.load(cached_sized_mask_file)
        if not np.equal(cached_sized_mask, sized_mask).all():
            return False

        # get the extended corner mask files
        lt.info(f"Loading extended corner masks from {self.cache_dir}")
        extended_corner_mask_files = ft.files_in_directory_by_extension(self.cache_dir, [".npy"])[".npy"]
        extended_corner_mask_files = list(
            filter(lambda ecmf: "extended_corner_masks" in ecmf, extended_corner_mask_files)
        )

        # load the extended corner masks
        self.extended_corner_masks.clear()
        for ecmf in extended_corner_mask_files:
            corner = ecmf.split("_")[-1].split(".")[0]
            self.extended_corner_masks[corner] = np.load(ft.join(self.cache_dir, ecmf))

        # verify we got all the required files
        required_corners = ['tl', 'tr', 'br', 'bl']
        for corner in required_corners:
            if corner not in self.corner_mask_scales:
                self.corner_mask_scales.clear()
                return False

        return True

    def _save_extended_corner_masks_to_cache(self, sized_mask: npt.NDArray[np.uint8]):
        if self.cache_dir is None:
            return
        ft.create_directories_if_necessary(self.cache_dir)

        # save the sized_mask
        cached_sized_mask_file = ft.join(self.cache_dir, f"{self.name}-cached_sized_mask.npy")
        np.save(cached_sized_mask_file, sized_mask)

        # clear out any old extended corner masks
        ft.delete_files_in_directory(self.cache_dir, f"{self.name}-extended_corner_masks_*.npy")

        # save the extended corner masks
        for corner in self.extended_corner_masks:
            fout_path_name_ext = ft.join(self.cache_dir, f"{self.name}-extended_corner_masks_{corner}.npy")
            np.save(fout_path_name_ext, self.extended_corner_masks[corner])

    def get_extended_corner_masks(self, sized_mask: npt.NDArray[np.uint8]) -> dict[str, npt.NDArray[np.uint8]]:
        width, height = sized_mask.shape[1], sized_mask.shape[0]

        # check for cached values
        if len(self.extended_corner_masks) > 0:
            return self.extended_corner_masks
        if self._load_extended_corner_masks_from_cache(sized_mask):
            return self.extended_corner_masks

        # get the required number of marching edge iterations
        num_marching_edges_iterations = self.get_num_marching_edges_iterations(sized_mask)

        # extend the mask to allow for interpolation between the two sides of
        # the marching edges
        for corner, xdir, ydir in [('tl', -1, -1), ('tr', 1, -1), ('br', 1, 1), ('bl', -1, 1)]:

            # create the extended corner mask
            self.extended_corner_masks[corner] = np.clip(sized_mask.copy(), 0, 1)
            for marching_iteration in range(num_marching_edges_iterations):
                # get the range to copy to/from
                from_xdir = (1) if (xdir == -1) else (-1)
                from_ydir = (1) if (ydir == -1) else (-1)
                xstart_to, xend_to = np.max([0 + xdir, 0]), np.min([width + xdir, width])
                ystart_to, yend_to = np.max([0 + ydir, 0]), np.min([height + ydir, height])
                xstart_from, xend_from = np.max([0 + from_xdir, 0]), np.min([width + from_xdir, width])
                ystart_from, yend_from = np.max([0 + from_ydir, 0]), np.min([height + from_ydir, height])

                # copy values
                extended_edge = np.zeros_like(self.extended_corner_masks[corner])
                extended_edge[:, xstart_to:xend_to] += self.extended_corner_masks[corner][:, xstart_from:xend_from]
                extended_edge[ystart_to:yend_to, :] += self.extended_corner_masks[corner][ystart_from:yend_from, :]

                # don't double-add values at the horizontal/vertical intersections
                extended_edge = np.clip(extended_edge, 0, 1)

                # update extended corner
                self.extended_corner_masks[corner] += extended_edge

        # debugging
        if False:
            self.setup_debug_images((2, 2))
            self.draw_debug_image(self.extended_corner_masks['tl'], block=False)
            self.draw_debug_image(self.extended_corner_masks['tr'], block=False)
            self.draw_debug_image(self.extended_corner_masks['br'], block=False)
            self.draw_debug_image(self.extended_corner_masks['bl'], block=True)

        # cache to disk
        if self.cache_dir is not None:
            self._save_extended_corner_masks_to_cache(sized_mask)

        return self.extended_corner_masks

    def _load_corner_mask_scales_from_cache(self, sized_mask: npt.NDArray[np.uint8]) -> bool:
        if self.cache_dir is None:
            return False

        # check that the cached sized_mask exists and matches the given sized_mask
        cached_sized_mask_file = ft.join(self.cache_dir, f"{self.name}-cached_sized_mask.npy")
        if not ft.file_exists(cached_sized_mask_file):
            return False
        cached_sized_mask = np.load(cached_sized_mask_file)
        if not np.equal(cached_sized_mask, sized_mask).all():
            return False

        # get the corner mask scale files
        lt.info(f"Loading corner mask scales from {self.cache_dir}")
        corner_mask_scale_files = ft.files_in_directory_by_extension(self.cache_dir, [".npy"])[".npy"]
        corner_mask_scale_files = list(filter(lambda ecmf: "corner_mask_scales" in ecmf, corner_mask_scale_files))

        # load the corner mask scales
        self.corner_mask_scales.clear()
        for ecmf in corner_mask_scale_files:
            corner = ecmf.split("_")[-1].split(".")[0]
            self.corner_mask_scales[corner] = np.load(ft.join(self.cache_dir, ecmf))

        # verify we got all the required files
        required_corners = ['tl', 'tr', 'br', 'bl']
        for corner in required_corners:
            if corner not in self.corner_mask_scales:
                self.corner_mask_scales.clear()
                return False

        return True

    def _save_corner_mask_scales_to_cache(self, sized_mask: npt.NDArray[np.uint8]):
        if self.cache_dir is None:
            return
        ft.create_directories_if_necessary(self.cache_dir)

        # save the sized_mask
        cached_sized_mask_file = ft.join(self.cache_dir, f"{self.name}-cached_sized_mask.npy")
        np.save(cached_sized_mask_file, sized_mask)

        # clear out any old corner mask scales
        ft.delete_files_in_directory(self.cache_dir, f"{self.name}-corner_mask_scales_*.npy")

        # save the corner mask scales
        for corner in self.corner_mask_scales:
            fout_path_name_ext = ft.join(self.cache_dir, f"{self.name}-corner_mask_scales_{corner}.npy")
            np.save(fout_path_name_ext, self.corner_mask_scales[corner])

    def get_corner_mask_scales(self, sized_mask: npt.NDArray[np.uint8]) -> dict[str, npt.NDArray[np.float32]]:
        width, height = sized_mask.shape[1], sized_mask.shape[0]

        # check for cached values
        if len(self.corner_mask_scales) > 0:
            return self.corner_mask_scales
        if self._load_corner_mask_scales_from_cache(sized_mask):
            return self.corner_mask_scales

        # get the number of iterations and the extended corner masks
        num_marching_edges_iterations = self.get_num_marching_edges_iterations(sized_mask)
        extended_corner_masks = self.get_extended_corner_masks(sized_mask)

        # create the corner mask scale from the extended corner masks
        corners = ['tl', 'tr', 'br', 'bl']
        for corner in corners:
            self.corner_mask_scales[corner] = extended_corner_masks[corner].astype(np.float32)

        for corner, xdir, ydir in [('tl', 1, 1), ('tr', -1, 1), ('br', -1, -1), ('bl', 1, -1)]:

            # start with this corner's extended mask
            scale_edge = np.zeros((height, width, 2), np.uint8)
            opposite_corner = corners[(corners.index(corner) + 2) % 4]
            for i in range(2):
                scale_edge[:, :, i] = extended_corner_masks[corner]

            # get the range to copy to/from for each iteration
            from_xdir = (1) if (xdir == -1) else (-1)
            from_ydir = (1) if (ydir == -1) else (-1)
            xstart_to, xend_to = np.max([0 + xdir, 0]), np.min([width + xdir, width])
            ystart_to, yend_to = np.max([0 + ydir, 0]), np.min([height + ydir, height])
            xstart_from, xend_from = np.max([0 + from_xdir, 0]), np.min([width + from_xdir, width])
            ystart_from, yend_from = np.max([0 + from_ydir, 0]), np.min([height + from_ydir, height])

            # iterate backwards to determine the scaling factor
            for marching_iteration in range(num_marching_edges_iterations):

                # copy the minimum amount from the previous horizontal and vertical direction
                scale_edge[ystart_to:yend_to, :, 0] = scale_edge[ystart_from:yend_from, :, 0]
                scale_edge[:, :, 0] = np.min(scale_edge, axis=2)
                scale_edge[:, xstart_to:xend_to, 1] = scale_edge[:, xstart_from:xend_from, 1]
                scale_edge[:, :, 1] = np.min(scale_edge, axis=2)

                # debugging
                if False:
                    show(self.corner_mask_scales[corner])

            # copy onto the scaler mask
            self.corner_mask_scales[corner] = scale_edge[:, :, 0].astype(np.float32)

            # clear outside the masked areas
            self.corner_mask_scales[corner][np.where(sized_mask == 0)] = 0

            # debugging
            if False:
                self.setup_debug_images((3, 2))
                self.draw_debug_image(sized_mask, title="sized_mask", block=False)
                self.draw_debug_image(
                    extended_corner_masks[corner], title=f"extended_corner_masks[{corner}]", block=False
                )
                self.draw_debug_image(
                    extended_corner_masks[opposite_corner],
                    title=f"extended_corner_masks[{opposite_corner}]",
                    block=False,
                )
                self.draw_debug_image(scale_edge[:, :, 0], title="scale_edge[:, :, 0]", block=False)
                self.draw_debug_image(scale_edge[:, :, 1], title="scale_edge[:, :, 1]", block=False)
                self.draw_debug_image(
                    self.corner_mask_scales[corner], title=f"self.corner_mask_scales[{corner}]", block=True
                )

        # rescale each corner mask scale so that in total they add up to 1
        total_mask = np.zeros_like(self.corner_mask_scales['tl'])
        mask_pixels = np.where(sized_mask > 0)
        for corner in corners:
            total_mask[mask_pixels] += self.corner_mask_scales[corner][mask_pixels]
        for corner in corners:
            self.corner_mask_scales[corner][mask_pixels] /= total_mask[mask_pixels]

        # debugging
        if False:
            self.setup_debug_images((3, 2))
            self.draw_debug_image(self.corner_mask_scales['tl'], title="self.corner_mask_scales['tl']", block=False)
            self.draw_debug_image(self.corner_mask_scales['tr'], title="self.corner_mask_scales['tr']", block=False)
            self.draw_debug_image(self.corner_mask_scales['br'], title="self.corner_mask_scales['br']", block=False)
            self.draw_debug_image(self.corner_mask_scales['bl'], title="self.corner_mask_scales['bl']", block=False)
            self.draw_debug_image(
                sum(self.corner_mask_scales.values()), title="sum(self.corner_mask_scales)", block=True
            )

        # cache these values to disk
        if self.cache_dir is not None:
            self._save_corner_mask_scales_to_cache(sized_mask)

        return self.corner_mask_scales

    def _fill_image(self, image: npt.NDArray[np.uint8], sized_mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        extended_corner_masks = self.get_extended_corner_masks(sized_mask)
        corner_mask_scales = self.get_corner_mask_scales(sized_mask)

        # discard the masked area
        inpaint_src = image.squeeze()
        inpaint_src[np.where(sized_mask > 0)] = 0

        # retain as much information as possible during inpainting
        if inpaint_src.ndim == 2:
            inpaint_src = inpaint_src.astype(np.float32)
        else:
            inpaint_src = inpaint_src

        # prepare the result to be returned
        filled_area = np.zeros_like(image, dtype=np.float32)

        # debugging
        if False:
            self.setup_debug_images((3, 2))

        # paint in from each of our corners
        inpaintRadius = 3
        flags = cv.INPAINT_TELEA
        for corner in corner_mask_scales:
            extended_corner_mask = extended_corner_masks[corner]
            corner_mask_scale = corner_mask_scales[corner]

            # match same dimensionality and number of color channels
            if corner_mask_scale.ndim < image.ndim:
                corner_mask_scale = np.expand_dims(corner_mask_scale, axis=2)
            if image.ndim >= 3:
                if corner_mask_scale.shape[2] < image.shape[2]:
                    new_corner_mask_scale = np.zeros(image.shape, dtype=corner_mask_scale.dtype)
                    for i in range(image.shape[2]):
                        new_corner_mask_scale[:, :, i] = corner_mask_scale[:, :, 0]
                    corner_mask_scale = new_corner_mask_scale

            # inpaint
            inpainted_corner = cv.inpaint(inpaint_src, extended_corner_mask, inpaintRadius, flags)
            filled_area += inpainted_corner * corner_mask_scale

            # debugging
            if False:
                debug_image = np.clip(inpainted_corner, 0, 255).astype(image.dtype)
                debug_image = debug_image + image
                self.draw_debug_image(debug_image, title=f"paint_corner['{corner}']", block=False)

        # convert to return type
        filled_area = np.clip(filled_area, 0, 255).astype(np.uint8)

        # add to the original image
        ret = image.copy() + filled_area

        # debugging
        if False:
            debug_image = ret.copy()
            try:
                debug_image = ir.false_color_reshaper(debug_image)
            except Exception:
                pass
            self.draw_debug_image(debug_image, title=f"filled_image", block=True)

        return ret

    def fill_image(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        sized_mask = self.get_sized_mask(image.shape)
        self.get_explainer_images(sized_mask)
        return self._fill_image(image, sized_mask)

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        image = operable.primary_image.nparray.copy()

        # explainer images
        explainer_images: list[CacheableImage] = self.get_explainer_images(self.get_sized_mask(image.shape))

        # fix the image
        filled_image: np.ndarray = self.fill_image(image)

        # vis images
        before1 = CacheableImage.from_single_source(operable.primary_image.nparray)
        after1 = CacheableImage.from_single_source(filled_image)
        before2 = CacheableImage.from_single_source(ir.false_color_reshaper(operable.primary_image.nparray))
        after2 = CacheableImage.from_single_source(ir.false_color_reshaper(filled_image))

        # update the result
        primary_image = CacheableImage(filled_image, source_path=operable.primary_image.source_path)
        visualization_images = copy.copy(operable.visualization_images)
        visualization_images[self] = [before1, after1, before2, after2]
        algorithm_images = copy.copy(operable.algorithm_images)
        algorithm_images[self] = explainer_images
        ret = dataclasses.replace(
            operable,
            primary_image=primary_image,
            visualization_images=visualization_images,
            algorithm_images=algorithm_images,
        )

        return [ret]


if __name__ == "__main__":
    import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
    from opencsp.common.lib.opencsp_path import opencsp_settings
    import opencsp.common.lib.tool.file_tools as ft
    import opencsp.common.lib.render.color as color
    import opencsp.common.lib.render.figure_management as fm
    import opencsp.common.lib.render.view_spec as vs
    import opencsp.common.lib.render_control.RenderControlAxis as rca
    import opencsp.common.lib.render_control.RenderControlFigure as rcfg
    import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
    import opencsp.common.lib.render_control.RenderControlPointSeq as rcps

    test_dir = ft.join(
        opencsp_settings["opencsp_root_path"]["collaborative_dir"],
        "NSTTF_Optics/Experiments/2024-05-22_SolarNoonTowerTest_4",
    )
    src_dir = ft.join(test_dir, "2_Data/BCS_data/20240522_115941 TowerTest4_NoSun/Raw Images")
    src_file = "20240522_115941.65 TowerTest4_NoSun Raw.JPG"
    mask_dir = ft.join(test_dir, "4_Analysis")
    mask_file = "shadow_mask.png"

    # Load the test image
    test_image = np.array(Image.open(ft.join(src_dir, src_file)))

    # Process the test image
    processor = InpaintImageProcessor(
        ft.join(mask_dir, mask_file), ft.join(orp.opencsp_cache_dir(), "MaskInfileImageProcessorTest")
    )
    processed_image = processor.process_images([test_image])[0].nparray

    # Draw the results
    axis_control = rca.image(grid=False)
    figure_control = rcfg.RenderControlFigure(tile=False, tile_array=(2, 1))
    view_spec_2d = vs.view_spec_im()

    fig_record1 = fm.setup_figure(figure_control, axis_control, view_spec_2d, title="img", equal=False)
    fig_record1.view.imshow(test_image)
    fig_record1.view.show(block=False)

    fig_record2 = fm.setup_figure(figure_control, axis_control, view_spec_2d, title="img", equal=False)
    fig_record2.view.imshow(processed_image)
    fig_record2.view.show(block=True)

    fig_record1.close()
    fig_record2.close()
