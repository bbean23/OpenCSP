import copy
import dataclasses

import cv2 as cv
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.tool.log_tools as lt


class EnclosedEnergyImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    A processor for calculating and visualizing the enclosed energy of an image.

    This processor calculates the enclosed energy of an image at expanding radii from its centroid
    and generates a plot of the enclosed energy. The plot can be used to visualize how the energy
    of the image is distributed around its centroid.

    Attributes
    ----------
    plot_x_range_pixels : int
        The x-axis range for the generated enclosed energy visualization image.
    """

    def __init__(
        self,
        plot_x_range_pixels: int = -1,
        calc_inner_radius_limit: int = -1,
        calc_radius_resolution: int = -1,
        calc_outer_radius_limit: int = -1,
    ):
        """
        Parameters
        ----------
        calc_inner_radius_limit: int, optional
            All radius values within this limit will be calculated, meaning that
            decimation won't be applied within this range. If <= 0 then all
            radius values for the entire calc_outer_radius_limit will be
            calculated (ignores calc_radius_resolution).
        calc_radius_resolution: int, optional
            How many radius values to measure when calculating the enclosed
            energy. For example, a resolution of 3 and inner radius of 4 means
            that the enclosed energy will be calculated for radii 1, 2, 3, 4, 7,
            10, 13... The in-between radii will be linearly interpolated from
            their neighbors. If <= 0 then all radius values will be included (no
            interpolation).
        calc_outer_radius_limit: int, optional
            How large of a radius to calculate the enclosed energy for. If <= 0
            then the entire size of the image is used.
        plot_x_range_pixels: int, optional
            Sets the x-axis range for the generated enclosed energy
            visualization image, or <= 0 to determine the range automatically.
            The x-axis is the pixels radius from the image centroid. Default is
            -1.
        """
        super().__init__()

        self.plot_x_range_pixels = plot_x_range_pixels
        self.calc_inner_radius_limit = calc_inner_radius_limit
        self.calc_radius_resolution = calc_radius_resolution
        self.calc_outer_radius_limit = calc_outer_radius_limit

    def _determine_interpolated_radii(self, max_radius: int) -> tuple[list[int], list[int]]:
        # Determine the subset of radii to calculate exact values for, and which
        # should be interpolated values.
        direct_radii: list[int] = []
        interpolated_radii: list[int] = []

        inner_limit = min(self.calc_inner_radius_limit, max_radius)
        resolution = 1 if self.calc_radius_resolution <= 0 else self.calc_radius_resolution
        outer_limit = max(min(self.calc_outer_radius_limit, max_radius), inner_limit)
        if self.calc_outer_radius_limit <= 0:
            outer_limit = max_radius
        if self.calc_inner_radius_limit <= 0:
            inner_limit = outer_limit

        for i in range(1, inner_limit + 1):
            direct_radii.append(i)
        for i in range(inner_limit + resolution, outer_limit + 1, resolution):
            direct_radii.append(i)

        for i in range(1, outer_limit):
            if i not in direct_radii:
                interpolated_radii.append(i)

        return direct_radii, interpolated_radii

    def calculate_enclosed_energy(self, image: CacheableImage) -> list[int]:
        """
        Calculates the enclosed energy of an image at expanding radii from its centroid.

        This method calculates the centroid of the input image and then iterates
        over radii from 1 to the image corners. For each radius, it creates a
        circular mask and calculates the enclosed energy by summing the pixel
        values within the circle.

        The maximum radius considered is the maximum distance from the centroid
        to the image corners.

        Parameters
        ----------
        image: CacheableImage
            The input image.

        Returns
        -------
        enclosed_energy_sums: list[int]
            A list of enclosed energies at each radius, as the sum of pixel
            values within that radius, from radius=1 to radius=max-radius.
        example_image: np.ndarray | None
            An example partial image of what is being measured.
        """
        enclosed_energy_sums: list[int] = []
        example_circle_image: np.ndarray = None

        # Calculate the centroid of the image
        moments = cv.moments(image.nparray)
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])

        # Determine the maximum radius to measure out to
        height, width = image.nparray.shape[0], image.nparray.shape[1]
        assert width == image.to_image().width
        corners = [(0, 0), (width, 0), (width, height), (0, height)]
        max_radius = 0
        for corner in corners:
            distance = np.sqrt((corner[0] - centroid_x) ** 2 + (corner[1] - centroid_y) ** 2)
            max_radius = int(max(max_radius, distance))
        enclosed_energy_sums = [-1 for i in range(1, max_radius + 1)]

        # Determine the subset of radii to calculate exact values for, and which
        # should be interpolated values.
        direct_radii, interpolated_radii = self._determine_interpolated_radii(max_radius)
        example_radius = direct_radii[int(len(direct_radii) / 2)]

        # Calculate the enclosed energy
        circle_mask = np.zeros_like(image.nparray)
        y_indicies, x_indicies = np.indices(image.nparray.shape)
        for radius in direct_radii:
            # Create a mask using np.where to select pixels within the circle
            circle_mask = cv.circle(
                circle_mask, center=(centroid_y, centroid_x), radius=radius, color=(1), thickness=cv.FILLED
            )

            # Create the example image
            if radius == example_radius:
                example_circle_image = np.copy(image.nparray) * circle_mask

            # Calculate the enclosed energy within the circle
            enclosed_energy = np.sum(image.nparray * circle_mask)
            enclosed_energy_sums[radius - 1] = enclosed_energy

        # Interpolate the rest of the results
        for radius in interpolated_radii:
            lower_radius = max(filter(lambda r: r < radius, direct_radii))
            upper_radius = min(filter(lambda r: r > radius, direct_radii))
            lower_val = enclosed_energy_sums[lower_radius - 1]
            upper_val = enclosed_energy_sums[upper_radius - 1]
            diff = upper_val - lower_val
            val = ((diff) / (upper_radius - lower_radius) * (radius - lower_radius)) + lower_val
            enclosed_energy_sums[radius - 1] = val

        return enclosed_energy_sums, example_circle_image

    def build_enclosed_energy_plot(self, enclosed_energy_sums: list[int]) -> np.ndarray:
        """
        Builds a plot of the enclosed energy around the centroid of the input image.

        This method uses the results from the :py:meth:`calculate_enclosed_energy` method to
        create a line plot of the enclosed energy at expanding radii from the image
        centroid.

        Parameters
        ----------
        enclosed_energy_sums: list[int]
            The output from calculate_enclosed_energy.

        Returns
        -------
        plot_image: np.ndarray
            A np.ndarray containing the enclosed energy plot.
        """
        # Determine the axes ranges
        if self.plot_x_range_pixels > 0:
            x_range = self.plot_x_range_pixels
        else:
            x_range = len(enclosed_energy_sums)

        # Build the data as a fraction of the total
        pq_vals: list[tuple[int, int]] = []
        enclosed_energy_fractions = [sum_val / enclosed_energy_sums[-1] for sum_val in enclosed_energy_sums]
        for radius in range(1, len(enclosed_energy_fractions) + 1):
            pq_vals.append(tuple(radius, enclosed_energy_fractions[radius - 1]))

        # Pad the plot for the given plot_x_range_pixels, if any is given
        for radius in range(len(enclosed_energy_fractions) + 1, x_range):
            pq_vals.append(tuple(radius, enclosed_energy_fractions[-1]))

        # Create a new figure for the plot
        figure_control = rcfg.RenderControlFigure()
        axis_control = rca.RenderControlAxis(x_label='Radius (pixels)', y_label='Enclosed Energy (Sum)')
        view_spec_2d = vs.view_spec_xy()
        fig_record = fm.setup_figure_for_3d_data(
            figure_control, axis_control, view_spec_2d, name="enclosed_energy", code_tag=f"{__file__}.main()"
        )
        view = fig_record.view

        # Draw the plot
        view.draw_pq_list(pq_vals)
        plot_image = fig_record.to_array()

        return plot_image

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        # Calculate the enclosed energy around the centroid of the image
        enclosed_energy_sums, example_enclosed_energy_image = self.calculate_enclosed_energy(operable.primary_image)

        # Generate a visual plot of the enclosed energy
        enclosed_energy_plot = self.build_enclosed_energy_plot(enclosed_energy_sums)

        # Build the new operable
        notes = copy.copy(operable.image_processor_notes)
        notes.append(self.name, [str(v) for v in enclosed_energy_sums])
        algorithm_images = copy.copy(operable.algorithm_images)
        algorithm_images[self] = [CacheableImage.from_single_source(example_enclosed_energy_image)]
        vis_images = copy.copy(operable.visualization_images)
        vis_images[self] = [CacheableImage.from_single_source(enclosed_energy_plot)]
        ret = dataclasses.replace(
            operable, image_processor_notes=notes, algorithm_images=algorithm_images, visualization_images=vis_images
        )

        return [ret]
