import copy
from io import TextIOWrapper
import multiprocessing
import multiprocessing.pool
import os
import re
import time

import numpy as np
import numpy.typing as npt
from PIL import Image
import psutil

from opencsp import opencsp_settings
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class PixelData:
    """
    This class interprets the image data from a yyyymmDD_HHMMss_PixelData.csv file.

    Some of the files that the NSTTF BCS camera capture software generates are the
    raw pixel data, in the form of text-readable values, one value per pixel, one
    row per line. For example, a 5x5 image with a bright white dot in the center
    would look like::

        0,0,0,0,0
        0,0,0,0,0
        0,0,255,0,0
        0,0,0,0,0
        0,0,0,0,0
        70,0,0,0,0

    Notice the last line, which starts with "70" followed by many 0s. This indicates
    the end of an image (an image break). For files with multiple images in them,
    there will be more data after this line, and another image break line after each
    image.
    """

    def __init__(self, csv_path_name_ext: str):
        self.csv_path_name_ext = csv_path_name_ext
        self._nrows: int = None
        self._ncols: int = None
        self._nimages: int = None

        self._parse_file_stats()

    def raw_image_names(self) -> list[str]:
        """The names of the images in the associated "Raw Images" directory."""
        csv_path, csv_name, csv_ext = ft.path_components(self.csv_path_name_ext)
        raw_dir = ft.join(csv_path, "Raw Images")
        image_files = it.image_files_in_directory(raw_dir)
        return image_files

    def _parse_file_stats(self):
        if self._nrows is not None:
            return

        # parse the file statistics
        example_line = ""
        nrows = 0
        nempty = 0
        with open(self.csv_path_name_ext, "r") as fin:
            for row in fin:
                row = row.strip()
                if row != "":
                    if example_line == "":
                        example_line = row
                    nrows += 1
                else:
                    nempty += 1
        if example_line == "" or nrows == 0:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData._get_file_stats(): "
                + f"expected at least one row for the csv file {self.csv_path_name_ext}, "
                + f"but found {nrows} rows",
            )
        ncols = len(example_line.split(","))

        # retrieve an example image, to compare height and width to rows and columns
        image_files = self.raw_image_names()
        if len(image_files) == 0:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData._get_file_stats(): " + f"expected at least one image in {raw_dir}, but found 0",
            )
        expected_nimages = len(image_files)
        raw_dir = ft.join(ft.path_components(self.csv_path_name_ext)[0], "Raw Images")
        image_file = ft.join(raw_dir, image_files[0])
        img = Image.open(image_file)

        # validate the parsed statistics
        expected_height, expected_width = img.height, img.width
        if ncols != expected_width:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData._get_file_stats(): "
                + f"expected images in {self.csv_path_name_ext} to be {expected_width} pixels wide, "
                + f"but images are instead {ncols} columns wide",
            )
        nimages = int(np.floor(nrows / (expected_height + 1)))
        if nimages != expected_nimages:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData._get_file_stats(): "
                + f"expected to find {expected_nimages} images in {self.csv_path_name_ext}, "
                + f"but instead found {nimages}",
            )
        expected_nrows = nimages * (expected_height + 1)
        if nrows != expected_nrows:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData._get_file_stats(): "
                + f"expected to find {expected_nrows} rows ({expected_height}+1 per image) in {self.csv_path_name_ext}, "
                + f"but instead found {nrows}",
            )

        # statistics look good, cache them
        self._nrows = nrows
        self._ncols = ncols
        self._nimages = nimages

    @property
    def nrows(self) -> int:
        """The total number of non-empty rows in this instance's "PixelData" csv file."""
        return self._nrows

    @property
    def ncols(self) -> int:
        """The number of columns per row in this instance's "PixelData" csv file."""
        return self._ncols

    @property
    def nimages(self) -> int:
        """The total number of images in this instance's "PixelData" csv file."""
        return self._nimages

    @property
    def width(self) -> int:
        """The width of each of this instance's images."""
        return self.ncols

    @property
    def height(self) -> int:
        """The height of each of this instance's images."""
        return int(np.floor(self.nrows / self.nimages)) - 1

    def parse_image_from_lines(self, line0_row_idx: int, lines: list[str]) -> np.ndarray:
        image = np.zeros((self.height, self.width), dtype=np.uint8)

        for y in range(self.height):
            row = lines[y]
            row = row.strip()
            svals = row.split(",")
            if len(svals) != self.width:
                lt.error_and_raise(
                    RuntimeError,
                    f"Unexpected error in PixelData.parse_images() on line {line0_row_idx+y} of {self.csv_path_name_ext}",
                )

            for x, sval in enumerate(svals):
                if sval == "":
                    lt.error_and_raise(
                        RuntimeError,
                        f"Unexpected error in PixelData.parse_images() on line {line0_row_idx+y} of {self.csv_path_name_ext}",
                    )
                val = int(sval)
                if x == 0 and val == 70:
                    pass
                if val < 0 or val > 255:
                    lt.error_and_raise(
                        RuntimeError,
                        f"Unexpected error in PixelData.parse_images() on line {line0_row_idx+y} of {self.csv_path_name_ext}",
                    )
                image[y, x] = val

        return image

    def _parse_next_image(self, fin: TextIOWrapper, line0_row_idx: int):
        lines = [fin.readline() for y in range(self.height)]
        image = self.parse_image_from_lines(line0_row_idx, lines)
        row_idx = line0_row_idx + len(lines)

        image_break_row = fin.readline()
        image_break_row = image_break_row.strip()
        if not image_break_row.startswith("70,"):
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData.parse_images(): "
                + f"expected the image break row {row_idx} to start with \"70,\", "
                + f"but instead it starts with \"{image_break_row[:10]}\"",
            )
        image_break_row_zeros = image_break_row[3:]
        empty_image_break_row = image_break_row_zeros.replace("0,", "")
        if empty_image_break_row != "0":
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData.parse_images(): "
                + f"expected the image break row {row_idx} to end with \"0,...,0\", "
                + f"but it is instead \"{image_break_row}\"",
            )

        return image, row_idx

    def peak_first_image(self) -> np.ndarray:
        with open(self.csv_path_name_ext, "r") as fin:
            image, _ = self._parse_next_image(fin, 0)
        return image

    def parse_images(self) -> list[np.ndarray]:
        """Retrieves the images from this instance's "PixelData" CSV file."""
        ret: list[np.ndarray] = []

        with open(self.csv_path_name_ext, "r") as fin:
            row_idx = 0

            for i in range(self.nimages):
                image, row_idx = self._parse_next_image(fin, row_idx)

                ret.append(image)
                lt.info(",", end="")

        return ret

    def subtract_from_raw(self, image_name_ext: str, image: np.ndarray) -> npt.NDArray[np.int32]:
        """Subtracts the given image from the matching image_name_ext in the
        "Raw Images" directory and returns the result."""
        raw_dir = ft.join(ft.path_components(self.csv_path_name_ext)[0], "Raw Images")
        raw_img = Image.open(ft.join(raw_dir, image_name_ext))
        raw_image = np.array(raw_img)
        return raw_image.astype(np.int32) - image.astype(np.int32)

    def convert_file(self, delete_after_conversion=False):
        """Converts the CSV file to .png image files, and possibly deletes the CSV file once complete."""
        max_allowed_raw_images_diff = 255

        # print simple stats
        csv_path, csv_name, csv_ext = ft.path_components(self.csv_path_name_ext)
        # lt.info(f"{csv_name}: {self.nrows=}, {self.ncols=}, {self.nimages=}, {self.width=}, {self.height=}")
        lt.info(f"Saving {self.nimages} images for {self.csv_path_name_ext}", end="")

        # parse and save all images from the file
        out_dir = ft.join(csv_path, "PixelData Images")
        ft.create_directories_if_necessary(out_dir)
        image_names = self.raw_image_names()
        for image_name_ext, image in zip(image_names, self.parse_images()):
            # verify the image isn't too different from it's "raw" counterpart
            comparison: np.ndarray = self.subtract_from_raw(image_name_ext, image)
            diff = np.max(np.abs(comparison))
            if diff > max_allowed_raw_images_diff:
                lt.error_and_raise(
                    RuntimeError,
                    "Error in convert_pixel_data_files.py: "
                    + f"{max_allowed_raw_images_diff=} but the maximum diff for {image_name_ext} is {diff}",
                )

            # save the image
            _, image_name, image_ext = ft.path_components(image_name_ext)
            img = Image.fromarray(image)
            png_path_name_ext = ft.join(out_dir, image_name + ".png")
            if ft.file_exists(png_path_name_ext):
                raise RuntimeError()
                ft.delete_file(png_path_name_ext)
            img.save(png_path_name_ext)

            lt.info(".", end="")

        # compare the before/after size
        before_size = ft.file_size(self.csv_path_name_ext)
        before_size += 4096  # don't know size on disk, assume one extra block
        after_size = 0
        for png_name_ext in ft.files_in_directory(out_dir, files_only=True):
            after_size += ft.file_size(ft.join(out_dir, png_name_ext))
            after_size += 4096  # don't know size on disk, assume one extra block
        lt.info(f" compression ratio={int(np.round(after_size / before_size * 100)):02d}%", end="")

        # delete the original file to save space on disk
        raise RuntimeError()
        ft.delete_file(self.csv_path_name_ext)


def find_pixel_data_files(search_dir: str):
    ret: list[str] = []
    pd_pattern = re.compile(r"^[0-9]{8}_[0-9]{6}_PixelData.csv$")

    for fn in ft.files_in_directory(search_dir):
        file_path_name_ext = ft.join(search_dir, fn)
        if os.path.isdir(file_path_name_ext):
            ret += find_pixel_data_files(file_path_name_ext)
        elif os.path.isfile(file_path_name_ext):
            if pd_pattern.match(fn) is not None:
                ret.append(file_path_name_ext)

    return ret


def process_pixel_data_file(csv_path_name_ext):
    pd = PixelData(csv_path_name_ext)
    pd.convert_file(delete_after_conversion=True)


if __name__ == "__main__":
    experiments_dir = ft.join(opencsp_settings['opencsp_root_path']['collaborative_dir'], "Experiments")
    experiments = [
        "2024-05-22_SolarNoonTowerTest_4",  # already extracted
        "2024-06-13_SolarNoonTowerTest_5",  # already extracted
        "2024-06-16_FluxMeasurement",  # already extracted
        "2024-06-18_SolarNoonTowerTest_6",  # already extracted
        "2024-07-07_SolarNoonTowerTest_7",  # already extracted
        "2024-07-12_Flux_Test",  # already extracted
    ]

    for experiment in experiments:
        experiment_dir = ft.join(experiments_dir, experiment)
        experiment_files = ft.files_in_directory(experiment_dir, recursive=True)
        lt.info(experiment_dir)

        # find pixel data files and the pixel data images
        pixel_data_files = find_pixel_data_files(experiment_dir)
        new_pixel_data_files: list[str] = []
        extracted_pixel_data_files: list[tuple[str, str, str]] = []
        for pixel_data_file in pixel_data_files:
            pixel_data_path, pixel_data_name, pixel_data_ext = ft.path_components(pixel_data_file)
            extracted_dir = ft.join(pixel_data_path, "PixelData Images")
            raw_dir = ft.join(pixel_data_path, "Raw Images")
            if ft.directory_exists(extracted_dir):
                extracted_pixel_data_files.append((pixel_data_file, extracted_dir, raw_dir))
            else:
                new_pixel_data_files.append(pixel_data_file)
        if len(pixel_data_files) != len(extracted_pixel_data_files) + len(new_pixel_data_files):
            raise RuntimeError()

        lt.info(f"{len(extracted_pixel_data_files)}", end="")
        for pixel_data_file, extracted_dir, raw_dir in extracted_pixel_data_files:
            # check that we have the expected number of extracted images
            parser = PixelData(pixel_data_file)
            raw_img_files = it.image_files_in_directory(raw_dir)
            if parser.nimages != len(raw_img_files):
                raise RuntimeError(
                    f"Number of raw images and number of images in pixel data file don't match! {len(raw_img_files)=}, {parser.nimages=}, {pixel_data_file=}"
                )
            extracted_img_files = it.image_files_in_directory(extracted_dir)
            if parser.nimages != len(extracted_img_files):
                raise RuntimeError(
                    f"Number of extracted images and number of images in pixel data file don't match! {len(extracted_img_files)=}, {parser.nimages=}, {pixel_data_file=}"
                )

            # check that extracted pixel data files match the extracted images
            first_extracted_image_file = ft.join(extracted_dir, extracted_img_files[0])
            first_extracted_image = np.array(Image.open(first_extracted_image_file))
            first_pixel_data_image = parser.peak_first_image()
            if not np.array_equal(first_extracted_image, first_pixel_data_image):
                raise RuntimeError(
                    f"The pixels for the first extracted and parsedf images don't match! {pixel_data_file=}, {first_extracted_image_file=}"
                )

            # If we made it this far, then we can be pretty confident that the
            # extracted images match the values in the original pixel data csv
            # file.
            # Delete the pixel data file.
            ex = None
            did_delete = False
            for i in range(10):
                try:
                    ft.delete_file(pixel_data_file)
                    lt.info(".", end="")
                    did_delete = True
                    break
                except PermissionError as ex:
                    time.sleep(1)
            if not did_delete:
                raise ex

        lt.info("")
