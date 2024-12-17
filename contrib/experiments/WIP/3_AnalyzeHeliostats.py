import copy
import datetime
import multiprocessing
import re
import shutil

import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract

from contrib.experiments.WIP.ViewCustomCrossSectionImageProcessor import ViewCustomCrossSectionImageProcessor
from opencsp import opencsp_settings
from contrib.common.lib.cv.spot_analysis.image_processor import *
from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.SpotAnalysis import SpotAnalysis
from opencsp.common.lib.cv.annotations.HotspotAnnotation import HotspotAnnotation
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import *
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render.PowerpointSlide as ps
import opencsp.common.lib.render.lib.PowerpointImage as ppi
import opencsp.common.lib.render.lib.PowerpointText as pptext
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlPowerpointPresentation as rcpp
import opencsp.common.lib.render_control.RenderControlPowerpointSlide as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


def get_parent_dir(image_path_name_ext: str):
    """
    Returns the parent's parent directory of the the given image and it's
    siblings. For example, "experiment_time" will be returned for the
    image_path_name_ext "experiment_name/experiment_time/heliostat/image_name.ext".
    """
    image_path, image_name, image_ext = ft.path_components(image_path_name_ext)
    parent_path, image_subdir, _ = ft.path_components(image_path)
    return parent_path


def crop_images(to_crop_dir: str, left_upper_right_lower: tuple[int, int, int, int]):
    for image_name_ext in it.image_files_in_directory(to_crop_dir):
        _, n, e = ft.path_components(image_name_ext)
        cropped_image_name_ext = n + "_cropped" + e
        img = Image.open(ft.join(to_crop_dir, image_name_ext))
        left, upper, right, lower = 965, 0, 965 + 661, 0 + 526
        img = img.crop((left, upper, right, lower))
        img.save(ft.join(to_crop_dir, cropped_image_name_ext))
        gain = it.get_exif_value(to_crop_dir, cropped_image_name_ext)
        if gain is None:
            gain = it.get_exif_value(to_crop_dir, image_name_ext)
            it.set_exif_value(to_crop_dir, cropped_image_name_ext, str(gain))
            gain = it.get_exif_value(to_crop_dir, cropped_image_name_ext)
            if gain is None:
                lt.error_and_raise(RuntimeError, cropped_image_name_ext)
            else:
                ft.delete_file(ft.join(to_crop_dir, image_name_ext))
        else:
            ft.delete_file(ft.join(to_crop_dir, image_name_ext))


def get_time_from_image_name(bcs_image_name: str):
    """Example name '20241004_150547.27 hourly_1500_images.png'"""
    _, bcs_image_name, _ = ft.path_components(bcs_image_name)
    time_pattern = re.compile(r"^([0-9]{8})_([0-9]{6})\.([0-9]{2}) .*$")

    # verify the format
    match = time_pattern.match(bcs_image_name.strip())
    if match is None:
        lt.error_and_raise(
            ValueError, "Error in get_time_from_image_name: " + f"unexpected format for image name {bcs_image_name}"
        )

    # parse out the time
    ymd = match.groups()[0]
    hms = match.groups()[1]
    centi = match.groups()[2]
    year, month, day = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:])
    hour, minute, second = int(hms[:2]), int(hms[2:4]), int(hms[4:])
    centisecond = int(centi)

    # create the time instance
    dt = datetime.datetime(year, month, day, hour, minute, second, centisecond * 10 * 1000)

    return dt


def preprocess_images(images_dir: str, image_names_exts: list[str]) -> CacheableImage:
    image_processors = {
        "AvgG": AverageByGroupImageProcessor(lambda o: 0, lambda l: None),
        "Conv": ConvolutionImageProcessor(),
    }
    sa = SpotAnalysis("preprocess_images", image_processors=list(image_processors.values()))
    sa.set_primary_images([ft.join(images_dir, fn) for fn in image_names_exts])
    for result in sa:
        pass

    return result.primary_image


def process_images(
    on_sun_dir: str,
    no_sun_dir: str | None,
    results_dir: str,
    on_sun_images: list[str] = None,
    no_sun_images: list[str] = None,
) -> tuple[str, str, str]:
    """
    Find the hotspot for the given on sun images, and plot the cross section
    (including the no-sun cross section). The resulting visualization images are
    saved in the results_dir.

    Parameters
    ----------
    on_sun_dir : str
        Directory containing the on-sun images for some number of heliostats.
    no_sun_dir : str | None
        Directory containing the corresponding no-sun images. None if there are
        no matching no-sun images.
    results_dir : str
        Where to save the output visualization images to.

    Returns
    -------
    hotspot_vis: str
        The path/name.ext of the saved hotspot visualization image.
    crosssection_vis_1: str
        The path/name.ext of the saved cross section visualization image.
    crosssection_vis_2: str
        The path/name.ext of the saved cross section visualization image.
    """
    # load the no-sun images
    no_sun_image = None
    if no_sun_dir is not None:
        if no_sun_images is None:
            no_sun_images = it.image_files_in_directory(no_sun_dir)
        no_sun_image = preprocess_images(no_sun_dir, no_sun_images)
    else:
        no_sun_images = []

    # load the heliostat images
    if on_sun_images is None:
        on_sun_images = it.image_files_in_directory(on_sun_dir)
    on_sun_image = preprocess_images(on_sun_dir, on_sun_images)

    # average the no_sun images
    if no_sun_image is not None:
        _, no_sun_name, no_sun_ext = ft.path_components(no_sun_images[0])
        avg_no_sun_image_path = ft.join(results_dir, f"{no_sun_name}_no_sun_avg.png")
        if ft.file_exists(avg_no_sun_image_path):
            ft.delete_file(avg_no_sun_image_path)
        no_sun_image.to_image().save(avg_no_sun_image_path)
        no_sun_image = CacheableImage.from_single_source(avg_no_sun_image_path)

    def hotspot_pixel_locator(operable: SpotAnalysisOperable) -> tuple[int, int]:
        """Returns the x/y pixel location of the hotspot center"""
        hotspots = filter(lambda a: isinstance(a, HotspotAnnotation), operable.annotations)
        hs_annotation = list(hotspots)[0]
        ret = hs_annotation.origin.astuple()
        return (int(ret[0]), int(ret[1]))

    # build the list of image processors for the primary image
    image_processors = {
        "Echo": EchoImageProcessor(),
        "BSub": BackgroundColorSubtractionImageProcessor('plane'),
        "HotS": HotspotImageProcessor(15, record_visualization=True),
        "XSec": ViewCustomCrossSectionImageProcessor(hotspot_pixel_locator, single_plot=False, y_range=(0, 255)),
        "Cent": MomentsImageProcessor(include_visualization=True),
        "Enrg": EnclosedEnergyImageProcessor("centroid", plot_x_limit_pixels=400),
    }

    # build the list of image processors for the no-sun image
    nosun_image_processors = {"BSub": BackgroundColorSubtractionImageProcessor()}

    # process the on-sun and no-sun images

    # Find the hotspot and visualize the vertical/horizontal cross-sections
    if no_sun_image is not None:
        nosun_spot_analysis = SpotAnalysis(
            "SpotAnalysis - No Sun", image_processors=list(nosun_image_processors.values())
        )
        nosun_hel_operable = SpotAnalysisOperable(
            no_sun_image, primary_image_source_path=ft.join(no_sun_dir, no_sun_images[0])
        )
        nosun_spot_analysis.set_input_operables([nosun_hel_operable])
        nosun_result = next(iter(nosun_spot_analysis))
        image_processors["XSec"].no_sun_image = nosun_result.primary_image.nparray
    spot_analysis = SpotAnalysis("SpotAnalysis", image_processors=list(image_processors.values()))
    hel_operable = SpotAnalysisOperable(on_sun_image, primary_image_source_path=ft.join(on_sun_dir, on_sun_images[0]))
    spot_analysis.set_input_operables([hel_operable])
    result = next(iter(spot_analysis))

    # Save the visualization images to the results directory
    background_sub_algo_images = result.algorithm_images[image_processors["BSub"]]
    hotspot_vis_image = result.visualization_images[image_processors["HotS"]][-1]
    crosssec_vis_images = result.visualization_images[image_processors["XSec"]]
    moments_vis_images = result.visualization_images[image_processors["Cent"]]
    enclosed_energy_vis_images = result.visualization_images[image_processors["Enrg"]]
    _, on_sun_name, on_sun_ext = ft.path_components(on_sun_images[0])
    if no_sun_image is not None:
        _, no_sun_name, no_sun_ext = ft.path_components(no_sun_images[0])
    ft.copy_file(hel_operable.primary_image_source_path, results_dir, f"{on_sun_name}_original.png")
    for i, image in enumerate(background_sub_algo_images):
        titles = ["color", "result"]
        image.to_image().save(ft.join(results_dir, f"{on_sun_name}_background_subtraction_{titles[i]}.png"))
    hotspot_vis_image.to_image().save(ft.join(results_dir, f"{on_sun_name}_hotspot.png"))
    for i, image in enumerate(crosssec_vis_images):
        titles = ["vis", "horizontal", "vertical"]
        image.to_image().save(ft.join(results_dir, f"{on_sun_name}_crosssection_{titles[i]}.png"))
    moments_vis_images[0].to_image().save(ft.join(results_dir, f"{on_sun_name}_centroid.png"))
    enclosed_energy_vis_images[0].to_image().save(ft.join(results_dir, f"{on_sun_name}_enclosed_energy.png"))


if __name__ == "__main__":
    data_dir = ft.join(opencsp_settings["opencsp_root_path"]["experiment_dir"], "2_Data", "BCS_Data_sorted")
    process_dir = ft.join(opencsp_settings["opencsp_root_path"]["experiment_dir"], "3_Process", "BCS_Data_sorted")
    image_path_name_exts = it.image_files_in_directory(data_dir, recursive=True)
    time_dirs = sorted(list(set([get_parent_dir(image_path_name_ext) for image_path_name_ext in image_path_name_exts])))

    doi = opencsp_settings["opencsp_root_path"]["data_of_interest"]
    cross_section_dir = ft.join(process_dir, f"{doi}_Cross_Sections")
    if ft.directory_exists(cross_section_dir):
        shutil.rmtree(cross_section_dir)
    for time_dirname in time_dirs:
        contained_dirs = ft.directories_in_directory(ft.join(data_dir, time_dirname))

        # make sure we have data of interest
        on_sun_dir = ft.join(time_dirname, doi)
        if doi not in contained_dirs:
            if f"{doi}Hel" in contained_dirs:
                ft.rename_directory(ft.join(data_dir, time_dirname, f"{doi}Hel"), ft.join(data_dir, time_dirname, doi))
            else:
                lt.info(f"{time_dirname=} doesn't have any {doi} data. ({contained_dirs=})")
                continue
        on_sun_dir = ft.join(data_dir, on_sun_dir)
        on_sun_images = it.image_files_in_directory(on_sun_dir)
        if len(on_sun_images) == 0:
            lt.info(f"{time_dirname=} has an empty on-sun directory {on_sun_dir}")
            continue

        # check for corresponding nosun data
        no_sun_dir = None
        possible_no_sun_dirnames = [f"NoSun{doi}", f"NoSun{doi}Hel", "NoSunSF"]
        for possible_no_sun_dirname in possible_no_sun_dirnames:
            if possible_no_sun_dirname in contained_dirs:
                no_sun_dir = ft.join(time_dirname, possible_no_sun_dirname)
        if no_sun_dir is None:
            lt.info(f"{time_dirname=} has no {possible_no_sun_dirnames} directories ({contained_dirs=})")
        else:
            no_sun_dir = ft.join(data_dir, no_sun_dir)

        # limit on-sun images to those with a consistent gain
        inconsistent_gain_images: list[str] = []
        on_sun_gains = it.get_exif_value(on_sun_dir, on_sun_images)
        target_on_sun_gain = on_sun_gains[-1]
        for on_sun_image, on_sun_gain in list(zip(on_sun_images, on_sun_gains)):
            if np.abs(on_sun_gain - target_on_sun_gain) > 10:
                on_sun_images.remove(on_sun_image)
                inconsistent_gain_images.append(on_sun_image)

        # Limit on-sun images to those within 2 seconds of each other, to reduce
        # blur caused by the motion of the sun.
        outside_time_range_images: list[str] = []
        target_time = get_time_from_image_name(on_sun_images[0])
        for on_sun_image in copy.copy(on_sun_images):
            image_time = get_time_from_image_name(on_sun_image)
            if (image_time - target_time).total_seconds() >= 2:
                on_sun_images.remove(on_sun_image)
                outside_time_range_images.append(on_sun_image)

        # limit the no-sun images to those with a gain near the on-sun data
        no_sun_images = None
        if no_sun_dir is not None:
            no_sun_images = it.image_files_in_directory(no_sun_dir)
            no_sun_gains = it.get_exif_value(no_sun_dir, no_sun_images)
            target_no_sun_gain = no_sun_gains[-1]
            for no_sun_image, no_sun_gain in list(zip(no_sun_images, no_sun_gains)):
                if np.abs(no_sun_gain - target_no_sun_gain) > 10:
                    no_sun_images.remove(no_sun_image)
        if no_sun_images is None or len(no_sun_images) == 0:
            lt.info(f"{time_dirname=} has an empty no-sun directory or has no no-sun directory")
            no_sun_dir = None

        # build the results for this directory
        lt.info(f"Processing images for {time_dirname}")
        lt.info(
            f"\ton_sun_images: {len(on_sun_images)}, "
            + f"off_sun_images: {0 if no_sun_images is None else len(no_sun_images)}, "
            + f"gain images rejected: {len(inconsistent_gain_images)}, "
            + f">2s images rejected: {len(outside_time_range_images)}"
        )
        results_dir = ft.join(cross_section_dir, time_dirname)
        ft.create_directories_if_necessary(results_dir)
        process_images(on_sun_dir, no_sun_dir, results_dir, on_sun_images=on_sun_images, no_sun_images=no_sun_images)

    # build the powerpoint
    ppt = rcpp.RenderControlPowerpointPresentation()
    for time_dirname in ft.directories_in_directory(cross_section_dir):
        dir = ft.join(cross_section_dir, time_dirname)
        result_image_name_exts = it.image_files_in_directory(dir)

        slide = ps.PowerpointSlide.template_content_grid(2, 2)
        slide.set_title(time_dirname)
        slide.add_image(
            ppi.PowerpointImage(
                ft.join(dir, list(filter(lambda f: "_original" in f, result_image_name_exts))[0]), caption="Original"
            )
        )
        slide.add_image(
            ppi.PowerpointImage(
                ft.join(dir, list(filter(lambda f: "_subtraction_result" in f, result_image_name_exts))[0]),
                caption="Background Color Subtraction",
            )
        )
        slide.add_image(
            ppi.PowerpointImage(
                ft.join(dir, list(filter(lambda f: "_centroid" in f, result_image_name_exts))[0]), caption="Centroid"
            )
        )
        slide.add_image(
            ppi.PowerpointImage(
                ft.join(dir, list(filter(lambda f: "_enclosed_energy" in f, result_image_name_exts))[0]),
                caption="Enclosed Energy",
            )
        )
        slide.save_and_bake()
        ppt.add_slide(slide)

        slide = ps.PowerpointSlide.template_content_grid(2, 2)
        slide.set_title(time_dirname)
        slide.add_image(
            ppi.PowerpointImage(
                ft.join(dir, list(filter(lambda f: "crosssection_vis" in f, result_image_name_exts))[0]),
                caption="Hotspot",
            ),
            index=0,
        )
        for result_image_name_ext in result_image_name_exts:
            if "no_sun_avg" in result_image_name_ext:
                ppt_img = ppi.PowerpointImage(ft.join(dir, result_image_name_ext), caption=result_image_name_ext)
                slide.add_image(ppt_img, index=1)
        slide.add_image(ft.join(dir, list(filter(lambda f: "_horizontal" in f, result_image_name_exts))[0]), index=2)
        slide.add_image(ft.join(dir, list(filter(lambda f: "_vertical" in f, result_image_name_exts))[0]), index=3)
        slide.save_and_bake()
        ppt.add_slide(slide)

    ppt.save(ft.join(process_dir, f"{doi}_Cross_Sections.pptx"), overwrite=True)
