import dataclasses


@dataclasses.dataclass
class ErrorSources():
    """ Each of the sources of error enumerated in this class affects the
    accuracy of a SOFAST analysis. The actual quantity of changes to be
    simulated aren't captured here, but rather the _degree_ of those changes. """
    camera_lens: float = 0
    """ Amount to modify the camera intrinsics and distortion coefficients
    by, where:
        0 = no change, 1 = maximum conceivable error """
    camera_model: float = 0
    """ Amount of irregular distortion to add to all images, where:
        0 = no change, 1 = maximum conceivable error """
    mirror_distortion: float = 0
    """ Scale of screen pixels to modify the reflected images by, where:
        0 = no change, 1 = maximum conceivable error """
    mirror_distortion_feature_size: float = 0
    """ Scale of the facet distortion features, where:
        0 = 1mm and 1 = whole facet. """
    camera_resolution: float = 0
    """ Amount to modify the camera angular resolution by, where:
        0 = no change and 1 = minimum conceivable resolution. """
    projector_screen_minimum_resolution: float = 0
    """ Amount to modify the minimum resolution (aka maximum screen area) of a resolution pixel by. """
    mirror_focus_point: float = 0.5
    """ Where on the mirror the sofast camera is focused, where:
        0 = upper left corner, 0.5 = center, and 1 = lower right corner. """
    perceived_intensity: float = 0
    """ Amount to modify the perceived intensity of the projector by the
    SOFAST camera by, where:
        0 = no change, 1 = maximum conceivable error """
    projector_non_uniformity: float = 0
    """ Amount of non-uniform projector intensity to apply to the screen,
    where:
        0 = no change, 1 = maximum conceivable error """
    gain_nose: float = 0
    """ Amount of noise in the camera, where:
        0 = no change, 1 = maximum conceivable error """
    ambient_light_floor: float = 0
    """ Changes to the minimum intensity value provided by the projector that
    the SOFAST camera can perceive, where:
        0 = no change, 1 = maximum conceivable error """
    perceived_intensity_ceiling: float = 0
    """ Changes to the maximum intensity value that the SOFAST camera can
    perceive, where:
        0 = no change, 1 = maximum conceivable error """
    camera_temperature_effects: float = 0
    """ Amount of noise added due to camera temperature. This is applied to
    subsequent measurement images based on their timestamp, to simulate the
    camera getting hotter with each image. 0 = no change and 1 = maximum
    conceivable error. """
    non_flat_screen: float = 0
    """ Amount to modify the flatness (z-axis uniformity) of the screen, both
    in terms of scale and frequency, where:
        0 = no change, 1 = maximum conceivable error """
    unaligned_screen_border_aruco_markers: float = 0
    """ Amount to adjust the 'measured' 0 values for the middle 4 aruco
    markers, where:
        0 = no change, 1 = maximum conceivable error """
    aruco_marker_distance_measurements: float = 0
    """ Amount to modify the measured distances between aruco marker point
    pairs, where:
        0 = no change, 1 = maximum conceivable error """
    aruco_marker_calibration_images_distortion: float = 0
    """ Amount to distort the aruco marker calibration images (all images
    distorted equally). This is to simulate errors in camera lens distortion
    w.r.t. aruco marker calibration images. 0 = no change and 1 = maximum
    conceivable error. """
    system_settlement: float = 0
    """ Amount to adjust camera-to-screen stability for system settlement.
    This simulates positional changes over time due to gravity sag, etc, such
    that the current position does not match the calibrated position. 0 = no
    change and 1 = maximum conceivable error. """
    system_temperature_swings: float = 0
    """ Amount to adjust the system measurements by, where:
        0 = no change, 1 = maximum conceivable error
    This simulates the expansion/contraction of the entire rigging structure due
    to temperature swings, and so also includes angular changes as well. """
    mirror_distance: float = 0
    """ Amount to adjust the mirror-to-screen measurement, where:
        0 = no change, 1 = maximum conceivable error """
    projector_temperature_swings: float = 0
    """ Amount to adjust the effect of temperature on the projector, after it
    has had time to reach its steady state, where:
        0 = no change, 1 = maximum conceivable error """
    assumed_mirror_shape: float = 0
    """ Amount to change the expected mirror shape, where:
        0 = no change, 1 = maximum conceivable error """
