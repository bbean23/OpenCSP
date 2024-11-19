import os
import unittest
from opencsp.common.lib.cv.spot_analysis.image_processor.HotspotImageProcessor import HotspotImageProcessor

import opencsp.common.lib.tool.file_tools as ft


class TestHotspotImageProcessor(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "HotspotImageProcessor")
        self.out_dir = os.path.join(path, "data", "output", "HotspotImageProcessor")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

    def test_internal_shapes_int(self):
        processor = HotspotImageProcessor(desired_shape=21)
        # assume that factor = 2
        # start = odd(21 * 2) = odd(42) = 43
        # assume that reduction = min(10, 21 / 3) = 6
        expected = [43, 37, 31, 25, 21]
        self.assertEqual(expected, processor.internal_shapes)

    def test_internal_shapes_tuple(self):
        processor = HotspotImageProcessor(desired_shape=(3, 21))
        # assume that factor = 2
        # start_x = odd(21 * 2) = odd(42) = 43
        # start_y = odd(3 * 2) = odd(6) = 7
        # assume that reduction = min(10, 7 / 3) = 1
        expected_x = [43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21]
        expected_y = [7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        expected = [tuple(v) for v in zip(expected_y, expected_x)]
        self.assertEqual(expected, processor.internal_shapes)


if __name__ == '__main__':
    unittest.main()
