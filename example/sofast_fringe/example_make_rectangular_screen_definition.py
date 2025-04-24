"""Generates a rectangular SOFAST screen shape definition file.

This module defines a function to create and save a rectangular screen shape
definition for use in SOFAST applications. The screen dimensions, name, and
output filename are specified within the function. The generated definition
file can be utilized in optical simulations and analyses involving the SOFAST
framework.

Usage:
    To execute the script and generate the screen definition file, run the
    module directly. The output file will be saved in the current working
    directory.
"""

# Chat GPT 4o-mini assisted with the creation of docstrings

import os
from os.path import join, dirname, exists

from opencsp.app.sofast.lib.DisplayShape import DisplayShape


def example_make_rectangular_screen_definition():
    """
    Creates and saves a rectangular screen shape definition for SOFAST.

    This function defines the dimensions of a rectangular screen and creates
    a corresponding `DisplayShape` object. The screen's width and height are
    specified in meters, along with a name for the display. The resulting
    screen shape definition is saved as an HDF5 file in the current working
    directory.

    Notes
    -----
    The screen dimensions are hardcoded in the function. Edit the USER INPUT section
    of the code to make a new definition file:

        - Width (screen_x): 0.3 meters
        - Height (screen_y): 0.2 meters
        - Display name: 'LCD monitor laptop'
        - Output filename: 'display_LCD_monitor.h5'

    The generated HDF5 file can be used in subsequent SOFAST analyses to
    represent the specified screen shape.

    Example
    -------
    To generate the screen definition file, simply call the function:

    >>> example_make_rectangular_screen_definition()

    The output file will be saved in the current working directory.
    """
    # USER INPUT: Define the screen width and height, name, and filename
    screen_x = 0.3  # meters
    screen_y = 0.2  # meters
    save_file_name = 'display_LCD_monitor.h5'
    name = 'LCD monitor laptop'
    # ##################################################################

    # Make save directory if needed
    save_path = join(dirname(__file__), 'data/output/rectangular_screen_definition')
    save_file_path = join(save_path, save_file_name)
    if not exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Create the display object
    grid_data = {"screen_model": 'rectangular2D', "screen_x": screen_x, "screen_y": screen_y}
    display = DisplayShape(grid_data, name)

    # Save the display object
    print('Saved data to:', save_file_path)
    display.save_to_hdf(save_file_path)


if __name__ == '__main__':
    example_make_rectangular_screen_definition()
