# SimpleGallery

The program utilizes Vulkan to render 2D drawings in 3D space.  The user has limited movement (moving sideways, zooming in and out, and rotating the camera) to observe the drawings.  All the drawings showcased within the program were drawn by the creator of this program.

## Prerequisites

Vulkan must be installed on the system attempting to use the program.

## Installation

Build the project in Visual Studio to create the executable.  The executable itself is plug-and-play.  However, the "textures" and "shaders" folders must be within the same directory as the executable.

## Usage

Double-click on the executable to run.  Once the program is running, the following inputs affect the program accordingly:

W = Zoom into the images at an angle
S = Zoom away from the images at an angle
A = Move the camera left
D = Move the camera right
Q = Rotate the camera counter-clockwise
E = Rotate the camera clockwise
R = Reset the camera angle
T = Reset the depth of the camera
F = Stop the hovering movement of the images
G = Resume the hovering movement of the images
ESC = Quit the program

## Built with

-Visual Studio (2017): the environment the program was written in
-Vulkan SDK: low-level graphics API

## Authors

-Marc Liander C. Reyes - creator of the project and drawings used in the project

## Acknowledgements

-Thanks to the authors of the following libraries:
	-OpenGL Mathematics (GLM)
	-Graphics Library Framework (GLFW)
-Special thanks to the author of the vulkan-tutorial.com website, where the majority of the code of this project was based off of.
